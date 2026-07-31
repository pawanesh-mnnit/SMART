from tools.imports import *  # noqa: F403

__all__ = ["DEFT", "CoAttnFusion", "ASL", "make_backbone",
           "build_transforms", "SMARTFrontEnd", "BACKBONE_SUFFIX"]


# Feature-file suffix per configuration — evaluate.py reads features via this.
BACKBONE_SUFFIX = {
    "resnet18": "",
    "resnet50": "_resnet",
    "efficientnet_b0": "_effnet",
    "clip": "_clip",
}


# --------------------------------------------------------------------------
# DEFT
# --------------------------------------------------------------------------

class DEFT(nn.Module):
    """Deformable Egocentric Focus Transform: bounded STN + Gaussian centre prior.

    The localisation net predicts a 6-vector delta in (-1, 1) via tanh; the
    actual affine parameters are `base + range * delta`, i.e. identity plus a
    strictly bounded deviation. Initialising the final bias to zero means the
    transform starts exactly at identity, so training never begins from a
    destructive warp.
    """

    def __init__(self, sigma_g=0.5, center=(0.0, 0.0)):
        super().__init__()
        self.loc = nn.Sequential(
            nn.Conv2d(3, 8, 7), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(8, 10, 5), nn.ReLU(True), nn.MaxPool2d(2, 2),
        )
        self.fc = None                      # built lazily on the first forward
        self.sigma_g = sigma_g
        self.register_buffer("center", torch.tensor(center).float())
        self.register_buffer("base", torch.tensor([1, 0, 0, 0, 1, 0.]))
        # allowed deviation from identity for [a, b, tx, c, d, ty]
        self.register_buffer("rng_", torch.tensor([0.3, 0.3, 0.5, 0.3, 0.3, 0.5]))

    def _build_fc(self, flat):
        self.fc = nn.Sequential(
            nn.Linear(flat, 32), nn.ReLU(True), nn.Linear(32, 6)
        ).to(self.center.device)
        nn.init.normal_(self.fc[-1].weight, std=1e-2)   # tiny random -> slightly off identity
        self.fc[-1].bias.data.zero_()                   # bias 0 -> bounded theta starts AT identity

    def mask(self, B, H, W, dev):
        """Fixed isotropic Gaussian centre prior, broadcast over the batch."""
        ys = torch.linspace(-1, 1, H, device=dev)
        xs = torch.linspace(-1, 1, W, device=dev)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        cx, cy = self.center
        g = torch.exp(-((gx - cx) ** 2 + (gy - cy) ** 2) / (2 * self.sigma_g ** 2))
        return g.view(1, 1, H, W).expand(B, 1, H, W)

    def theta(self, x):
        f = self.loc(x).flatten(1)
        if self.fc is None:
            self._build_fc(f.shape[1])
        delta = torch.tanh(self.fc(f))       # (-1, 1), bounded
        th = self.base + self.rng_ * delta   # identity +/- allowed range
        return th.view(-1, 2, 3)

    def warp(self, x, th):
        grid = Fnn.affine_grid(th, x.size(), align_corners=False)
        return Fnn.grid_sample(x, grid, align_corners=False)

    def forward(self, rgb, flow=None, warp_flow=False):
        """Returns (rgb_focused, flow_focused_or_passthrough).

        The flow stream shares the RGB-estimated theta when `warp_flow` is set;
        by default flow passes through untouched so the warp is only ever
        supervised through the appearance branch.
        """
        th = self.theta(rgb)
        rw = self.warp(rgb, th)
        g = self.mask(rw.size(0), rw.size(2), rw.size(3), rw.device)
        rgb_out = rw * g

        if flow is None:
            return rgb_out, None
        if warp_flow:
            fw = self.warp(flow, th)
            return rgb_out, fw * g
        return rgb_out, flow


# --------------------------------------------------------------------------
# Cross-modal co-attention fusion
# --------------------------------------------------------------------------

class CoAttnFusion(nn.Module):
    """Cross-modal co-attention.

    a = <normalize(f_rgb), normalize(f_flow)> is a per-sample agreement scalar.
    When the two modalities agree the cross-injection is strong; when they
    disagree (camera motion without hand motion, say) it fades out. The final
    Linear(2d -> d) is learned jointly with DEFT on the seed frames.
    """

    def __init__(self, d=512):
        super().__init__()
        self.proj = nn.Linear(2 * d, d)

    def forward(self, frgb, fflow):
        a = (Fnn.normalize(frgb, dim=1) * Fnn.normalize(fflow, dim=1)).sum(1, keepdim=True)
        r_rgb = frgb + a * fflow
        r_flow = fflow + a * frgb
        return self.proj(torch.cat([r_rgb, r_flow], dim=1))


# --------------------------------------------------------------------------
# Asymmetric Loss
# --------------------------------------------------------------------------

class ASL(nn.Module):
    """Asymmetric Loss for multi-label classification.

    Negatives are focused harder than positives (gamma_neg > gamma_pos) and are
    additionally probability-shifted by `clip`, discarding very easy negatives
    outright. On per-frame action labels the negatives outnumber positives by
    two orders of magnitude, so plain BCE collapses to predicting nothing.
    """

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8):
        super().__init__()
        self.gn, self.gp, self.clip, self.eps = gamma_neg, gamma_pos, clip, eps

    def forward(self, logits, y):
        xs_pos = torch.sigmoid(logits)
        xs_neg = 1 - xs_pos
        if self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg
        pt = xs_pos * y + xs_neg * (1 - y)
        gamma = self.gp * y + self.gn * (1 - y)
        loss = loss * (1 - pt) ** gamma
        return -loss.sum() / y.shape[0]


# --------------------------------------------------------------------------
# Backbones
# --------------------------------------------------------------------------

def make_backbone(name="resnet18", device="cpu", finetune_block=None, clip_name=None):
    """Build a frozen image encoder. Returns (forward_fn, feat_dim, modules).

    `finetune_block` (e.g. 'layer4') unfreezes one ResNet block so the encoder
    can adapt slightly to the seed frames. The module is still kept in eval()
    mode so BatchNorm running statistics stay frozen — with only a few hundred
    seed frames, updating them destabilises training.
    """
    name = name.lower()

    if name in ("resnet18", "resnet50"):
        ctor = tv.models.resnet18 if name == "resnet18" else tv.models.resnet50
        weights = (tv.models.ResNet18_Weights.IMAGENET1K_V1 if name == "resnet18"
                   else tv.models.ResNet50_Weights.IMAGENET1K_V2)
        m = ctor(weights=weights)
        feat_dim = m.fc.in_features
        m.fc = nn.Identity()
        for p in m.parameters():
            p.requires_grad = False
        if finetune_block and hasattr(m, finetune_block):
            for p in getattr(m, finetune_block).parameters():
                p.requires_grad = True
        m = m.eval().to(device)
        return (lambda x: m(x)), feat_dim, [m]

    if name in ("efficientnet_b0", "effnet"):
        m = tv.models.efficientnet_b0(
            weights=tv.models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        feat_dim = m.classifier[-1].in_features            # 1280
        body = nn.Sequential(m.features, m.avgpool).to(device).eval()
        for p in body.parameters():
            p.requires_grad = False
        return (lambda x: body(x).flatten(1)), feat_dim, [body]

    if name == "clip":
        from transformers import CLIPModel
        clip_name = clip_name or "openai/clip-vit-base-patch32"
        clip = CLIPModel.from_pretrained(clip_name).to(device).eval()
        for p in clip.parameters():
            p.requires_grad = False
        feat_dim = clip.config.projection_dim               # 512 for ViT-B/32
        return (lambda x: clip.get_image_features(pixel_values=x)), feat_dim, [clip]

    raise ValueError(f"unknown backbone: {name}")


def build_transforms(backbone="resnet18", img_size=224):
    """Normalisation statistics matching the chosen encoder."""
    if backbone == "clip":
        mean = [0.48145466, 0.45782750, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]
    else:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


# --------------------------------------------------------------------------
# Assembled front-end
# --------------------------------------------------------------------------

class SMARTFrontEnd(nn.Module):
    """DEFT -> frozen encoder(s) -> fusion -> (optional) classifier head.

    modality:
        'fusion'   RGB (through DEFT) + flow, combined by CoAttnFusion
        'rgbonly'  RGB through DEFT only, no fusion
        'flowonly' flow only, no DEFT, no fusion
    """

    def __init__(self, backbone="resnet18", num_classes=None, modality="fusion",
                 device="cpu", finetune_block=None, warp_flow=False, clip_name=None):
        super().__init__()
        self.modality = modality
        self.warp_flow = warp_flow
        self.device_str = device

        self.encode, self.feat_dim, self._enc_modules = make_backbone(
            backbone, device, finetune_block, clip_name)

        self.deft = DEFT().to(device) if modality != "flowonly" else None
        self.fusion = CoAttnFusion(self.feat_dim).to(device) if modality == "fusion" else None
        self.head = (nn.Linear(self.feat_dim, num_classes).to(device)
                     if num_classes else None)

    def embed(self, rgb, flow=None):
        """Frame(s) -> FEAT_DIM embedding. This is what gets written to disk."""
        if self.modality == "flowonly":
            return self.encode(flow)

        rgb_d, flow_d = self.deft(rgb, flow, warp_flow=self.warp_flow)
        if self.modality == "rgbonly":
            return self.encode(rgb_d)

        frgb = self.encode(rgb_d)
        fflow = self.encode(flow_d if self.warp_flow else flow)
        return self.fusion(frgb, fflow)

    def forward(self, rgb, flow=None):
        """Training path: embedding -> classifier logits."""
        z = self.embed(rgb, flow)
        return self.head(z)

    def trainable_parameters(self):
        """Parameter groups with their own learning rates.

        DEFT gets the highest LR (its gradient is the weakest in the chain,
        arriving through the frozen encoder), the unfrozen backbone block the
        lowest.
        """
        deft_p = [p for p in self.deft.parameters() if p.requires_grad] if self.deft else []
        hf_p = []
        if self.fusion:
            hf_p += [p for p in self.fusion.parameters() if p.requires_grad]
        if self.head:
            hf_p += [p for p in self.head.parameters() if p.requires_grad]
        bb_p = [p for m in self._enc_modules for p in m.parameters() if p.requires_grad]
        return deft_p, hf_p, bb_p

    def state_dict_trainable(self):
        """Only the pieces that are actually learned — a checkpoint is a few MB."""
        sd = {}
        if self.deft:
            sd["deft"] = self.deft.state_dict()
        if self.fusion:
            sd["fusion"] = self.fusion.state_dict()
        if self.head:
            sd["head"] = self.head.state_dict()
        bb = {}
        for i, m in enumerate(self._enc_modules):
            bb[str(i)] = {k: v for k, v in m.state_dict().items()}
        sd["backbone"] = bb
        return sd

    def load_trainable(self, sd, strict_backbone=False):
        if self.deft is not None and "deft" in sd:
            # DEFT.fc is lazily built; run one dummy pass first if it is missing
            if self.deft.fc is None:
                dummy = torch.zeros(1, 3, 224, 224, device=self.center_device())
                with torch.no_grad():
                    self.deft.theta(dummy)
            self.deft.load_state_dict(sd["deft"])
        if self.fusion is not None and "fusion" in sd:
            self.fusion.load_state_dict(sd["fusion"])
        if self.head is not None and "head" in sd:
            self.head.load_state_dict(sd["head"])
        if strict_backbone and "backbone" in sd:
            for i, m in enumerate(self._enc_modules):
                m.load_state_dict(sd["backbone"][str(i)])

    def center_device(self):
        return self.deft.center.device if self.deft else self.device_str
