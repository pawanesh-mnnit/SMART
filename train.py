from tools.imports import *
from model import ASL, SMARTFrontEnd, build_transforms, BACKBONE_SUFFIX
from dataset import rgb_dir, flow_dir, frame_name


def load_img(path, tf):
    return tf(Image.open(path).convert("RGB")).unsqueeze(0)


def load_batch(items, dataset, rgb_root, flow_root, tf, device, need_flow=True):
    """items: list of (unit, frame_idx). Units may be mixed inside one batch."""
    rgb = torch.cat([
        load_img(rgb_dir(dataset, rgb_root, u) / frame_name(dataset, i), tf)
        for u, i in items], 0).to(device)
    flow = None
    if need_flow:
        flow = torch.cat([
            load_img(flow_dir(dataset, flow_root, u) / frame_name(dataset, i), tf)
            for u, i in items], 0).to(device)
    return rgb, flow


def main():
    ap = argparse.ArgumentParser(description="SMART — train DEFT + fusion on seed frames")
    ap.add_argument("--dataset", required=True, choices=["adl", "charades"])
    ap.add_argument("--rgb_root", required=True)
    ap.add_argument("--flow_root", default=None)
    ap.add_argument("--data_dir", required=True,
                    help="folder written by dataset.py (class_map.pkl + *_data.npz)")
    ap.add_argument("--units", nargs="*", default=None)
    ap.add_argument("--save_path", required=True)

    ap.add_argument("--backbone", default="resnet50",
                    choices=["resnet18", "resnet50", "efficientnet_b0", "clip"])
    ap.add_argument("--modality", default="fusion",
                    choices=["fusion", "rgbonly", "flowonly"])
    ap.add_argument("--clip_name", default="openai/clip-vit-base-patch32")
    ap.add_argument("--finetune_block", default=None,
                    help="e.g. layer4 — unfreeze one ResNet block (ResNet only)")
    ap.add_argument("--warp_flow", action="store_true",
                    help="apply the DEFT warp to the flow stream as well")

    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3, help="fusion + head")
    ap.add_argument("--deft_lr", type=float, default=5e-3)
    ap.add_argument("--backbone_lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--loss", default="asl", choices=["asl", "bce"])
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.modality != "rgbonly" and not args.flow_root:
        ap.error("--flow_root is required unless --modality rgbonly")

    set_seed(args.seed)
    data_dir = Path(args.data_dir)
    cmap = pickle.load(open(data_dir / "class_map.pkl", "rb"))
    C = cmap["C"]

    units = args.units or sorted(p.stem.replace("_data", "")
                                 for p in data_dir.glob("*_data.npz"))

    banner(f"SMART train — {args.dataset} | backbone={args.backbone} | modality={args.modality}")
    print(f"device={DEVICE} | units={units} | C={C}")

    # ---- pool the seed frames of every unit -------------------------------
    train_items, TARG = [], {}
    for u in units:
        d = np.load(data_dir / f"{u}_data.npz", allow_pickle=True)
        TARG[u] = d["targets"]
        for i in np.where(d["seeds"])[0]:
            train_items.append((u, int(i)))

    if not train_items:
        raise RuntimeError("no seed frames found — run dataset.py first")

    ck_y = np.stack([TARG[u][i] for u, i in train_items])
    print(f"training on {len(train_items)} seed frames")
    print(f"  [check] seed labels: shape={ck_y.shape} "
          f"pos/frame mean={ck_y.sum(1).mean():.2f} "
          f"class coverage={(ck_y.sum(0) > 0).sum()}/{C}")

    # ---- model ------------------------------------------------------------
    model = SMARTFrontEnd(backbone=args.backbone, num_classes=C,
                          modality=args.modality, device=DEVICE,
                          finetune_block=args.finetune_block,
                          warp_flow=args.warp_flow, clip_name=args.clip_name)
    tf = build_transforms(args.backbone, args.img_size)
    criterion = ASL() if args.loss == "asl" else nn.BCEWithLogitsLoss()
    need_flow = args.modality != "rgbonly"

    # DEFT.fc is lazily built — one warm-up forward so its parameters exist
    # before the optimizer is constructed, otherwise they never get updated.
    if model.deft is not None:
        with torch.no_grad():
            r, _ = load_batch([train_items[0]], args.dataset, args.rgb_root,
                              args.flow_root, tf, DEVICE, need_flow=False)
            model.deft.theta(r)

    deft_p, hf_p, bb_p = model.trainable_parameters()
    groups = [g for g in [
        {"params": deft_p, "lr": args.deft_lr},
        {"params": hf_p, "lr": args.lr},
        {"params": bb_p, "lr": args.backbone_lr},
    ] if g["params"]]
    opt = torch.optim.Adam(groups, weight_decay=args.weight_decay)
    n_par = sum(p.numel() for g in groups for p in g["params"])
    print(f"trainable tensors: {sum(len(g['params']) for g in groups)} | params: {n_par:,}")

    # ---- train ------------------------------------------------------------
    rng = np.random.default_rng(args.seed)
    items = np.array(train_items, dtype=object)
    if model.deft:
        model.deft.train()
    if model.fusion:
        model.fusion.train()
    model.head.train()

    for epoch in range(args.epochs):
        perm = rng.permutation(len(items))
        tot, nb = 0.0, 0
        pbar = tqdm(range(0, len(perm), args.batch), desc=f"epoch {epoch + 1}/{args.epochs}")
        for b in pbar:
            sel = [(u, int(i)) for u, i in items[perm[b:b + args.batch]]]
            rgb, flow = load_batch(sel, args.dataset, args.rgb_root, args.flow_root,
                                   tf, DEVICE, need_flow)
            yb = torch.tensor(np.stack([TARG[u][i] for u, i in sel]),
                              dtype=torch.float32, device=DEVICE)

            logits = model(rgb, flow)
            loss = criterion(logits, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()

            tot += float(loss.item())
            nb += 1
            if hasattr(pbar, "set_postfix"):
                pbar.set_postfix(loss=f"{tot / max(nb, 1):.4f}")

            if epoch == 0 and b == 0:
                print(f"\n  [check] shape chain: rgb{tuple(rgb.shape)} "
                      f"-> logits{tuple(logits.shape)} | first loss={loss.item():.4f}")
        print(f"epoch {epoch + 1}/{args.epochs} | mean loss = {tot / max(nb, 1):.4f}")

    # ---- save -------------------------------------------------------------
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state": model.state_dict_trainable(),
        "config": {
            "backbone": args.backbone,
            "modality": args.modality,
            "clip_name": args.clip_name,
            "warp_flow": args.warp_flow,
            "finetune_block": args.finetune_block,
            "img_size": args.img_size,
            "feat_dim": model.feat_dim,
            "num_classes": C,
            "feat_suffix": BACKBONE_SUFFIX.get(args.backbone, ""),
            "dataset": args.dataset,
        },
    }, save_path)
    print(f"\nsaved -> {save_path.resolve()}")
    print("next: python extract_features.py --checkpoint", save_path)


if __name__ == "__main__":
    main()
