"""
extract_features.py — Step 3 of the SMART pipeline.

Runs the trained DEFT + fusion front-end over EVERY frame of every unit and
writes one feature matrix per unit:

    <out_dir>/<unit>_feats<suffix>.npz   ->  feats (n_frames, FEAT_DIM)

The suffix encodes the configuration ('', '_resnet', '_effnet', '_clip',
'_rgbonly', '_flowonly') so several feature sets can coexist in one folder and
evaluate.py can pick between them with --feat_suffix. That is how the
feature-level ablations are run without touching the graph code.

The classifier head is not loaded here — only the embedding path is used.

Usage
-----
    python extract_features.py \
        --checkpoint checkpoints/adl_deft_fusion_resnet.pt \
        --rgb_root  /path/to/ADL/Frames \
        --flow_root /path/to/ADL/OpticalFlow \
        --data_dir  artifacts/adl \
        --out_dir   artifacts/adl
"""

from tools.imports import *  # noqa: F403
from model import SMARTFrontEnd, build_transforms
from dataset import rgb_dir, flow_dir, frame_name, num_frames


def main():
    ap = argparse.ArgumentParser(description="SMART — extract per-frame features")
    ap.add_argument("--checkpoint", required=True, help="written by train.py")
    ap.add_argument("--rgb_root", required=True)
    ap.add_argument("--flow_root", default=None)
    ap.add_argument("--data_dir", required=True, help="folder written by dataset.py")
    ap.add_argument("--out_dir", default=None, help="default = --data_dir")
    ap.add_argument("--units", nargs="*", default=None)
    ap.add_argument("--dataset", default=None, choices=["adl", "charades"],
                    help="default = whatever the checkpoint recorded")
    ap.add_argument("--feat_suffix", default=None,
                    help="override the suffix recorded in the checkpoint")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--force", action="store_true", help="re-extract even if cached")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
    cfg = ckpt["config"]
    dataset = args.dataset or cfg["dataset"]
    suffix = args.feat_suffix if args.feat_suffix is not None else cfg["feat_suffix"]

    # a single-modality run overrides the backbone suffix
    if cfg["modality"] in ("rgbonly", "flowonly"):
        suffix = f"_{cfg['modality']}"

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir or args.data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    units = args.units or sorted(p.stem.replace("_data", "")
                                 for p in data_dir.glob("*_data.npz"))

    banner(f"SMART extract — {dataset} | backbone={cfg['backbone']} "
           f"| modality={cfg['modality']} | suffix='{suffix}'")
    print(f"device={DEVICE} | units={units}")

    model = SMARTFrontEnd(backbone=cfg["backbone"], num_classes=None,
                          modality=cfg["modality"], device=DEVICE,
                          finetune_block=cfg["finetune_block"],
                          warp_flow=cfg["warp_flow"],
                          clip_name=cfg.get("clip_name"))
    model.load_trainable(ckpt["state"], strict_backbone=bool(cfg["finetune_block"]))
    if model.deft:
        model.deft.eval()
    if model.fusion:
        model.fusion.eval()

    tf = build_transforms(cfg["backbone"], cfg["img_size"])
    need_flow = cfg["modality"] != "rgbonly"
    feat_dim = model.feat_dim

    def load_img(path):
        return tf(Image.open(path).convert("RGB")).unsqueeze(0)

    @torch.inference_mode()
    def extract(unit):
        n = num_frames(dataset, args.rgb_root, unit)
        rd = rgb_dir(dataset, args.rgb_root, unit)
        fd = flow_dir(dataset, args.flow_root, unit) if need_flow else None
        feats = np.zeros((n, feat_dim), np.float32)

        for b in tqdm(range(0, n, args.batch), desc=unit, unit="batch"):
            idxs = range(b, min(b + args.batch, n))
            rgb = torch.cat([load_img(rd / frame_name(dataset, i)) for i in idxs], 0).to(DEVICE)
            flow = None
            if need_flow:
                flow = torch.cat([load_img(fd / frame_name(dataset, i)) for i in idxs], 0).to(DEVICE)

            z = model.embed(rgb, flow)
            if b == 0:
                print(f"  [check] rgb{tuple(rgb.shape)} -> embedding{tuple(z.shape)}")
            feats[b:b + z.shape[0]] = z.cpu().numpy()
        return feats

    for u in units:
        out = out_dir / f"{u}_feats{suffix}.npz"
        if out.exists() and not args.force:
            print(f"{u}: cached (skip) — pass --force to re-extract")
            continue
        feats = extract(u)
        np.savez_compressed(out, feats=feats)
        print(f"  [check] {u} feats: shape={feats.shape} mean={feats.mean():.3f} "
              f"std={feats.std():.3f} min={feats.min():.3f} max={feats.max():.3f}")
        print(f"saved -> {out}")

    print(f"\ndone -> {out_dir.resolve()}")
    print(f"next: python evaluate.py --data_dir {out_dir} --feat_suffix '{suffix}'")


if __name__ == "__main__":
    main()
