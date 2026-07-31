from tools.imports import *  # noqa: F403
from tools.graph import make_windows


# --------------------------------------------------------------------------
# Dataset-specific path conventions
# --------------------------------------------------------------------------

def rgb_dir(dataset, rgb_root, unit):
    """Directory holding the RGB frames of one unit."""
    if dataset == "adl":
        return Path(rgb_root) / unit / f"Original_{unit}"
    return Path(rgb_root) / unit / "rgb"


def flow_dir(dataset, flow_root, unit):
    """Directory holding the optical-flow visualisations of one unit."""
    if dataset == "adl":
        return Path(flow_root) / unit / "viz"
    return Path(flow_root) / unit / "flow"


def frame_name(dataset, i):
    """0-based internal index -> on-disk filename.

    ADL frames are 5-digit and 0-indexed; Charades frames are 6-digit and
    1-indexed. Everything inside SMART uses the 0-based index.
    """
    return f"frame_{i:05d}.jpg" if dataset == "adl" else f"frame_{i + 1:06d}.jpg"


def num_frames(dataset, rgb_root, unit):
    return len(sorted(rgb_dir(dataset, rgb_root, unit).glob("frame_*.jpg")))


# --------------------------------------------------------------------------
# Annotations
# --------------------------------------------------------------------------

def load_annotations(dataset, anno_root=None, anno_csv=None, units=None):
    """Return {unit: DataFrame} with normalised columns.

    Normalised schema: action_id, action_name, start, end
      ADL       one CSV per participant: ActionLabel, ActionName, StartFrame, EndFrame
                (StartFrame/EndFrame 0-indexed, inclusive)
      Charades  one CSV for all videos:  video_id, Action_class, Action_name,
                StartFrame, EndFrame (1-indexed, inclusive)
    """
    out = {}

    if dataset == "adl":
        for u in units:
            path = Path(anno_root) / f"{u}_labeled.csv"
            if not path.exists():
                raise FileNotFoundError(f"annotation missing: {path}")
            df = pd.read_csv(path, encoding="utf-8-sig")
            df.columns = [c.strip() for c in df.columns]
            out[u] = pd.DataFrame({
                "action_id": df["ActionLabel"].astype(int),
                "action_name": df["ActionName"].astype(str),
                "start": df["StartFrame"].astype(int),        # 0-indexed inclusive
                "end": df["EndFrame"].astype(int),
            })
    else:
        df = pd.read_csv(anno_csv, encoding="utf-8-sig")      # utf-8-sig strips the BOM
        df.columns = [c.strip() for c in df.columns]
        vids = units if units else sorted(df["video_id"].unique().tolist())
        for v in vids:
            sub = df[df["video_id"] == v]
            out[v] = pd.DataFrame({
                "action_id": sub["Action_class"].astype(int),
                "action_name": sub["Action_name"].astype(str),
                "start": sub["StartFrame"].astype(int) - 1,   # 1-indexed -> 0-indexed
                "end": sub["EndFrame"].astype(int) - 1,
            })
    return out


def build_class_map(annos, include_background=False):
    """Global, contiguous class map over every unit's annotations."""
    alla = pd.concat(annos.values())[["action_id", "action_name"]]
    alla = alla.drop_duplicates("action_id").sort_values("action_id")
    id2name = dict(zip(alla.action_id, alla.action_name))
    ids = sorted(id2name)

    off = 1 if include_background else 0
    labelid2cls = {l: i + off for i, l in enumerate(ids)}
    names = {0: "background"} if include_background else {}
    for l, ci in labelid2cls.items():
        names[ci] = id2name[l]
    return dict(labelid2cls=labelid2cls, names=names, C=len(ids) + off)


def build_multihot(df, n_frames, cmap, include_background=False):
    """Per-frame multi-hot target matrix (n_frames, C).

    Segments are inclusive on both ends and OR-accumulate, so overlapping
    annotations produce frames with more than one active class — the
    concurrency that SMART is built to recover.
    """
    Y = np.zeros((n_frames, cmap["C"]), np.float32)
    for _, r in df.iterrows():
        ci = cmap["labelid2cls"][int(r.action_id)]
        s = max(0, int(r.start))
        e = min(int(r.end), n_frames - 1)
        if e >= s:
            Y[s:e + 1, ci] = 1.0
    if include_background:
        Y[Y[:, 1:].sum(1) == 0, 0] = 1.0
    return Y


# --------------------------------------------------------------------------
# Seeds
# --------------------------------------------------------------------------

def stratified_seeds(Y, frac, min_per_class, rng):
    """Pick ~`frac` of the labelled frames as seeds, >= min_per_class per class.

    Two passes: first guarantee every class that appears at all contributes at
    least min_per_class seeds (otherwise a rare class can never be recovered by
    propagation), then top up at random to reach the budget.
    """
    n, C = Y.shape
    labeled = np.where(Y.sum(1) > 0)[0]
    budget = max(int(round(frac * len(labeled))), 1)
    chosen = set()

    for c in range(C):
        pool = np.where(Y[:, c] > 0)[0]
        if len(pool):
            chosen.update(rng.choice(pool, min(min_per_class, len(pool)),
                                     replace=False).tolist())

    rem = [i for i in labeled if i not in chosen]
    if len(chosen) < budget and rem:
        chosen.update(rng.choice(rem, min(budget - len(chosen), len(rem)),
                                 replace=False).tolist())

    mask = np.zeros(n, dtype=bool)
    mask[list(chosen)] = True
    return mask


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def report_coverage(units, targets, seeds, cmap):
    """Per-unit and pooled seed counts per class — catches unrecoverable classes."""
    names = cmap["names"]
    pooled_seed = np.zeros(cmap["C"])
    pooled_lab = np.zeros(cmap["C"])
    per_seed = {u: targets[u][seeds[u]].sum(0) for u in units}
    per_lab = {u: (targets[u] > 0).sum(0) for u in units}

    print("\nclass coverage (seed frames / labelled frames)")
    print(f'{"class":<34}{"POOLED seed":>13}{"POOLED lab":>12}')
    for ci in range(cmap["C"]):
        pooled_seed[ci] = sum(per_seed[u][ci] for u in units)
        pooled_lab[ci] = sum(per_lab[u][ci] for u in units)
        if pooled_lab[ci] > 0:
            print(f"  {names[ci][:31]:<32}{int(pooled_seed[ci]):>13}{int(pooled_lab[ci]):>12}")

    zero = [names[ci] for ci in range(cmap["C"])
            if pooled_lab[ci] > 0 and pooled_seed[ci] == 0]
    print(f"\nclasses present: {int((pooled_lab > 0).sum())}/{cmap['C']}")
    print(f"present classes with ZERO pooled seeds (unrecoverable): {zero if zero else 'none'}")


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="SMART — build targets, windows and seeds")
    ap.add_argument("--dataset", required=True, choices=["adl", "charades"])
    ap.add_argument("--rgb_root", required=True,
                    help="ADL: <root>/Frames   Charades: <root>/RGB")
    ap.add_argument("--anno_root", default=None,
                    help="ADL only — folder of P_XX_labeled.csv (e.g. ADL/Labels)")
    ap.add_argument("--anno_csv", default=None,
                    help="Charades only — the single annotation CSV")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--units", nargs="*", default=None,
                    help="participants / video ids; default = all in the annotations")
    ap.add_argument("--window", type=int, default=100,
                    help="frames per graph window N (ADL 100, Charades 60)")
    ap.add_argument("--stride", type=int, default=None, help="default = N (non-overlapping)")
    ap.add_argument("--seed_frac", type=float, default=0.10)
    ap.add_argument("--min_per_class", type=int, default=2)
    ap.add_argument("--include_background", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.dataset == "adl" and not args.anno_root:
        ap.error("--anno_root is required for --dataset adl")
    if args.dataset == "charades" and not args.anno_csv:
        ap.error("--anno_csv is required for --dataset charades")

    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    units = args.units
    if args.dataset == "adl" and not units:
        units = sorted(p.stem.replace("_labeled", "")
                       for p in Path(args.anno_root).glob("*_labeled.csv"))

    banner(f"SMART dataset build — {args.dataset}")
    annos = load_annotations(args.dataset, args.anno_root, args.anno_csv, units)
    units = list(annos.keys())
    print(f"units ({len(units)}): {units}")

    cmap = build_class_map(annos, args.include_background)
    print(f"num classes C = {cmap['C']}")

    targets, seeds, windows = {}, {}, {}
    for u in units:
        d = rgb_dir(args.dataset, args.rgb_root, u)
        if not d.exists():
            raise FileNotFoundError(f"RGB frames missing: {d}")
        nf = num_frames(args.dataset, args.rgb_root, u)

        Y = build_multihot(annos[u], nf, cmap, args.include_background)
        W = make_windows(nf, args.window, args.stride or args.window)
        S = stratified_seeds(Y, args.seed_frac, args.min_per_class, rng)

        targets[u], windows[u], seeds[u] = Y, W, S
        n_lab = int((Y.sum(1) > 0).sum())
        n_multi = int((Y.sum(1) >= 2).sum())
        print(f"{u}: frames={nf} labelled={n_lab} multi(>=2)={n_multi} "
              f"({100 * n_multi / max(nf, 1):.1f}%) windows={len(W)} seeds={int(S.sum())}")

    report_coverage(units, targets, seeds, cmap)

    pickle.dump(cmap, open(out_dir / "class_map.pkl", "wb"))
    for u in units:
        np.savez_compressed(
            out_dir / f"{u}_data.npz",
            targets=targets[u], seeds=seeds[u],
            windows=np.array(windows[u], dtype=object),
        )
    print(f"\nsaved class_map.pkl + {len(units)} *_data.npz -> {out_dir.resolve()}")


if __name__ == "__main__":
    main()
