from tools.imports import * 
from tools.graph import (build_window_graph, make_windows, propagate_dispatch)
from tools.head import (cooccurrence, head_transform, fit_thresholds,
                        h75_threshold, decode_rankgap)
from tools.metrics import (evaluate as eval_metrics, topk_accuracy,
                           single_label_accuracy, ranking_metrics)

METRIC_KEYS = ["mAP", "macro_f1", "micro_f1", "top1", "top5", "top1_acc", "balanced_acc"]


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

class Config:
    """Every knob in one object so the ablation notebook can copy-and-modify it."""

    def __init__(self, **kw):
        self.feat_norm = "center"       # 'none' | 'center' | 'standardize'
        self.sigma_mode = "median"      # 'median' or a fixed float
        self.sparsify = "cdf"           # 'cdf' | 'knn'
        self.gamma = 0.90               # CDF cumulative-keep threshold
        self.knn_k = 10                 # neighbours kept when sparsify == 'knn'
        self.use_cdf = True             # ablation toggle: sparsification on/off
        self.use_temporal = True        # ablation toggle: temporal edges on/off
        self.prop_mode = "independent"  # 'independent' (ours) | 'joint' (baseline)
        self.rw_steps = 10              # random-walk propagation steps
        self.w_smooth = 5               # temporal smoothing window
        self.lam = 0.3                  # co-occurrence boost weight
        self.row_center = True          # per-frame across-class centering
        self.decoder = "fbeta"          # 'fbeta' | 'rankgap'
        self.fbeta = 0.5                # < 1 favours precision
        self.max_k = 3                  # cap labels per frame
        self.topk = 0                   # 0 = no forced labels (honest eval)
        self.topk_floor = 0.5
        self.act_pctl = 10              # background gate percentile (None = off)
        self.rank_persist = 2           # rank-gap decoder only
        self.__dict__.update(kw)

    def __repr__(self):
        keys = ["feat_norm", "sparsify", "gamma", "knn_k", "rw_steps", "w_smooth",
                "lam", "row_center", "decoder", "fbeta", "max_k", "prop_mode"]
        return "Config(" + ", ".join(f"{k}={getattr(self, k)}" for k in keys) + ")"


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_unit(data_dir, unit, feat_suffix=""):
    """Return (feats, targets, seeds, saved_windows)."""
    data_dir = Path(data_dir)
    fpath = data_dir / f"{unit}_feats{feat_suffix}.npz"
    if not fpath.exists():
        raise FileNotFoundError(
            f"{fpath} not found — run extract_features.py, or fix --feat_suffix")
    feats = np.load(fpath)["feats"]
    d = np.load(data_dir / f"{unit}_data.npz", allow_pickle=True)
    return feats, d["targets"], d["seeds"], [list(w) for w in d["windows"]]


def list_units(data_dir):
    return sorted(p.stem.replace("_data", "") for p in Path(data_dir).glob("*_data.npz"))


# --------------------------------------------------------------------------
# Core: score one unit
# --------------------------------------------------------------------------

def score_unit(feats, Y, seeds, windows, Co, cfg):
    """Propagate + head over every window. Returns (scores, covered_mask, sigmas)."""
    n, C = Y.shape
    S = np.zeros((n, C))
    covered = np.zeros(n, dtype=bool)
    sigmas = []

    for w in windows:
        idx = np.array(w, dtype=int)
        P, sigma = build_window_graph(
            feats[idx], cfg.feat_norm, cfg.sigma_mode,
            cfg.use_cdf, cfg.use_temporal, cfg.sparsify, cfg.gamma, cfg.knn_k)
        sigmas.append(sigma)

        raw = propagate_dispatch(P, Y[idx], seeds[idx], cfg.rw_steps, cfg.prop_mode)
        S[idx] = head_transform(raw, P, Co, cfg.w_smooth, cfg.lam, cfg.row_center)
        covered[idx] = True

    return S, covered, sigmas


def decode(scores, taus, cfg, act_gate=None):
    if cfg.decoder == "rankgap":
        return decode_rankgap(scores, cfg.max_k, act_gate, cfg.rank_persist)
    return h75_threshold(scores, taus, cfg.topk, cfg.topk_floor, cfg.max_k, act_gate)


def run_unit(data_dir, unit, cfg, feat_suffix="", window=None, verbose=True):
    """Full pipeline for one unit. Metrics are computed on non-seed frames only."""
    feats, Y, seeds, saved_windows = load_unit(data_dir, unit, feat_suffix)
    n, C = Y.shape
    windows = make_windows(n, window) if window else saved_windows

    Co = cooccurrence(Y[seeds])
    S, covered, sigmas = score_unit(feats, Y, seeds, windows, Co, cfg)

    taus = fit_thresholds(S[seeds], Y[seeds], cfg.fbeta)
    act = (np.percentile(S[seeds].max(1), cfg.act_pctl)
           if cfg.act_pctl is not None else None)

    ev = covered & (~seeds)
    pred = decode(S[ev], taus, cfg, act)

    m = eval_metrics(pred, S[ev], Y[ev])
    m.update(topk_accuracy(S[ev], Y[ev], ks=(1, 5)))
    m.update(single_label_accuracy(S[ev], Y[ev]))
    m.update(ranking_metrics(S[ev], Y[ev]))
    m["_frames"] = np.where(ev)[0]
    m["_pred"] = pred
    m["_scores"] = S[ev]
    m["_Y"] = Y[ev]
    m["_sigma_med"] = float(np.median(sigmas))

    if verbose:
        print(f'{unit} | N={window or "saved"} steps={cfg.rw_steps} '
              f'gamma={cfg.gamma} | sigma med={np.median(sigmas):.3f} '
              f'| eval frames={int(ev.sum())}')
        print("  ", {k: round(m[k], 4) for k in METRIC_KEYS if k in m})
    return m


def run_all(data_dir, units, cfg, feat_suffix="", window=None, verbose=True):
    """Protocol 1: fixed config per unit, then average across units."""
    cmap = pickle.load(open(Path(data_dir) / "class_map.pkl", "rb"))
    names = cmap["names"]
    rows, per_class = {}, {}

    for u in units:
        m = run_unit(data_dir, u, cfg, feat_suffix, window, verbose=False)
        rows[u] = {k: m[k] for k in METRIC_KEYS if k in m}
        for c, ap in m["per_class_ap"].items():
            per_class.setdefault(int(c), []).append(ap)

    keys = [k for k in METRIC_KEYS if k in rows[units[0]]]
    mean = {k: float(np.mean([rows[u][k] for u in units])) for k in keys}

    if verbose:
        hdr = f'{"unit":<14}' + "".join(f"{k:>14}" for k in keys)
        print(f'\nconfig: N={window or "saved"} steps={cfg.rw_steps} '
              f'gamma={cfg.gamma} | {len(units)} units\n')
        print(hdr)
        for u in units:
            print(f"{u:<14}" + "".join(f"{rows[u][k]:>14.4f}" for k in keys))
        print("-" * len(hdr))
        print(f'{"MEAN":<14}' + "".join(f"{mean[k]:>14.4f}" for k in keys))

        print("\nper-class AP (mean over units where the class appears):")
        for c in sorted(per_class, key=lambda c: -np.mean(per_class[c])):
            aps = per_class[c]
            print(f"  {names[c][:30]:<32} AP={np.mean(aps):.3f}  (n={len(aps)})")

    return dict(per_unit=rows, mean=mean,
                per_class={c: float(np.mean(v)) for c, v in per_class.items()})


def run_pooled(data_dir, units, cfg, feat_suffix="", window=None, verbose=True):
    """Protocol 2: per-unit graphs, but global co-occurrence, thresholds and metrics."""
    data = {}
    seedY = []
    for u in units:
        feats, Y, seeds, saved = load_unit(data_dir, u, feat_suffix)
        data[u] = (feats, Y, seeds, saved)
        seedY.append(Y[seeds])

    Cs = {data[u][1].shape[1] for u in units}
    assert len(Cs) == 1, (f"inconsistent class counts {Cs} — rerun dataset.py "
                          f"over all units together so they share a class map")

    Co = cooccurrence(np.vstack(seedY))     # GLOBAL co-occurrence
    allS, allY, sS, sY = [], [], [], []

    for u in units:
        feats, Y, seeds, saved = data[u]
        windows = make_windows(Y.shape[0], window) if window else saved
        S, covered, _ = score_unit(feats, Y, seeds, windows, Co, cfg)
        ev = covered & (~seeds)
        allS.append(S[ev]); allY.append(Y[ev])
        sS.append(S[seeds]); sY.append(Y[seeds])

    seedScores = np.vstack(sS)
    taus = fit_thresholds(seedScores, np.vstack(sY), cfg.fbeta)   # GLOBAL thresholds
    act = (np.percentile(seedScores.max(1), cfg.act_pctl)
           if cfg.act_pctl is not None else None)

    S = np.vstack(allS)
    Yv = np.vstack(allY)
    pred = decode(S, taus, cfg, act)

    m = eval_metrics(pred, S, Yv)
    m.update(topk_accuracy(S, Yv, ks=(1, 5)))
    m.update(single_label_accuracy(S, Yv))

    if verbose:
        print(f'\nPOOLED | {len(units)} units | N={window or "saved"} '
              f'steps={cfg.rw_steps} gamma={cfg.gamma} | eval frames={len(S)}')
        print("  ", {k: round(m[k], 4) for k in METRIC_KEYS if k in m})
    return m


# --------------------------------------------------------------------------
# Diagnostics and outputs
# --------------------------------------------------------------------------

def diagnostics(data_dir, unit, cfg, feat_suffix="", window=None):
    """Where is the signal being lost? Seed coverage, sigma spread, per-class AP."""
    from tools.graph import normalize_features, pdf_weights

    feats, Y, seeds, saved = load_unit(data_dir, unit, feat_suffix)
    windows = make_windows(Y.shape[0], window) if window else saved

    seedcounts = [int(seeds[np.array(w, int)].sum()) for w in windows]
    sig = [pdf_weights(normalize_features(feats[np.array(w, int)], cfg.feat_norm),
                       sigma_mode=cfg.sigma_mode)[3] for w in windows]

    zero = sum(s == 0 for s in seedcounts)
    print(f"{unit}: {len(windows)} windows | zero-seed windows: {zero} "
          f"({100 * zero / max(len(windows), 1):.1f}%)")
    print(f"  seeds/window: mean={np.mean(seedcounts):.2f} "
          f"min={min(seedcounts)} max={max(seedcounts)}")
    print(f"  sigma/window: mean={np.mean(sig):.3f} "
          f"min={np.min(sig):.3f} max={np.max(sig):.3f}")

    m = run_unit(data_dir, unit, cfg, feat_suffix, window, verbose=False)
    print("  per-class AP (worst first):")
    for cc, ap in sorted(m["per_class_ap"].items(), key=lambda x: x[1]):
        print(f"    class {cc}: AP={ap:.3f}")
    return m


def save_predicted_labels(data_dir, unit, m, out_path=None, only_multi=False):
    """Write the decoded per-frame label SETS next to the ground truth."""
    cmap = pickle.load(open(Path(data_dir) / "class_map.pkl", "rb"))
    names = cmap["names"]
    frames, pred, Yt = m["_frames"], m["_pred"], m["_Y"]

    out_path = Path(out_path or Path(data_dir) /
                    f'{unit}_predicted_labels{"_multi" if only_multi else ""}.csv')
    n_multi = 0
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "n_pred", "pred_names", "n_true", "true_names",
                    "is_multi_pred", "is_multi_true"])
        for r in range(len(frames)):
            pl = [int(c) for c in np.where(pred[r] == 1)[0]]
            tl = [int(c) for c in np.where(Yt[r] == 1)[0]]
            n_multi += len(pl) >= 2
            if only_multi and len(pl) < 2:
                continue
            w.writerow([int(frames[r]), len(pl), "|".join(names[c] for c in pl),
                        len(tl), "|".join(names[c] for c in tl),
                        int(len(pl) >= 2), int(len(tl) >= 2)])

    print(f"saved -> {out_path}")
    print(f"  multi-action predicted frames: {n_multi}/{len(frames)} "
          f"({100 * n_multi / max(len(frames), 1):.1f}%)")
    return out_path


def timeline_figure(data_dir, unit, m, save_path=None, classes=None,
                    formats=("svg", "pdf", "png")):
    """Qualitative GT-vs-SMART per-class timeline. Vector output for the paper.

    Colours are Okabe-Ito (colourblind-safe and distinct in greyscale) and no
    alpha is used, so EPS export stays faithful.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    GT_COLOR, PRED_COLOR, BAND_COLOR = "#3A3A3A", "#0072B2", "#EDEDED"
    names = pickle.load(open(Path(data_dir) / "class_map.pkl", "rb"))["names"]
    frames, pred, Yt = m["_frames"], m["_pred"], m["_Y"]
    n = int(frames.max()) + 1

    def tracks(mat):
        T = np.zeros((n, mat.shape[1]))
        T[frames] = mat
        return T

    GT, PR = tracks(Yt), tracks(pred)
    if classes is None:
        classes = [c for c in range(GT.shape[1]) if GT[:, c].sum() > 0]

    fig, ax = plt.subplots(figsize=(11, 0.52 * len(classes) + 1))

    def spans(track):
        d = np.diff(np.concatenate([[0], track, [0]]))
        return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))

    for r, c in enumerate(classes):
        y = len(classes) - 1 - r
        if r % 2 == 0:
            ax.axhspan(y - 0.5, y + 0.5, color=BAND_COLOR, zorder=0)
        for s, e in spans(GT[:, c]):
            ax.barh(y + 0.19, e - s, left=s, height=0.34, color=GT_COLOR, zorder=2)
        for s, e in spans(PR[:, c]):
            ax.barh(y - 0.19, e - s, left=s, height=0.34, color=PRED_COLOR, zorder=2)

    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels([names[c][:26] for c in reversed(classes)], fontsize=9)
    ax.set_xlabel("frame", fontsize=10)
    ax.set_xlim(0, n)
    ax.set_ylim(-0.5, len(classes) - 0.5)
    ax.tick_params(axis="x", labelsize=9)
    for sp in ["top", "right", "left"]:
        ax.spines[sp].set_visible(False)
    ax.legend(handles=[mpatches.Patch(color=GT_COLOR, label="Ground truth"),
                       mpatches.Patch(color=PRED_COLOR, label="SMART")],
              loc="upper right", fontsize=9, framealpha=1.0)
    ax.set_title(f"{unit}: per-frame multi-action timeline", fontsize=11)
    plt.tight_layout()

    if save_path:
        for ext in formats:
            out = f"{save_path}.{ext}"
            plt.savefig(out, format=ext, bbox_inches="tight",
                        **({} if ext in ("svg", "eps", "pdf") else {"dpi": 300}))
            print("saved ->", out)
    return fig


# --------------------------------------------------------------------------

def build_parser():
    ap = argparse.ArgumentParser(
        description="SMART — graph propagation, multi-action head, evaluation")
    ap.add_argument("--data_dir", required=True,
                    help="folder with class_map.pkl, *_data.npz and *_feats*.npz")
    ap.add_argument("--units", nargs="*", default=None)
    ap.add_argument("--feat_suffix", default="",
                    help="'' | _resnet | _effnet | _clip | _rgbonly | _flowonly")
    ap.add_argument("--protocol", default="per_unit", choices=["per_unit", "pooled"])
    ap.add_argument("--window", type=int, default=None,
                    help="rebuild windows at this N; default = the saved windows")

    g = ap.add_argument_group("graph")
    g.add_argument("--feat_norm", default="center",
                   choices=["none", "center", "standardize"])
    g.add_argument("--sigma_mode", default="median")
    g.add_argument("--sparsify", default="cdf", choices=["cdf", "knn"])
    g.add_argument("--gamma", type=float, default=0.90)
    g.add_argument("--knn_k", type=int, default=10)
    g.add_argument("--no_cdf", action="store_true", help="ablation: skip sparsification")
    g.add_argument("--no_temporal", action="store_true", help="ablation: no temporal edges")
    g.add_argument("--prop_mode", default="independent", choices=["independent", "joint"])
    g.add_argument("--rw_steps", type=int, default=10)

    h = ap.add_argument_group("head")
    h.add_argument("--w_smooth", type=int, default=5)
    h.add_argument("--lam", type=float, default=0.3)
    h.add_argument("--no_row_center", action="store_true", help="ablation")
    h.add_argument("--decoder", default="fbeta", choices=["fbeta", "rankgap"])
    h.add_argument("--fbeta", type=float, default=0.5)
    h.add_argument("--max_k", type=int, default=3)
    h.add_argument("--topk", type=int, default=0)
    h.add_argument("--act_pctl", type=float, default=10)
    h.add_argument("--rank_persist", type=int, default=2)

    o = ap.add_argument_group("outputs")
    o.add_argument("--diagnostics", action="store_true")
    o.add_argument("--save_labels", action="store_true")
    o.add_argument("--timeline", default=None, metavar="UNIT",
                   help="save the qualitative timeline figure for this unit")
    o.add_argument("--fig_dir", default="images")
    o.add_argument("--save_json", default=None, help="write the metrics to this path")
    return ap


def main():
    args = build_parser().parse_args()

    cfg = Config(
        feat_norm=args.feat_norm, sigma_mode=args.sigma_mode,
        sparsify=args.sparsify, gamma=args.gamma, knn_k=args.knn_k,
        use_cdf=not args.no_cdf, use_temporal=not args.no_temporal,
        prop_mode=args.prop_mode, rw_steps=args.rw_steps,
        w_smooth=args.w_smooth, lam=args.lam, row_center=not args.no_row_center,
        decoder=args.decoder, fbeta=args.fbeta, max_k=args.max_k,
        topk=args.topk, act_pctl=args.act_pctl, rank_persist=args.rank_persist,
    )

    units = args.units or list_units(args.data_dir)
    banner(f"SMART evaluate — {len(units)} units | feat_suffix='{args.feat_suffix}'")
    print(cfg)

    if args.diagnostics:
        for u in units:
            diagnostics(args.data_dir, u, cfg, args.feat_suffix, args.window)
        return

    if args.protocol == "pooled":
        res = run_pooled(args.data_dir, units, cfg, args.feat_suffix, args.window)
        headline = res["mAP"]
    else:
        res = run_all(args.data_dir, units, cfg, args.feat_suffix, args.window)
        headline = res["mean"]["mAP"]
        if args.save_labels:
            for u in units:
                m = run_unit(args.data_dir, u, cfg, args.feat_suffix,
                             args.window, verbose=False)
                save_predicted_labels(args.data_dir, u, m)

    print(f"\nHEADLINE mAP = {headline:.4f}")

    if args.timeline:
        m = run_unit(args.data_dir, args.timeline, cfg, args.feat_suffix,
                     args.window, verbose=False)
        Path(args.fig_dir).mkdir(parents=True, exist_ok=True)
        timeline_figure(args.data_dir, args.timeline, m,
                        save_path=str(Path(args.fig_dir) / f"timeline_{args.timeline}"))

    if args.save_json:
        clean = {k: v for k, v in res.items() if not k.startswith("_")}
        Path(args.save_json).parent.mkdir(parents=True, exist_ok=True)
        json.dump(clean, open(args.save_json, "w"), indent=2, default=float)
        print(f"metrics -> {args.save_json}")


if __name__ == "__main__":
    main()
