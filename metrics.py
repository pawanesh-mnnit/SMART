import numpy as np

__all__ = [
    "average_precision", "evaluate", "topk_accuracy",
    "single_label_accuracy", "ranking_metrics",
]


def average_precision(y_true, y_score):
    """Area under the precision-recall curve for one class."""
    o = np.argsort(-y_score)
    y = y_true[o]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    prec = tp / (tp + fp + 1e-12)
    rec = tp / (y.sum() + 1e-12)
    ap, prev = 0.0, 0.0
    for p_, r_ in zip(prec, rec):
        ap += p_ * (r_ - prev)
        prev = r_
    return ap


def evaluate(pred, scores, Y):
    """mAP + micro/macro F1 + per-class AP over classes present in Y."""
    present = np.where(Y.sum(0) > 0)[0]
    if len(present) == 0:
        return dict(mAP=float("nan"), micro_f1=float("nan"), macro_f1=float("nan"),
                    per_class_ap={})

    mAP = np.mean([average_precision(Y[:, cc], scores[:, cc]) for cc in present])

    tp = (pred * Y).sum()
    fp = (pred * (1 - Y)).sum()
    fn = ((1 - pred) * Y).sum()
    micro = 2 * tp / (2 * tp + fp + fn + 1e-12)

    f1s = []
    for cc in present:
        tpc = (pred[:, cc] * Y[:, cc]).sum()
        fpc = (pred[:, cc] * (1 - Y[:, cc])).sum()
        fnc = ((1 - pred[:, cc]) * Y[:, cc]).sum()
        f1s.append(2 * tpc / (2 * tpc + fpc + fnc + 1e-12))

    return dict(
        mAP=float(mAP),
        micro_f1=float(micro),
        macro_f1=float(np.mean(f1s)),
        per_class_ap={int(cc): float(average_precision(Y[:, cc], scores[:, cc]))
                      for cc in present},
    )


def topk_accuracy(scores, Y, ks=(1, 5)):
    """Multi-label top-k: a frame is a hit if ANY true label is in its top-k.

    Scored on labelled frames only — background frames have no correct class.
    """
    labeled = Y.sum(1) > 0
    s, y = scores[labeled], Y[labeled]
    if len(y) == 0:
        return {f"top{k}": float("nan") for k in ks}
    order = np.argsort(-s, axis=1)
    out = {}
    for k in ks:
        kk = min(k, s.shape[1])
        topk = order[:, :kk]
        hit = np.array([y[i, topk[i]].any() for i in range(len(y))])
        out[f"top{kk}"] = float(hit.mean())
    return out


def single_label_accuracy(scores, Y):
    """Single-label protocol for comparison against single-action baselines.

    Each frame makes ONE prediction (the argmax class) and is correct if that
    class is among its true labels. `balanced_acc` is the mean over classes of
    recall at top-1, which is the fair number on a long-tailed class list.
    """
    labeled = Y.sum(1) > 0
    s, y = scores[labeled], Y[labeled]
    if len(y) == 0:
        return dict(top1_acc=float("nan"), balanced_acc=float("nan"))

    top = np.argmax(s, axis=1)
    hit = np.array([y[i, top[i]] > 0 for i in range(len(y))])
    top1 = float(hit.mean())

    present = np.where(y.sum(0) > 0)[0]
    recs = []
    for c in present:
        idx = np.where(y[:, c] > 0)[0]
        recs.append(float(np.mean([y[i, top[i]] > 0 and top[i] == c for i in idx]))
                    if len(idx) else 0.0)
    return dict(top1_acc=top1, balanced_acc=float(np.mean(recs)))


def ranking_metrics(scores, Y):
    """Decoder-independent quality of the score ranking itself.

    LRAP    — for each positive label, the precision of positives ranked above
              it (higher is better).
    coverage— how deep in the ranking one must go to cover all true labels
              (lower is better).
    """
    n, C = scores.shape
    lraps, covs = [], []
    for i in range(n):
        pos = np.where(Y[i] == 1)[0]
        if len(pos) == 0:
            continue
        rank = np.empty(C)
        rank[np.argsort(-scores[i])] = np.arange(1, C + 1)
        r = rank[pos]
        lraps.append(np.mean([(r <= rj).sum() / rj for rj in r]))
        covs.append(r.max())
    if not lraps:
        return dict(LRAP=float("nan"), coverage=float("nan"))
    return dict(LRAP=float(np.mean(lraps)), coverage=float(np.mean(covs)))
