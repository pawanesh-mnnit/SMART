import numpy as np

__all__ = [
    "h71_stationary", "h72_calibrate", "h73_smooth", "cooccurrence", "h74_cooccur",
    "head_transform", "h75_threshold", "fit_thresholds", "decode_rankgap",
]

from .graph import stationary


# --------------------------------------------------------------------------
# 7.1 - 7.4  score transforms
# --------------------------------------------------------------------------

def h71_stationary(scores, P):
    """Subtract the stationary distribution: penalise structurally popular frames."""
    return scores - stationary(P)[:, None]


def h72_calibrate(scores):
    """Per-class z-score so classes with different seed counts become comparable."""
    mu = scores.mean(0, keepdims=True)
    sd = scores.std(0, keepdims=True) + 1e-12
    return (scores - mu) / sd


def h73_smooth(scores, w=5):
    """Moving average along time — actions are temporally contiguous."""
    w = min(w, scores.shape[0])          # clamp so N < w still works
    if w <= 1:
        return scores
    k = np.ones(w) / w
    out = np.empty_like(scores)
    for cc in range(scores.shape[1]):
        out[:, cc] = np.convolve(scores[:, cc], k, mode="same")
    return out


def cooccurrence(Ys):
    """Class co-occurrence matrix estimated from the SEED labels only.

    Diagonal is zeroed (a class does not boost itself) and the matrix is scaled
    to [0, 1] so LAMBDA has the same meaning across datasets.
    """
    Co = Ys.T @ Ys
    np.fill_diagonal(Co, 0.0)
    return Co / (Co.max() + 1e-12)


def h74_cooccur(scores, Co, lam=0.3):
    """Boost a class when the classes it habitually co-occurs with score high."""
    return scores + lam * (scores @ Co.T)


def head_transform(raw, P, Co, w_smooth=5, lam=0.3, row_center=True):
    """Apply 7.1 -> 7.4 (+ optional row centering) in order."""
    s = h71_stationary(raw, P)
    s = h72_calibrate(s)
    s = h73_smooth(s, w_smooth)
    s = h74_cooccur(s, Co, lam)
    if row_center:
        s = s - s.mean(1, keepdims=True)
    return s


# --------------------------------------------------------------------------
# 7.5  decoding: per-class thresholds
# --------------------------------------------------------------------------

def fit_thresholds(scores, Y, beta=0.5, grid=np.linspace(-2, 3, 51)):
    """Fit one threshold per class on the SEED frames by maximising F-beta.

    beta < 1 favours precision (fewer, cleaner labels); beta = 1 is plain F1.
    Classes absent from the seeds get tau = +inf and are never predicted.
    """
    b2 = beta * beta
    C = scores.shape[1]
    taus = np.full(C, np.inf)
    for cc in range(C):
        if Y[:, cc].sum() == 0:
            continue
        best = -1.0
        for t in grid:
            pr = (scores[:, cc] >= t).astype(int)
            tp = (pr * Y[:, cc]).sum()
            fp = (pr * (1 - Y[:, cc])).sum()
            fn = ((1 - pr) * Y[:, cc]).sum()
            fb = (1 + b2) * tp / ((1 + b2) * tp + b2 * fn + fp + 1e-12)
            if fb > best:
                best, taus[cc] = fb, t
    return taus


def h75_threshold(scores, taus, topk=0, floor=0.5, max_k=3, act_gate=None):
    """Threshold -> optional top-k rescue -> cap at max_k -> background gate.

    topk = 0 means no forced labels (honest evaluation: a frame may legitimately
    predict nothing). max_k caps labels per frame, which stops the 5-8 label
    over-firing seen without it. act_gate zeroes frames whose peak score falls
    below a percentile of the seed peak scores — the background gate.
    """
    n, C = scores.shape
    pred = (scores >= taus[None, :]).astype(int)

    if topk > 0:
        for i in range(n):
            if pred[i].sum() == 0 and scores[i].max() >= floor:
                pred[i, np.argsort(-scores[i])[:topk]] = 1

    if max_k:
        for i in range(n):
            if pred[i].sum() > max_k:
                pidx = np.where(pred[i] == 1)[0]
                keep = pidx[np.argsort(-scores[i, pidx])[:max_k]]
                pred[i] = 0
                pred[i, keep] = 1

    if act_gate is not None:
        pred[scores.max(1) < act_gate] = 0

    return pred


# --------------------------------------------------------------------------
# 7.5b  decoding: rank-gap (threshold-free, experimental)
# --------------------------------------------------------------------------

def decode_rankgap(scores, max_k=3, act_gate=None, persist=2):
    """Threshold-free set decoding.

    Per frame, sort classes by row-centered score and cut the label set at the
    largest consecutive score gap within the first max_k ranks. With
    `persist > 0` a candidate survives only if it appears in the top region for
    a majority of frames in a (2*persist+1) neighbourhood.

    mAP and top-k are score-based and therefore identical under this decoder and
    the F-beta one; only the set-based F1 scores change.
    """
    n, C = scores.shape
    pred = np.zeros((n, C), dtype=int)
    order = np.argsort(-scores, axis=1)
    kmax = min(max_k or 3, C - 1)

    for i in range(n):
        s = scores[i, order[i]]
        k = int(np.argmax(s[:kmax] - s[1:kmax + 1])) + 1
        pred[i, order[i, :k]] = 1

    if persist and persist > 0:
        top = np.zeros((n, C))
        for i in range(n):
            top[i, order[i, :kmax]] = 1
        cs = np.vstack([np.zeros((1, C)), np.cumsum(top, axis=0)])
        keep = np.zeros((n, C))
        for i in range(n):
            a = max(0, i - persist)
            b = min(n, i + persist + 1)
            keep[i] = (cs[b] - cs[a]) / (b - a)
        pred = (pred * (keep > 0.5)).astype(int)

    if act_gate is not None:
        pred[scores.max(1) < act_gate] = 0

    return pred
