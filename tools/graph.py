"""
tools/graph.py — graph construction and random-walk label propagation.

Pipeline for one N-frame window of D-dimensional features:

    F  ->  normalize_features   (per-window mean-centering; removes the scene DC term)
       ->  pdf_weights          (cosine similarity -> distance -> Gaussian PDF kernel)
       ->  sparsify             (CDF cumulative-mass pruning, or fixed top-k kNN)
       ->  add_temporal_edges   (chain edges between consecutive frames)
       ->  transition_matrix    (column-normalized -> column-stochastic P)
       ->  propagate            (one independent random walk per class)

`propagate_joint` is the classical single-diffusion baseline in which classes
compete for probability mass; it exists only for the propagation ablation.

All functions are pure — every knob is an explicit argument, so the module can
be imported by evaluate.py, the ablation notebook and the visualisation
notebook without any shared global state.
"""

import numpy as np

__all__ = [
    "normalize_features", "cosine_similarity", "estimate_sigma", "pdf_weights",
    "row_normalize", "cdf_sparsify", "knn_sparsify", "sparsify",
    "add_temporal_edges", "transition_matrix",
    "random_walk", "stationary", "propagate", "propagate_joint", "propagate_dispatch",
    "build_window_graph", "make_windows",
]


# --------------------------------------------------------------------------
# Stage 1-3 — features to affinities (PDF)
# --------------------------------------------------------------------------

def normalize_features(F, mode="center"):
    """Per-window feature normalisation.

    The frames inside one window share a scene: a large component of every
    feature vector is the room, not the action. Mean-centering inside the
    window removes that common 'DC' term so cosine similarity reflects action
    variation instead of scene identity.

    mode: 'none' | 'center' | 'standardize'
    """
    if mode == "none":
        return F
    F = F - F.mean(0, keepdims=True)
    if mode == "standardize":
        F = F / (F.std(0, keepdims=True) + 1e-8)
    return F


def cosine_similarity(F):
    """(n, D) features -> (n, n) cosine similarity matrix."""
    Fn = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-12)
    return Fn @ Fn.T


def estimate_sigma(D):
    """Data-driven kernel bandwidth: median off-diagonal pairwise distance.

    A fixed sigma tuned on a 2-D toy example collapses the kernel on real
    512-D features, so sigma is estimated per window.
    """
    iu = np.triu_indices_from(D, k=1)
    d = D[iu]
    s = np.median(d) if d.size else 1.0
    return float(s) if s > 1e-6 else 1.0


def pdf_weights(F, sigma=None, sigma_mode="median"):
    """Gaussian PDF kernel over cosine distance.

    Returns (S, D, W, sigma) where S is cosine similarity, D = 1 - S is the
    distance, and W is the unnormalised affinity matrix.
    """
    S = cosine_similarity(F)
    D = 1.0 - S
    if sigma is None:
        sigma = estimate_sigma(D) if sigma_mode == "median" else float(sigma_mode)
    W = (1.0 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-(D ** 2) / (2 * sigma ** 2))
    return S, D, W, sigma


# --------------------------------------------------------------------------
# Stage 4 — sparsification, temporal edges, transition matrix (CDF)
# --------------------------------------------------------------------------

def row_normalize(M):
    return M / (M.sum(1, keepdims=True) + 1e-12)


def cdf_sparsify(W, gamma=0.90):
    """CDF pruning: per frame, keep the strongest neighbours whose row-normalised
    affinities cumulatively reach `gamma` of the total mass.

    Unlike fixed top-k this adapts the neighbourhood size per frame — a frame
    in a homogeneous stretch keeps many neighbours, a transition frame keeps few.
    """
    P = row_normalize(W)
    keep = np.zeros_like(W, dtype=bool)
    for i in range(W.shape[0]):
        o = np.argsort(-P[i])
        cs = np.cumsum(P[i][o])
        k = np.searchsorted(cs, gamma) + 1
        keep[i, o[:k]] = True
    return W * keep


def knn_sparsify(W, k=10):
    """Fixed top-k pruning baseline.

    Selection runs on the row-normalised affinities (same ordering as raw W) and
    everything downstream is identical to the CDF path, so a CDF-vs-kNN
    comparison isolates the pruning rule alone.
    """
    P = row_normalize(W)
    keep = np.zeros_like(W, dtype=bool)
    for i in range(W.shape[0]):
        keep[i, np.argsort(-P[i])[:min(k, W.shape[0])]] = True
    return W * keep


def sparsify(W, mode="cdf", gamma=0.90, knn_k=10):
    """Dispatch between the two pruning rules."""
    return knn_sparsify(W, knn_k) if mode == "knn" else cdf_sparsify(W, gamma)


def add_temporal_edges(A, wt=1.0):
    """Guarantee a chain edge between consecutive frames.

    Without this a visually atypical frame can be orphaned by sparsification and
    receive no probability mass at all.
    """
    A = A.copy()
    for i in range(A.shape[0] - 1):
        A[i, i + 1] = max(A[i, i + 1], wt)
        A[i + 1, i] = max(A[i + 1, i], wt)
    return A


def transition_matrix(A):
    """Column-normalise the adjacency -> column-stochastic transition matrix P."""
    return A / (A.sum(0, keepdims=True) + 1e-12)


def build_window_graph(F, feat_norm="center", sigma_mode="median",
                       use_cdf=True, use_temporal=True,
                       sparsify_mode="cdf", gamma=0.90, knn_k=10):
    """Full feature-window -> transition-matrix path. Returns (P, sigma)."""
    Fw = normalize_features(F, feat_norm)
    _, _, W, sigma = pdf_weights(Fw, sigma_mode=sigma_mode)
    A = sparsify(W, sparsify_mode, gamma, knn_k) if use_cdf else W
    if use_temporal:
        A = add_temporal_edges(A)
    return transition_matrix(A), sigma


# --------------------------------------------------------------------------
# Stage 6 — random-walk propagation
# --------------------------------------------------------------------------

def random_walk(P, p0, steps=2):
    """Plain t-step random walk: p_t = P^t p_0."""
    p = p0.astype(float).copy()
    for _ in range(steps):
        p = P @ p
    return p


def stationary(P, steps=200):
    """Approximate stationary distribution — the graph's structural prior.

    Subtracted from the raw scores in the head so that hub frames don't score
    high for every class simply because they are well connected.
    """
    n = P.shape[0]
    return random_walk(P, np.ones(n) / n, steps)


def propagate(P, Y, seed_mask, steps=2):
    """SMART propagation: one INDEPENDENT random walk per class.

    Each class starts from a uniform distribution over its own seed frames and
    diffuses without any cross-class coupling. Because the walks never compete,
    two co-occurring actions can both reach high scores on the same frame —
    which is what makes concurrent multi-action prediction possible.
    """
    n, C = Y.shape
    sc = np.zeros((n, C))
    si = np.where(seed_mask)[0]
    for cc in range(C):
        cls = [i for i in si if Y[i, cc] > 0]
        if not cls:
            continue
        p0 = np.zeros(n)
        p0[cls] = 1.0 / len(cls)
        sc[:, cc] = random_walk(P, p0, steps)
    return sc


def propagate_joint(P, Y, seed_mask, steps=2):
    """Classical JOINT propagation baseline (ablation only).

    All classes diffuse over the same graph but COMPETE: after each step every
    frame's scores are normalised across classes to sum to 1, so mass gained by
    one class is taken from the others. Seeds are re-clamped to their known
    labels each step. Everything else (graph, head, decoding) is identical to
    the independent walk, so the ablation isolates only the independence
    property.
    """
    n, C = Y.shape
    si = np.where(seed_mask)[0]
    F = np.zeros((n, C))
    for cc in range(C):
        cls = [i for i in si if Y[i, cc] > 0]
        if cls:
            F[cls, cc] = 1.0 / len(cls)
    seedrows = F[si].copy()
    for _ in range(steps):
        F = P @ F
        rs = F[si].sum(1, keepdims=True)
        F[si] = np.where(rs > 0, seedrows, F[si])
        rowsum = F.sum(1, keepdims=True)
        F = np.divide(F, rowsum, out=F.copy(), where=rowsum > 0)
    return F


def propagate_dispatch(P, Y, seed_mask, steps=2, mode="independent"):
    if mode == "independent":
        return propagate(P, Y, seed_mask, steps)
    return propagate_joint(P, Y, seed_mask, steps)


# --------------------------------------------------------------------------

def make_windows(n_frames, N, stride=None):
    """Split a video of n_frames into non-overlapping N-frame windows.

    Short videos (n_frames < N) fall back to a single ragged window so that
    Charades clips shorter than the window size are still evaluated.
    """
    stride = stride or N
    if n_frames < N:
        return [list(range(n_frames))]
    return [list(range(s, s + N)) for s in range(0, n_frames - N + 1, stride)]
