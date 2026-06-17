"""Synthetic rotating-concept drift stream — controlled test bed for the
ignition theory. v2: full-space rotation + genuine class rarity.

Concept: y ~ Bernoulli(sigma(beta * w . x - b)), x ~ N(0, I_d).
At t*, the FULL weight vector rotates toward an orthogonal concept:
    w_post = cos(theta) * w_pre + sin(theta) * w_perp
with w_perp a fresh unit vector orthogonalized against w_pre (Gram-Schmidt).
  theta = 0     -> identical concept (no drift)
  theta = pi/2  -> w_post orthogonal to w_pre (a pre-trained model scores
                   near-randomly: this is the CLIFF the v1 plane-rotation lacked)
Base rate p is hit each step via a solved intercept, INDEPENDENT of theta.

FROZEN PARAMETERS (committed before observing sweep results):
"""
import numpy as np

T = 40                 # total steps
T_SHUTDOWN = 25        # concept rotates here
N_PER_STEP = 700       # transactions per step
D = 20                 # feature dimension
BETA = 6.0             # logit slope (separability)
TARGET_POS_PER_STEP = None  # if set, overrides base_rate to hit ~N positives/step


def _solve_intercept(scores, target_rate):
    lo, hi = -60.0, 60.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if np.mean(1.0 / (1.0 + np.exp(-(scores - mid)))) > target_rate:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def generate_stream(theta, base_rate, seed):
    rng = np.random.default_rng(seed)

    w_pre = rng.standard_normal(D)
    w_pre /= np.linalg.norm(w_pre)

    # fresh direction, orthogonalized against w_pre (Gram-Schmidt) -> w_perp
    r = rng.standard_normal(D)
    r -= (r @ w_pre) * w_pre
    w_perp = r / np.linalg.norm(r)

    w_post = np.cos(theta) * w_pre + np.sin(theta) * w_perp
    w_post /= np.linalg.norm(w_post)

    Xs, ys, ts = [], [], []
    for step in range(T):
        X = rng.standard_normal((N_PER_STEP, D))
        w = w_pre if step < T_SHUTDOWN else w_post
        raw = BETA * (X @ w)
        b = _solve_intercept(raw, base_rate)
        p = 1.0 / (1.0 + np.exp(-(raw - b)))
        y = (rng.random(N_PER_STEP) < p).astype(int)
        Xs.append(X); ys.append(y); ts.append(np.full(N_PER_STEP, step))

    return {"X": np.vstack(Xs), "y": np.concatenate(ys),
            "t": np.concatenate(ts),
            "w_pre": w_pre, "w_post": w_post, "w_perp": w_perp,
            "t_shutdown": T_SHUTDOWN}


def concept_angle(learned_w, true_w):
    """Angle (radians) between a learned boundary and the true concept.
    Uses signed cosine so anti-alignment (angle > pi/2) is visible."""
    a = learned_w / (np.linalg.norm(learned_w) + 1e-12)
    b = true_w / (np.linalg.norm(true_w) + 1e-12)
    return float(np.arccos(np.clip(a @ b, -1, 1)))