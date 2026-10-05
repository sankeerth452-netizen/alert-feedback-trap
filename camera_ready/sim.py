"""Unified, instrumented re-implementation of the AEGIS simulators.

With default arguments this reproduces, line for line, the logic of
  experiments/run_frontier.py            (policy sweep; caught = investigated)
  experiments/run_aegis_final.py         (static / adaptive_full / aegis_cg)
  experiments/run_step42_retrainers.py   (replace / average gates)
Every step records BOTH catch definitions:
  caught_topk : illicit among the top-k of the detector's score (paper Sec 6 / Table 3)
  caught_inv  : illicit among transactions actually investigated (paper Sec 8 / frontier)
They coincide under greedy investigation.

Options (off by default):
  boot_seed  : bootstrap transactions within every simulated step
  feedback   : 'alert' (top-k + exploration) | 'bulk' (all labels of the step)
  gate       : 'competence' | 'psi' | 'replace' | 'average' | 'static'
  retrainer  : 'window' (paper) | 'window_ipw' | 'window_ri' |
               'cumulative' (history + all alert labels, refit each step) |
               'cumulative_ipw' | 'alerts_all' (all alert labels, no history)
  min_pos, window, tau, ema : gate/trigger hyper-parameters for sensitivity
"""
import numpy as np
from camera_ready.common import (fit_lgbm, safe_ap, mean_psi, LABEL_DELAY, ALARM_Z,
                       MIN_POS, MIN_NEG, ADAPT_WINDOW, COMP_TAU, COMP_EMA)

POLICY_EPS = {"greedy": 0.0, "eps05": 0.05, "eps10": 0.1, "eps20": 0.2,
              "eps40": 0.4, "stratified": 0.2, "novelty": 0.2}


def choose(policy, prob, novelty, k, rng, nx_rule):
    """-> chosen, exploit, explore (local indices), propensity of explore items."""
    order = np.argsort(prob)[::-1]
    eps = POLICY_EPS[policy]
    if eps == 0:
        return order[:k], order[:k], np.array([], int), 1.0
    n_x = max(1, int(eps * k)) if nx_rule == "frontier" else int(eps * k)
    exploit = order[: k - n_x]
    rest = np.setdiff1d(np.arange(len(prob)), exploit)
    if len(rest) == 0 or n_x == 0:
        return exploit, exploit, np.array([], int), 1.0
    if policy.startswith("eps"):
        explore = rng.choice(rest, size=min(n_x, len(rest)), replace=False)
    elif policy == "stratified":
        deciles = np.array_split(rest[np.argsort(prob[rest])[::-1]], 10)
        picks, i = [], 0
        while len(picks) < min(n_x, len(rest)):
            d = deciles[i % 10]
            if len(d):
                picks.append(rng.choice(d))
                deciles[i % 10] = d[d != picks[-1]]
            i += 1
        explore = np.array(picks)
    else:  # novelty
        explore = rest[np.argsort(novelty[rest])[::-1][:n_x]]
    prop = min(n_x, len(rest)) / len(rest)
    return np.concatenate([exploit, explore]).astype(int), exploit, explore, prop


def run(D, train_end, policy="greedy", k=50, seed=42, *, boot_seed=None,
        feedback="alert", gate="competence", retrainer="window",
        nx_rule="frontier", min_pos=MIN_POS, window=ADAPT_WINDOW,
        tau=COMP_TAU, ema=COMP_EMA, record_from=43):
    X, y, t = D["X"], D["y"], D["t"]
    psi_z, p_base_all = D["psi_z"][train_end], D["p_base"][train_end]
    novelty_all = D["novelty"][train_end]
    base_ref = X[t <= train_end] if gate == "psi" else None
    rng = np.random.default_rng(seed)
    brng = np.random.default_rng(10_000 + boot_seed) if boot_seed is not None else None
    cumulative = retrainer.startswith("cumulative") or retrainer == "alerts_all"

    model, model_ref = None, None          # adaptive expert (or cumulative model)
    comp = {"base": 0.5, "adp": 0.5}
    revealed, uninvest, prop_of = {}, {}, {}
    pool_info = {}
    pos_post = 0
    rows = []

    for s in range(train_end + 1, 50):
        idx_s = np.where((t == s) & (y != -1))[0]
        if brng is not None:
            idx_s = idx_s[brng.integers(0, len(idx_s), len(idx_s))]
        ys, Xs = y[idx_s], X[idx_s]
        p_base = p_base_all[idx_s]

        # ---------------- score ----------------
        w_adp, p_adp = 0.0, None
        if gate == "static" or model is None:
            prob = p_base
        else:
            p_adp = model.predict_proba(Xs)[:, 1]
            if cumulative or gate == "replace":
                prob, w_adp = p_adp, 1.0
            elif gate == "average":
                prob, w_adp = 0.5 * p_base + 0.5 * p_adp, 0.5
            else:
                if gate == "competence":
                    logits = np.array([comp["base"], comp["adp"]]) / tau
                else:  # psi gate (run_aegis_final adaptive_full / adaptive_budget)
                    logits = -np.array([mean_psi(base_ref, Xs), mean_psi(model_ref, Xs)]) / 0.05
                logits -= logits.max()
                w = np.exp(logits); w /= w.sum()
                prob, w_adp = w[0] * p_base + w[1] * p_adp, float(w[1])

        kk = min(k, len(idx_s))
        topk = np.argsort(prob)[::-1][:kk]
        if feedback == "bulk":
            chosen, exploit, explore, prop = np.arange(len(idx_s)), np.arange(len(idx_s)), np.array([], int), 1.0
        else:
            chosen, exploit, explore, prop = choose(policy, prob, novelty_all[idx_s], kk, rng, nx_rule)

        if s >= record_from:
            inv = chosen if feedback == "alert" else topk
            rows.append(dict(time_step=s, cum_pos_before=pos_post,
                             pr_auc=safe_ap(ys, prob), prev=float(ys.mean()),
                             caught_inv=int(ys[inv].sum()),
                             caught_topk=int(ys[topk].sum()),
                             caught_explore=int(ys[explore].sum()) if len(explore) else 0,
                             n_illicit=int(ys.sum()), w_adp=round(w_adp, 4),
                             **pool_info))
        if gate == "static":
            continue

        revealed[s] = idx_s[chosen]
        for j in exploit:
            prop_of[idx_s[j]] = 1.0
        for j in explore:
            prop_of[idx_s[j]] = prop
        if retrainer == "window_ri":
            m = np.ones(len(idx_s), bool); m[chosen] = False
            uninvest[s] = idx_s[m]
        if s >= 43:
            pos_post += int(ys[chosen].sum())

        if gate == "competence" and model is not None and not cumulative:
            yl = ys[chosen]
            if (yl == 1).any() and (yl == 0).any():
                comp["base"] = ema * comp["base"] + (1 - ema) * safe_ap(yl, p_base[chosen])
                comp["adp"] = ema * comp["adp"] + (1 - ema) * safe_ap(yl, p_adp[chosen])

        # ---------------- retrain ----------------
        if cumulative:
            avail = [q for q in revealed if q <= s - LABEL_DELAY]
            if avail:
                new = np.concatenate([revealed[q] for q in avail])
                pre = D["pre_idx"][train_end] if retrainer != "alerts_all" else np.array([], int)
                tr = np.concatenate([pre, new]).astype(int)
                sw = None
                if retrainer == "cumulative_ipw":
                    sw = np.concatenate([np.ones(len(pre)), [1.0 / prop_of[i] for i in new]])
                if (y[tr] == 1).sum() >= min_pos and (y[tr] == 0).sum() >= MIN_NEG:
                    model = fit_lgbm(X[tr], y[tr], seed, sample_weight=sw)
                    pool_info = dict(pool_pos_new=int((y[new] == 1).sum()),
                                     pool_pos_new_post=int(((y[new] == 1) & (t[new] >= 43)).sum()),
                                     pool_n=int(len(tr)))
        elif psi_z[s] >= ALARM_Z or model is not None:
            avail = [q for q in revealed if s - LABEL_DELAY - window < q <= s - LABEL_DELAY]
            if avail:
                pool = np.concatenate([revealed[q] for q in avail])
                yp = y[pool]
                sw = None
                if retrainer == "window_ipw":
                    sw = np.array([1.0 / prop_of[i] for i in pool])
                if retrainer == "window_ri":
                    neg = np.concatenate([uninvest[q] for q in avail])
                    pool = np.concatenate([pool, neg])
                    yp = np.concatenate([yp, np.zeros(len(neg), int)])
                if (yp == 1).sum() >= min_pos and (yp == 0).sum() >= MIN_NEG:
                    model = fit_lgbm(X[pool], yp, seed, sample_weight=sw)
                    model_ref = X[np.isin(t, avail)]
                    tp = t[pool]
                    pool_info = dict(pool_pos_pre=int(((yp == 1) & (tp < 43)).sum()),
                                     pool_pos_post=int(((yp == 1) & (tp >= 43)).sum()),
                                     first_fit=pool_info.get("first_fit", s))
    return rows
