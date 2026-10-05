"""ICAIF'26 camera-ready experiments (responses to Reviewer 2 / Reviewer 3).

Run from the repo root with the SAME interpreter that produced the paper's
numbers (the project .venv), so the base models are bit-identical:

    caffeinate -i python camera_ready/run_all.py               # everything (resumable)
    python camera_ready/run_all.py --only repro                 # ~2 min sanity check
    caffeinate -i python camera_ready/run_all.py --workers 4    # fewer cores / less RAM

Stage 0 fits the two base detectors once, caches them, and checks that this
re-implementation reproduces numbers already in results/metrics/. It stops if
they do not match (use --force to override). All later stages append to
camera_ready/results/*.jsonl and skip finished jobs, so it is safe to stop
(Ctrl-C) and restart at any time.
"""
import argparse, json, os, pickle, platform, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
from camera_ready.common import (load_xyt, fit_lgbm, psi_z_scores)
from camera_ready.sim import run

D_CACHE = ROOT / "data" / "processed" / "cr_D.pkl"
OUT = ROOT / "camera_ready" / "results"
OUT.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------- stage 0
def prep():
    if D_CACHE.exists():
        return
    print("[prep] loading data + fitting base detectors (step 34 and step 42) ...", flush=True)
    X, y, t = load_xyt()
    D = {"p_base": {}, "psi_z": {}, "novelty": {}, "pre_idx": {}}
    for te, (ref_end, calib) in {34: (34, 34), 42: (42, 34)}.items():
        bm = (t <= te) & (y != -1)
        D["p_base"][te] = fit_lgbm(X[bm], y[bm], 42).predict_proba(X)[:, 1]
        D["psi_z"][te] = psi_z_scores(X, t, ref_end, calib)
        ref = X[t <= te]
        mu, sd = ref.mean(0), ref.std(0) + 1e-8
        D["novelty"][te] = np.abs((X - mu) / sd).mean(1)
        D["pre_idx"][te] = np.where(bm)[0]
        print(f"[prep] base step-{te} done", flush=True)
    pickle.dump(D, open(D_CACHE, "wb"))


_D = None
def _init():
    global _D
    import warnings; warnings.filterwarnings("ignore")
    _D = pickle.load(open(D_CACHE, "rb"))
    _D["X"], _D["y"], _D["t"] = load_xyt()


def _work(job):
    exp, jid, kw = job
    t0 = time.time()
    rows = run(_D, **kw)
    return exp, jid, kw, rows, round(time.time() - t0, 2)


def tot(rows, key):
    return int(sum(r[key] for r in rows if r["time_step"] >= 43))


def repro():
    """Numbers already in results/metrics (produced on the author's machine)."""
    checks = [
        ("frozen step-34 (paper 14)", dict(train_end=34, gate="static"), "caught_topk", 14),
        ("frozen step-42 (paper 113)", dict(train_end=42, gate="static"), "caught_topk", 113),
        ("AEGIS alert-only greedy, step-42 (paper 93)", dict(train_end=42, nx_rule="canonical"), "caught_topk", 93),
        ("replace, step-42 (paper 74)", dict(train_end=42, gate="replace", nx_rule="canonical"), "caught_topk", 74),
        ("average, step-42 (paper 89)", dict(train_end=42, gate="average", nx_rule="canonical"), "caught_topk", 89),
        ("AEGIS eps=0.2 step-34 seed 42 (trap_seeds 18)", dict(train_end=34, policy="eps20", nx_rule="canonical", seed=42), "caught_topk", 18),
        ("frontier greedy k=50 (csv 10)", dict(train_end=34, k=50), "caught_inv", 10),
        ("frontier greedy k=75 (csv 19)", dict(train_end=34, k=75), "caught_inv", 19),
        ("frontier greedy k=95 (csv 52)", dict(train_end=34, k=95), "caught_inv", 52),
    ] + [(f"frontier eps10 k=75 seed {s} (csv {v})", dict(train_end=34, policy="eps10", k=75, seed=s), "caught_inv", v)
         for s, v in zip(range(42, 47), [37, 62, 21, 57, 51])]
    _init()
    out, ok = [], True
    for name, kw, key, expect in checks:
        got = tot(run(_D, **kw), key)
        good = got == expect
        ok &= good
        out.append(dict(check=name, expected=expect, got=got, match=good))
        print(f"  {'OK ' if good else 'XX '} {name:52s} expected {expect:4d}  got {got:4d}", flush=True)
    import lightgbm, sklearn
    env = dict(platform=platform.platform(), python=sys.version.split()[0],
               lightgbm=lightgbm.__version__, sklearn=sklearn.__version__,
               numpy=np.__version__, all_match=ok, checks=out)
    json.dump(env, open(OUT / "repro_check.json", "w"), indent=1)
    return ok


# ----------------------------------------------------------------- job list
def build_jobs():
    J = []
    add = lambda exp, kw: J.append((exp, json.dumps(kw, sort_keys=True), kw))
    C = dict(nx_rule="canonical")          # k=50 protocol of Sec 6 / Table 3
    K = list(range(50, 101, 5))

    # P0  KEY: practitioner retrainers that keep the historical labels vs ones that do not
    #     cumulative  = refit base on (all pre-deployment labels + every alert label so far)
    #     alerts_all  = refit on every alert label so far, NO historical labels (ablation)
    for te in (42, 34):
        for r in ("cumulative", "alerts_all"):
            add("key", dict(train_end=te, retrainer=r, **C))
        for s in range(42, 47):
            for r in ("cumulative", "cumulative_ipw"):
                add("key", dict(train_end=te, policy="eps10", seed=s, retrainer=r, **C))
        for s in range(42, 52):
            for pol in ("eps10", "eps20"):
                add("key", dict(train_end=te, policy=pol, seed=s, retrainer="alerts_all", **C))
    for k in (25, 100):                     # cumulative on the escape-frontier budgets
        add("key", dict(train_end=34, k=k, retrainer="cumulative"))

    # P1  bootstrap CIs for the headline (realistic step-42 base, k=50)
    for b in range(200):
        for kw in [dict(gate="static"), dict(), dict(gate="replace"), dict(gate="average"),
                   dict(policy="eps10", seed=1000 + b), dict(policy="eps20", seed=1000 + b),
                   dict(policy="eps10", retrainer="window_ipw", seed=1000 + b)]:
            add("boot42", dict(train_end=42, boot_seed=b, **C, **kw))
    # P1b bootstrap for cold-start (step-34 base, k=50): frozen, greedy, paper's eps=0.2
    for b in range(200):
        for kw in [dict(gate="static"), dict(), dict(policy="eps20", seed=1000 + b)]:
            add("boot34", dict(train_end=34, boot_seed=b, **C, **kw))

    # P2  gate / trigger sensitivity on the realistic base (R2: m, window, temperature, EMA)
    grid = [("min_pos", v) for v in (3, 5, 10, 20)] + [("window", v) for v in (2, 4, 8, 12)] + \
           [("tau", v) for v in (0.02, 0.05, 0.1, 0.2, 0.5)] + [("ema", v) for v in (0.3, 0.6, 0.9)]
    for p, v in grid:
        add("sens42", dict(train_end=42, **C, **{p: v}))
        for s in range(42, 52):
            add("sens42", dict(train_end=42, policy="eps10", seed=s, **C, **{p: v}))
    # P2b window length vs the stale-label mechanism on the cold-start frontier
    for w in (2, 4, 8, 12):
        for k in (50, 75, 100):
            add("window34", dict(train_end=34, k=k, window=w))
            for s in range(42, 62):
                add("window34", dict(train_end=34, k=k, window=w, policy="eps10", seed=s))

    # P3  cheap practitioner baselines (k=50), both bases
    for te in (34, 42):
        add("base_cmp", dict(train_end=te, gate="static"))
        add("base_cmp", dict(train_end=te, **C))
        add("base_cmp", dict(train_end=te, retrainer="window_ri", **C))
        for s in range(42, 72):
            for pol in ("eps10", "eps20"):
                add("base_cmp", dict(train_end=te, policy=pol, seed=s, **C))
                add("base_cmp", dict(train_end=te, policy=pol, seed=s, retrainer="window_ipw", **C))
        for s in range(42, 52):
            for pol in ("eps10", "eps20"):
                add("base_cmp", dict(train_end=te, policy=pol, seed=s, retrainer="window_ri", **C))

    # P4  ignition: (A) paper protocol with 50 seeds; (B) transaction bootstrap, paired across policies
    for k in K:
        add("ignite_seeds", dict(train_end=34, k=k))
        for s in range(42, 92):
            for pol in ("eps10", "eps20"):
                add("ignite_seeds", dict(train_end=34, k=k, policy=pol, seed=s))
    for b in range(40):
        for k in K:
            for pol in ("greedy", "eps10", "eps20"):
                add("ignite_boot", dict(train_end=34, k=k, policy=pol, seed=2000 + b, boot_seed=b))

    # P5  mediation: remaining policies of the paper's frontier
    for k in (50, 75, 100):
        add("mediation", dict(train_end=34, k=k, policy="novelty"))
        for s in range(42, 62):
            for pol in ("eps40", "stratified"):
                add("mediation", dict(train_end=34, k=k, policy=pol, seed=s))

    # P6  more cumulative seeds (expensive: refits the full base every step)
    for te in (42, 34):
        for s in range(47, 52):
            add("cumulative", dict(train_end=te, policy="eps10", seed=s, retrainer="cumulative", **C))
            add("cumulative", dict(train_end=te, policy="eps10", seed=s, retrainer="cumulative_ipw", **C))
        for s in range(42, 47):
            add("cumulative", dict(train_end=te, policy="eps20", seed=s, retrainer="cumulative", **C))
            add("cumulative", dict(train_end=te, policy="eps20", seed=s, retrainer="cumulative_ipw", **C))

    # P7  expensive bootstraps: cumulative and bulk feedback
    for b in range(40):
        add("boot_cum", dict(train_end=42, boot_seed=b, retrainer="cumulative", **C))
        add("boot_cum", dict(train_end=42, boot_seed=b, retrainer="window_ri", **C))
    for b in range(20):
        add("boot_cum", dict(train_end=34, boot_seed=b, retrainer="cumulative", **C))
    for b in range(30):
        add("boot_bulk", dict(train_end=34, boot_seed=b, feedback="bulk", gate="psi"))
        add("boot_bulk", dict(train_end=42, boot_seed=b, feedback="bulk", gate="psi"))
    add("boot_bulk", dict(train_end=34, feedback="bulk", gate="psi"))
    add("boot_bulk", dict(train_end=42, feedback="bulk", gate="psi"))
    return J


def done_ids():
    done = set()
    for f in OUT.glob("*.jsonl"):
        for line in open(f):
            try:
                done.add((f.stem, json.loads(line)["jid"]))
            except Exception:
                pass
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, min(6, (os.cpu_count() or 4) - 2)))
    ap.add_argument("--only", nargs="*", help="experiment names to run (or 'repro')")
    ap.add_argument("--force", action="store_true", help="continue even if repro check fails")
    args = ap.parse_args()

    prep()
    print("\n[stage 0] reproduction check against results/metrics/", flush=True)
    ok = repro()
    if not ok and not args.force:
        print("\nReproduction check FAILED - stopping. Send camera_ready/results/repro_check.json.")
        sys.exit(1)
    print("[stage 0] all checks match.\n" if ok else "[stage 0] mismatches ignored (--force).\n")
    if args.only == ["repro"]:
        return

    jobs = build_jobs()
    if args.only:
        jobs = [j for j in jobs if j[0] in args.only]
    done = done_ids()
    todo = [j for j in jobs if (j[0], j[1]) not in done]
    print(f"[run] {len(todo)} of {len(jobs)} jobs to do, {args.workers} workers. "
          f"Safe to Ctrl-C and re-run; finished jobs are kept.", flush=True)
    if not todo:
        return

    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    files, t0, n = {}, time.time(), 0
    try:
        with ctx.Pool(args.workers, initializer=_init) as pool:
            for exp, jid, kw, rows, secs in pool.imap_unordered(_work, todo, chunksize=1):
                f = files.get(exp) or files.setdefault(exp, open(OUT / f"{exp}.jsonl", "a"))
                f.write(json.dumps(dict(jid=jid, kw=kw, secs=secs, rows=rows)) + "\n"); f.flush()
                n += 1
                if n % 25 == 0 or n == len(todo):
                    el = time.time() - t0
                    print(f"[run] {n}/{len(todo)} done | {el/60:5.1f} min elapsed | "
                          f"~{el/n*(len(todo)-n)/60:5.1f} min left | last: {exp}", flush=True)
    except KeyboardInterrupt:
        print("\n[run] interrupted - progress saved; re-run the same command to continue.")
    finally:
        for f in files.values():
            f.close()
    print("[run] finished." if n == len(todo) else "")


if __name__ == "__main__":
    main()
