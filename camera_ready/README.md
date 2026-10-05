# Camera-ready experiments (ICAIF '26 reviews)

Run from the repo root with the project `.venv` (the interpreter that produced
the paper's numbers — LightGBM results differ slightly across OS builds):

    caffeinate -i python camera_ready/run_all.py

* Stage 0 fits the two base detectors, caches them in `data/processed/`, and
  checks that this code reproduces numbers already in `results/metrics/`.
  It stops if they don't match.
* Then ~5.5k simulation jobs run on several cores (default: CPUs-2, max 6).
  Output: `camera_ready/results/*.jsonl`. Ctrl-C any time; re-running resumes.
* `--only key` runs just the most important block; `--workers 4` for less RAM.

| block | what | reviewer point |
|---|---|---|
| key | cumulative retraining / all-alert-labels ablation | R2: practitioner baselines |
| boot42, boot34 | transaction bootstrap CIs for 113/93, 14/19 | R2: no bootstrap |
| sens42, window34 | trigger m, window, temperature, EMA | R2: no sensitivity |
| base_cmp | IPW, reject inference, 30-seed AEGIS | R2: IPW / reject inference |
| ignite_seeds, ignite_boot | ignition with 50 seeds / 40 bootstrap reps | R2: 5 seeds per point |
| mediation | remaining policies for the within-step analysis | R2: confounded "law" |
| cumulative, boot_cum, boot_bulk | more seeds / CIs for the expensive arms | R2 |

## Reproducing the paper's tables and figures from the results

    python camera_ready/analyze.py                       # summary.txt, ignition table, diagnostics
    python camera_ready/make_fig7.py camera_ready/results paper_figures/frontier_cr
    python camera_ready/make_fig_trap.py paper_figures/trap_cr

`camera_ready/results/` holds the raw per-run outputs (JSONL) behind every number
in Sections 6-8 of the ICAIF '26 camera-ready paper (macOS/arm64, LightGBM 4.6.0).
