#!/usr/bin/env python3
"""
Generate a TSV manifest: one line per (config x seed) run for the revision experiments.

run_array.job reads the N-th data line (by SLURM_ARRAY_TASK_ID) and launches main.py with
exactly those arguments. Metrics are captured from each run's stdout log and aggregated by
parse_results.py.

Scope (per user decisions): the 6 ready rows -- PD-NS (all_simple_paths) is deferred.
  baselines : uniform, alpha75 (popularity)             [negitem/tau ignored by these samplers]
  PL-NS     : naive_random_walk 'normal' (Long-Distance), add_randomness in {0,1}
  DD-NS     : commute_distance 'scaled' (as-published == uniform), add_randomness in {0,1}
  DD-NS*    : commute_distance 'scaled_weighted' (CORRECTED distance-weighted), add_randomness in {0,1}

Sensitivity = one-factor-at-a-time around the reported config (negitem=1, tau=1):
  - negitem sweep {1,2,3,4,5} at tau=1
  - tau sweep {0.5,1,2,4,8} at negitem=1   (DD-NS only; PL-NS has no tau)

Columns (tab-separated, with header):
  runtag  neg_sample  neg_samp_strategy  add_randomness  positem  negitem  tau
  commute_matrix_path  seed  epochs  dataset  recdim  layer  lr  decay  topks  multicore
"""
import argparse
import os

DATA_LASTFM = "/var/scratch/eerdem/network_oriented_NS/LightGCN-PyTorch/data/lastfm"

FIXED = dict(dataset="lastfm", recdim=64, layer=3, lr=0.001, decay=1e-4,
             topks="[10]", multicore=1)

COLUMNS = ["runtag", "neg_sample", "neg_samp_strategy", "add_randomness", "positem",
           "negitem", "tau", "commute_matrix_path", "seed", "epochs",
           "dataset", "recdim", "layer", "lr", "decay", "topks", "multicore"]


def tau_path(tau):
    return os.path.join(DATA_LASTFM, f"distance_tau{tau}.csv")


def row(neg_sample, strategy, ar, positem, negitem, tau, seed, epochs):
    # use "na" (not "") so empty fields don't collapse under tab-IFS `read` in run_array.job
    cmp = "na" if tau is None else tau_path(tau)
    t = "na" if tau is None else f"{tau}"
    rt = f"{neg_sample}-{strategy}-ar{ar}-pi{positem}-ni{negitem}-tau{t}-s{seed}"
    return [rt, neg_sample, strategy, str(ar), str(positem), str(negitem), t, cmp,
            str(seed), str(epochs), FIXED["dataset"], str(FIXED["recdim"]),
            str(FIXED["layer"]), str(FIXED["lr"]), str(FIXED["decay"]),
            FIXED["topks"], str(FIXED["multicore"])]


def dd_configs(strategy, negitems, taus, base_ni, base_tau):
    """one-factor-at-a-time (negitem@base_tau) + (tau@base_ni), de-duplicated."""
    cfgs = set()
    for ni in negitems:
        cfgs.add((ni, base_tau))
    for tau in taus:
        cfgs.add((base_ni, tau))
    return sorted(cfgs)


def build(phase, seeds, epochs, negitems, taus, positem, base_ni, base_tau):
    rows = []
    if phase in ("baseline", "all"):
        for seed in seeds:
            rows.append(row("uniform", "na", 0, positem, 1, None, seed, epochs))
            rows.append(row("alpha75", "na", 0, positem, 1, None, seed, epochs))

    if phase == "convergence":
        rows.append(row("uniform", "na", 0, positem, 1, None, seeds[0], epochs))

    if phase in ("graph", "all"):
        # PL-NS: negitem sweep, no tau
        for ar in (0, 1):
            for ni in negitems:
                for seed in seeds:
                    rows.append(row("naive_random_walk", "normal", ar, positem, ni, None, seed, epochs))
        # DD-NS as-published ('scaled') and corrected ('scaled_weighted')
        for strategy in ("scaled", "scaled_weighted"):
            for ar in (0, 1):
                for (ni, tau) in dd_configs(strategy, negitems, taus, base_ni, base_tau):
                    for seed in seeds:
                        rows.append(row("commute_distance", strategy, ar, positem, ni, tau, seed, epochs))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["baseline", "convergence", "graph", "all"], required=True)
    ap.add_argument("--out", default="manifest.tsv")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(2020, 2030)))
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--negitems", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--taus", type=str, nargs="+", default=["0.5", "1", "2", "4", "8"])
    ap.add_argument("--positem", type=int, default=10)
    ap.add_argument("--base_ni", type=int, default=1, help="reported negitem (held fixed in tau sweep)")
    ap.add_argument("--base_tau", type=str, default="1", help="reported tau (held fixed in negitem sweep)")
    args = ap.parse_args()

    rows = build(args.phase, args.seeds, args.epochs, args.negitems, args.taus,
                 args.positem, args.base_ni, args.base_tau)
    with open(args.out, "w") as f:
        f.write("\t".join(COLUMNS) + "\n")
        for r in rows:
            f.write("\t".join(r) + "\n")
    print(f"wrote {args.out}: {len(rows)} runs (phase={args.phase}, seeds={len(args.seeds)}, epochs={args.epochs})")
    print(f"  submit with:  sbatch --array=1-{len(rows)} --export=ALL,MANIFEST=$(pwd)/{args.out} run_array.job")


if __name__ == "__main__":
    main()
