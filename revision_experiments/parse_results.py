#!/usr/bin/env python3
"""
Aggregate experiment results from per-run stdout logs into a tidy CSV with mean +/- std
across seeds. Metrics are emitted by Procedure.Test() as lines like:

  {'precision': array([0.0252422]), 'recall': array([0.04875858]), 'ndcg': array([0.0401383])}

one per epoch. NOTE: Test() runs at the START of each epoch, so the i-th such line reflects
the model after i-1 epochs of training. We report, per run, the BEST epoch (max precision)
and the metrics there, plus the final-epoch metrics.

Runtags (and thus log filenames results/<runtag>.log) encode the config:
  <neg_sample>-<strategy>-ar<ar>-pi<positem>-ni<negitem>-tau<tau>-s<seed>
We group by everything except the trailing -s<seed> and aggregate across seeds.

Usage:
  python parse_results.py --results_dir results --out results/summary.csv [--metric precision]
"""
import argparse
import csv
import glob
import os
import re
import statistics

METRIC_RE = re.compile(
    r"\{'precision':\s*array\(\[([-\d.eE+]+)\]\),\s*"
    r"'recall':\s*array\(\[([-\d.eE+]+)\]\),\s*"
    r"'ndcg':\s*array\(\[([-\d.eE+]+)\]\)\}"
)
SEED_RE = re.compile(r"-s(\d+)$")


def parse_log(path):
    """Return list of (precision, recall, ndcg) per epoch, in order."""
    seq = []
    with open(path, errors="ignore") as f:
        for line in f:
            m = METRIC_RE.search(line)
            if m:
                seq.append(tuple(float(x) for x in m.groups()))
    return seq


def config_and_seed(runtag):
    m = SEED_RE.search(runtag)
    if not m:
        return runtag, None
    return runtag[:m.start()], int(m.group(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--out", default="results/summary.csv")
    ap.add_argument("--metric", choices=["precision", "recall", "ndcg"], default="precision",
                    help="metric used to pick the best epoch per run")
    args = ap.parse_args()
    midx = {"precision": 0, "recall": 1, "ndcg": 2}[args.metric]

    runs = {}  # config -> list of dict(seed, best_prec, best_recall, best_ndcg, best_epoch, n_epochs)
    per_run_rows = []
    for path in sorted(glob.glob(os.path.join(args.results_dir, "*.log"))):
        runtag = os.path.splitext(os.path.basename(path))[0]
        seq = parse_log(path)
        if not seq:
            print(f"WARN: no metrics parsed from {path}")
            continue
        best_i = max(range(len(seq)), key=lambda i: seq[i][midx])
        bp, br, bn = seq[best_i]
        fp, fr, fn = seq[-1]
        cfg, seed = config_and_seed(runtag)
        runs.setdefault(cfg, []).append((bp, br, bn, best_i, len(seq)))
        per_run_rows.append([runtag, cfg, seed, len(seq), best_i, bp, br, bn, fp, fr, fn])

    def ms(xs):
        if len(xs) == 1:
            return xs[0], 0.0
        return statistics.mean(xs), statistics.stdev(xs)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "n_seeds",
                    "precision_mean", "precision_std",
                    "recall_mean", "recall_std",
                    "ndcg_mean", "ndcg_std",
                    "best_epoch_mean"])
        for cfg in sorted(runs):
            vals = runs[cfg]
            pm, ps = ms([v[0] for v in vals])
            rm, rs = ms([v[1] for v in vals])
            nm, ns = ms([v[2] for v in vals])
            em = statistics.mean([v[3] for v in vals])
            w.writerow([cfg, len(vals),
                        f"{pm:.5f}", f"{ps:.5f}",
                        f"{rm:.5f}", f"{rs:.5f}",
                        f"{nm:.5f}", f"{ns:.5f}",
                        f"{em:.1f}"])

    # also dump per-run detail next to the summary
    detail = os.path.splitext(args.out)[0] + "_perrun.csv"
    with open(detail, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["runtag", "config", "seed", "n_epochs", "best_epoch",
                    "best_precision", "best_recall", "best_ndcg",
                    "final_precision", "final_recall", "final_ndcg"])
        w.writerows(per_run_rows)

    print(f"wrote {args.out} ({len(runs)} configs) and {detail} ({len(per_run_rows)} runs)")
    # quick console view
    for cfg in sorted(runs):
        vals = runs[cfg]
        pm, ps = ms([v[0] for v in vals])
        nm, ns = ms([v[2] for v in vals])
        print(f"  {cfg:55s}  n={len(vals):2d}  P@10={pm:.4f}+/-{ps:.4f}  NDCG@10={nm:.4f}+/-{ns:.4f}")


if __name__ == "__main__":
    main()