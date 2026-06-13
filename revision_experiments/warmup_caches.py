#!/usr/bin/env python3
"""
Warm the DD-NS caches for ONE distance matrix (tau), so the 360 parallel DD array tasks all
load a prebuilt cache instead of racing to build it. Builds:
  ns_cache/distance_dict_<tag>.pkl   (parsed matrix; via the dataloader)
  ns_cache/commute_bases_<tag>.pkl   (per-user {item: distance} minus positives)

Usage:  python warmup_caches.py /path/to/distance_tau<X>.csv
"""
import sys
import os

CODE = "/var/scratch/eerdem/network_oriented_NS/LightGCN-PyTorch/code"
sys.path.insert(0, CODE)
os.chdir(CODE)

cmp = sys.argv[1]
sys.argv = ["warmup", "--dataset=lastfm", "--neg_sample=commute_distance",
            "--neg_samp_strategy=scaled", "--commute_matrix_path=" + cmp,
            "--positem=10", "--negitem=1", "--seed=2020", "--epochs=1",
            "--tensorboard=0", "--topks=[10]", "--add_randomness=0"]

import world  # noqa: E402  (parses argv at import)
import utils   # noqa: E402
from register import dataset  # noqa: E402  (dataloader builds distance_dict if needed)

base = utils._commute_bases(dataset)
print("WARMED commute_bases for %s : %d users" % (cmp, len(base)))
