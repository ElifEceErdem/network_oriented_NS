#!/usr/bin/env python3
"""
Validate that the PL-NS (naive_random_walk) optimization is faithful:
the cached per-user candidate structures must equal an original-style rebuild,
element-for-element and in the same order. Since the optimization only replaces
this deterministic rebuild (the rng draws are untouched and in the same order),
equality here implies the produced negative samples are identical for add_randomness=0.

Run from the code dir with the networkit_env python:
  cd LightGCN-PyTorch/code
  python /.../revision_experiments/validate_pl_cache.py
"""
import sys
import os
import random

# Make the LightGCN code importable and switch into it (relative ../data paths).
CODE = "/var/scratch/eerdem/network_oriented_NS/LightGCN-PyTorch/code"
sys.path.insert(0, CODE)
os.chdir(CODE)

# Configure argv the way main.py would for a naive_random_walk run, BEFORE importing world.
sys.argv = ["validate", "--dataset=lastfm", "--neg_sample=naive_random_walk",
            "--neg_samp_strategy=normal", "--add_randomness=0", "--positem=10",
            "--negitem=2", "--seed=2020", "--epochs=1", "--tensorboard=0", "--topks=[10]"]

import numpy as np  # noqa: E402
import world        # noqa: E402  (parses argv at import)
import utils        # noqa: E402
from register import dataset  # noqa: E402  (builds the LastFM dataset incl. path_length dicts)

print("building PL-NS caches ...")
base_filtered_list, base_filtered_dict = utils._naive_bases(dataset)

allPos = dataset.allPos
path_length_dict = dataset.path_length_dict
path_length_prob_dict = dataset.path_length_prob_dict

# Pick users that actually have positives (those the sampler would process).
rng = random.Random(123)
candidate_users = [u for u in range(dataset.n_users) if len(allPos[u]) > 0]
sample_users = rng.sample(candidate_users, k=min(80, len(candidate_users)))

mismatches = 0
checked = 0
for user in sample_users:
    user_id = 'u_' + str(user)
    if user_id not in path_length_prob_dict:
        continue
    prefixed_all_pos = ['i_' + str(x) for x in allPos[user]]
    # original-style rebuild (exactly the replaced lines)
    fl = [item for item in path_length_prob_dict[user_id] if item.startswith('i_')]
    fl = [x for x in fl if x not in prefixed_all_pos]
    all_path = path_length_dict[user_id]
    path_for_user = {k: v for k, v in all_path.items() if not k.startswith('u_')}
    fd = {k: v for k, v in path_for_user.items() if k not in prefixed_all_pos}

    checked += 1
    if fl != base_filtered_list.get(user_id):
        mismatches += 1
        print(f"  LIST MISMATCH user={user} len_orig={len(fl)} len_cache={len(base_filtered_list.get(user_id, []))}")
    if fd != base_filtered_dict.get(user_id):
        mismatches += 1
        print(f"  DICT MISMATCH user={user}")

print(f"checked {checked} users; mismatches={mismatches}")
print("RESULT:", "PASS - cache is byte-identical to original rebuild" if mismatches == 0 else "FAIL")
sys.exit(1 if mismatches else 0)