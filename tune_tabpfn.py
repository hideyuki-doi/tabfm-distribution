# TabPFN 3.5 configuration search for presence-background SDM (v40 protocol).
# To keep the comparison with Valavi et al. fair, configurations are selected on a
# development half of the species (random 50% within each region) and then
# confirmed on the held-out half.
# Usage: python tune_tabpfn.py --split dev --variants V0,V1,...   -> results/v40/tuning/tune.csv
import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
from scipy.special import expit, logit

import benchmark_v40_valavi as B
from valavi_config import HERE, REGIONS

OUT = os.path.join(HERE, "results", "v40", "tuning", "tune.csv")

# name: (n_bags, presences resampled?, background per bag, n_estimators, extra kwargs)
#   background: ("ratio", r) -> r * n_p points;  ("min", k) -> max(n_p, k) points
VARIANTS = {
    "V0_baseline":      dict(bags=10, boot=True,  bg=("ratio", 1)),
    "V1_allpres":       dict(bags=10, boot=False, bg=("ratio", 1)),
    "V2_bg3":           dict(bags=10, boot=False, bg=("ratio", 3)),
    "V3_bg10":          dict(bags=10, boot=False, bg=("ratio", 10)),
    "V4_bg2000":        dict(bags=10, boot=False, bg=("min", 2000)),
    "V5_bg10000":       dict(bags=3,  boot=False, bg=("min", 10000)),
    "V6_allpres_est16": dict(bags=10, boot=False, bg=("ratio", 1), n_estimators=16),
    "V7_bg3_est16":     dict(bags=10, boot=False, bg=("ratio", 3), n_estimators=16),
}


def split_of(region, i, n):
    rng = np.random.default_rng([7, REGIONS.index(region)])
    dev = set(rng.permutation(n)[: n // 2])
    return "dev" if i in dev else "holdout"


def run_variant(d, region, cfg, rng):
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion
    train = pd.concat([d["pres"], d["bg"]], ignore_index=True)
    P = B.Prep(train, region)
    Xp, Xb, Xt = P.coded(d["pres"]), P.coded(d["bg"]), P.coded(d["test"])
    n_p, n_b = len(Xp), len(Xb)
    kind, k = cfg["bg"]
    n_bg = min(n_b, int(k * n_p) if kind == "ratio" else max(n_p, int(k)))
    logits = []
    for _ in range(cfg["bags"]):
        ip = rng.choice(n_p, n_p, replace=True) if cfg["boot"] else np.arange(n_p)
        ib = rng.choice(n_b, n_bg, replace=n_bg > n_b)
        X = np.vstack([Xp[ip], Xb[ib]])
        y = np.r_[np.ones(len(ip)), np.zeros(len(ib))]
        kw = dict(device=B.DEVICE, random_state=int(rng.integers(1 << 31)),
                  categorical_features_indices=P.cat_idx or None, ignore_pretraining_limits=True)
        if "n_estimators" in cfg:
            kw["n_estimators"] = cfg["n_estimators"]
        tf = TabPFNClassifier.create_default_for_version(ModelVersion("v3.5"), **kw).fit(X, y)
        logits.append(logit(np.clip(tf.predict_proba(Xt)[:, 1], 1e-6, 1 - 1e-6)))
    return expit(np.mean(logits, axis=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "holdout", "all"])
    ap.add_argument("--variants", default=",".join(VARIANTS))
    args = ap.parse_args()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    variants = args.variants.split(",")
    done = set()
    if os.path.exists(OUT):
        prev = pd.read_csv(OUT)
        done = set(zip(prev.Region, prev.Species, prev.Variant))
    for region in REGIONS:
        sp = B.load_region(region)
        for i, d in enumerate(sp):
            s = split_of(region, i, len(sp))
            if args.split != "all" and s != args.split:
                continue
            rows = []
            for v in variants:
                if (region, d["species"], v) in done:
                    continue
                rng = np.random.default_rng([B.SEED, REGIONS.index(region), i, list(VARIANTS).index(v)])
                t0 = time.time()
                torch.cuda.empty_cache()
                p = run_variant(d, region, VARIANTS[v], rng)
                rows.append({"Region": region, "Species": d["species"], "split": s, "Variant": v,
                             "n_pres": len(d["pres"]), **B.calc_metrics(d["y_test"], p),
                             "Time": time.time() - t0})
            if rows:
                pd.DataFrame(rows).to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)
                print(f"[{region} {d['species']} {s} n_p={len(d['pres'])}] " +
                      " ".join(f"{r['Variant'].split('_')[0]}={r['AUC']:.3f}({r['Time']:.0f}s)" for r in rows),
                      flush=True)


if __name__ == "__main__":
    main()
