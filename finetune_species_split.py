# Species-split fine-tuning (as in Dinnage & Warren 2026), to test whether fine-tuning on other
# species of the SAME regions inflates the gain compared with leave-one-region-out (LORO).
#
# The 225 species are split into 5 folds at random within each region (seed 2026). For fold k,
# TabPFN 3.5 is fine-tuned on the species of the other four folds, from all six regions, and
# evaluated on the species of fold k. In this benchmark the species of a biological group share
# their survey sites, so with presence-absence supervision the test species' survey sites (with
# other species' labels) are seen during fine-tuning. Two arms, same settings as the LORO runs:
#   pb_species : presence-background tasks, 10 epochs (finetune_sdm.finetune)
#   pa_species : presence-absence supervision, 10 epochs (finetune_pa.finetune_pa)
# Evaluation: Fast (5 bags), categorical covariates as numeric codes, and the same per-species
# seeds as the LORO arms (finetune_pa_results.csv), so differences are due to the weights only.
# Output: results/v40/finetune/finetune_species_split_results.csv
import sys
import time

import numpy as np
import pandas as pd
import torch

import benchmark_v40_valavi as B
import finetune_pa as FPA
import finetune_sdm as FS
from finetune_sdm import OUT, load_tasks, predict_fast
from valavi_config import REGIONS

K = 5
RES = OUT / "finetune_species_split_results.csv"


def folds_of(tasks_by_region):
    """fold id for every (region, index) — random, balanced within each region"""
    rng = np.random.default_rng(2026)
    f = {}
    for r in REGIONS:
        n = len(tasks_by_region[r])
        ids = np.resize(np.arange(K), n)
        rng.shuffle(ids)
        for i in range(n):
            f[(r, i)] = int(ids[i])
    return f


def main(arms):
    tasks = {r: load_tasks([r]) for r in REGIONS}
    fold = folds_of(tasks)
    done = set()
    if RES.exists():
        prev = pd.read_csv(RES)
        done = set(zip(prev.arm, prev.fold))
    for arm in arms:
        for k in range(K):
            if (arm, k) in done:
                continue
            t0 = time.time()
            train = [t for r in REGIONS for i, t in enumerate(tasks[r]) if fold[(r, i)] != k]
            ckpt = OUT / f"tabpfn35_{arm}_fold{k}_e10_lr1e-05.ckpt"
            if not ckpt.exists():
                if arm == "pb_species":
                    FS.finetune(train, 10, 1e-5, ckpt)
                else:
                    FPA.finetune_pa(train, 10, 1e-5, ckpt)
            rows = []
            for r in REGIONS:
                for i, t in enumerate(tasks[r]):
                    if fold[(r, i)] != k:
                        continue
                    torch.cuda.empty_cache()
                    p = predict_fast(t, ckpt, [B.SEED, REGIONS.index(r), i])
                    rows.append({"holdout": r, "Species": t["species"], "arm": arm, "fold": k,
                                 "n_pres": len(t["Xp"]), **B.calc_metrics(t["y_test"], p)})
            df = pd.DataFrame(rows)
            df.to_csv(RES, mode="a", header=not RES.exists(), index=False)
            print(f"[{arm} fold {k}] n_train={len(train)} n_test={len(df)} AUC={df.AUC.mean():.4f} "
                  f"({(time.time() - t0) / 60:.0f} min)", flush=True)
    print("ALLDONE", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["pa_species", "pb_species"])
