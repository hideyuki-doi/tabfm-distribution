# Ensemble of fine-tuned TabPFN 3.5 models (leave-one-region-out checkpoints).
# Rule fixed before evaluation: average (logit scale) the predictions of ALL fine-tuned models
# trained on presence-background structure: PB (finetune_sdm.py), VS and mixed (finetune_vs.py),
# 5 bags each = 15 bags. Control: the PB model alone with 15 bags (same compute), so that a gain
# from combining models is not confused with a gain from more bags.
# Protocol as in the fine-tuning comparisons (categorical covariates as numeric codes).
# Output: results/v40/finetune/finetune_ens_results.csv and per-species predictions.
import os
import sys

import numpy as np
import pandas as pd
import torch
from scipy.special import expit, logit

import benchmark_v40_valavi as B
from finetune_sdm import OUT, load_tasks
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion
from valavi_config import REGIONS

CKPT = {"pb": "tabpfn35_ft_{R}_e10_lr1e-05.ckpt", "vs": "tabpfn35_vsft_{R}_e5_lr1e-05.ckpt",
        "mix": "tabpfn35_mixft_{R}_e5_lr1e-05.ckpt"}
PRED = OUT / "ens_preds"


def bag_logits(t, model_path, seed, bags):
    """Same bag sequence as finetune_sdm.predict_fast (its first 5 bags are identical)."""
    rng = np.random.default_rng(seed)
    n_p, n_b = len(t["Xp"]), len(t["Xb"])
    out = []
    for _ in range(bags):
        ip = rng.choice(n_p, n_p, replace=True)
        ib = rng.choice(n_b, n_p, replace=True)
        X = np.vstack([t["Xp"][ip], t["Xb"][ib]])
        y = np.r_[np.ones(n_p), np.zeros(n_p)]
        tf = TabPFNClassifier.create_default_for_version(
            ModelVersion.V3_5, device="cuda", random_state=int(rng.integers(1 << 31)),
            ignore_pretraining_limits=True, model_path=str(model_path)).fit(X, y)
        out.append(logit(np.clip(tf.predict_proba(t["Xt"])[:, 1], 1e-6, 1 - 1e-6)))
    return np.array(out)


def main(regions):
    PRED.mkdir(parents=True, exist_ok=True)
    f = OUT / "finetune_ens_results.csv"
    done = set(pd.read_csv(f).holdout) if f.exists() else set()
    for R in regions:
        if R in done:
            continue
        rows = []
        for i, t in enumerate(load_tasks([R])):
            torch.cuda.empty_cache()
            seed = [B.SEED, REGIONS.index(R), i]
            L = {"pb": bag_logits(t, OUT / CKPT["pb"].format(R=R), seed, 15)}
            for k in ("vs", "mix"):
                L[k] = bag_logits(t, OUT / CKPT[k].format(R=R), seed, 5)
            preds = {
                "pb_finetuned": expit(L["pb"][:5].mean(0)),
                "pb_finetuned_15bags": expit(L["pb"].mean(0)),
                "vs_finetuned": expit(L["vs"].mean(0)),
                "mix_finetuned": expit(L["mix"].mean(0)),
                "ft_ensemble": expit(np.vstack([L["pb"][:5], L["vs"], L["mix"]]).mean(0)),
            }
            np.savez_compressed(PRED / f"{R}_{t['species']}.npz", y_test=t["y_test"].astype(np.int8),
                                **{k: v.astype(np.float32) for k, v in preds.items()})
            for arm, p in preds.items():
                rows.append({"holdout": R, "Species": t["species"], "arm": arm, "n_pres": len(t["Xp"]),
                             **B.calc_metrics(t["y_test"], p)})
            r = {x["arm"]: x["AUC"] for x in rows[-5:]}
            print(f"  {R} {t['species']}: " + " ".join(f"{k}={v:.3f}" for k, v in r.items()), flush=True)
        df = pd.DataFrame(rows)
        df.to_csv(f, mode="a", header=not f.exists(), index=False)
        print(f"[{R}] " + " ".join(f"{a}={g.AUC.mean():.4f}" for a, g in df.groupby("arm")), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or REGIONS)
