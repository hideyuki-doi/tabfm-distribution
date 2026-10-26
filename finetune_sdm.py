# Leave-one-region-out fine-tuning of TabPFN 3.5 on presence-background SDM tasks (pilot).
#
# Training data: every species of the other five regions; each epoch draws a new balanced
# sample per species (all presences + the same number of random background points), and
# TabPFN's own fine-tuning loop learns to predict held-out points of each sample from the
# rest (context/query split). The held-out region is never seen. Evaluation: the v40
# "Fast" protocol (5 balanced bags) on the held-out region, default vs fine-tuned weights.
# Categorical covariates are treated as numeric codes in BOTH arms (the fine-tuning loop
# uses one global categorical setting), so the comparison is like-for-like.
#
# Usage: python finetune_sdm.py --holdout CAN [--epochs 10] [--lr 1e-5]
import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import expit, logit

import benchmark_v40_valavi as B
import tabpfn.finetuning.finetuned_base as FB
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion
from tabpfn.finetuning import FinetunedTabPFNClassifier
from tabpfn.model_loading import save_tabpfn_model
from valavi_config import HERE, REGIONS

OUT = Path(HERE) / "results" / "v40" / "finetune"
MAX_PRES = 2000  # cap presences per species sample (SWI has up to 5,822)


def load_tasks(regions):
    tasks = []
    for r in regions:
        for d in B.load_region(r):
            train = pd.concat([d["pres"], d["bg"]], ignore_index=True)
            P = B.Prep(train, r)
            tasks.append({"region": r, "species": d["species"], "Xp": P.coded(d["pres"]), "Xb": P.coded(d["bg"]),
                          "Xt": P.coded(d["test"]), "y_test": d["y_test"]})
    return tasks


def balanced_samples(tasks, seed):
    rng = np.random.default_rng(seed)
    Xs, ys = [], []
    for t in tasks:
        n_p = len(t["Xp"])
        ip = rng.permutation(n_p)[:MAX_PRES]
        ib = rng.choice(len(t["Xb"]), len(ip), replace=False)
        X = np.vstack([t["Xp"][ip], t["Xb"][ib]]).astype(np.float32)
        y = np.r_[np.ones(len(ip)), np.zeros(len(ib))].astype(np.int64)
        Xs.append(X)
        ys.append(y)
    return Xs, ys


def finetune(train_tasks, epochs, lr, out_path, seed=0):
    orig = FB.get_preprocessed_dataset_chunks

    def patched(calling_instance, X_raw, y_raw, split_fn, max_data_size, model_type, **kw):
        # replace the single training set by one balanced sample per training species
        Xs, ys = balanced_samples(train_tasks, kw.get("data_shuffle_seed", 0))
        return orig(calling_instance, Xs, ys, split_fn, max_data_size, model_type, **kw)

    FB.get_preprocessed_dataset_chunks = patched
    try:
        ft = FinetunedTabPFNClassifier(
            device="cuda", epochs=epochs, learning_rate=lr, validation_split_ratio=None, early_stopping=False,
            n_finetune_ctx_plus_query_samples=20_000, finetune_ctx_query_split_ratio=0.3,
            n_estimators_finetune=2, n_estimators_final_inference=2, random_state=seed,
            save_checkpoint_interval=None, model_version=ModelVersion.V3_5,
            extra_classifier_kwargs={"ignore_pretraining_limits": True})
        X0, y0 = balanced_samples(train_tasks[:1], seed)
        ft.fit(X0[0], y0[0])
    finally:
        FB.get_preprocessed_dataset_chunks = orig
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_tabpfn_model(model=ft.finetuned_estimator_, save_path=out_path)
    return out_path


def predict_fast(t, model_path, seed, bags=5):
    rng = np.random.default_rng(seed)
    n_p, n_b = len(t["Xp"]), len(t["Xb"])
    logits = []
    for _ in range(bags):
        ip = rng.choice(n_p, n_p, replace=True)
        ib = rng.choice(n_b, n_p, replace=True)
        X = np.vstack([t["Xp"][ip], t["Xb"][ib]])
        y = np.r_[np.ones(n_p), np.zeros(n_p)]
        kw = dict(device="cuda", random_state=int(rng.integers(1 << 31)), ignore_pretraining_limits=True)
        if model_path is not None:
            kw["model_path"] = str(model_path)
        tf = TabPFNClassifier.create_default_for_version(ModelVersion.V3_5, **kw).fit(X, y)
        logits.append(logit(np.clip(tf.predict_proba(t["Xt"])[:, 1], 1e-6, 1 - 1e-6)))
    return expit(np.mean(logits, axis=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max-eval-species", type=int, default=0)
    args = ap.parse_args()
    tag = f"{args.holdout}_e{args.epochs}_lr{args.lr:g}"
    train_regions = [r for r in REGIONS if r != args.holdout]
    t0 = time.time()
    train_tasks = load_tasks(train_regions)
    test_tasks = load_tasks([args.holdout])
    if args.max_eval_species:
        test_tasks = test_tasks[: args.max_eval_species]
    print(f"[{tag}] fine-tuning on {len(train_tasks)} species from {train_regions}", flush=True)
    ckpt = finetune(train_tasks, args.epochs, args.lr, OUT / f"tabpfn35_ft_{tag}.ckpt")
    print(f"[{tag}] fine-tuned in {(time.time() - t0) / 60:.1f} min -> {ckpt}", flush=True)

    rows = []
    for i, t in enumerate(test_tasks):
        torch.cuda.empty_cache()
        seed = [B.SEED, REGIONS.index(args.holdout), i]
        for arm, mp in [("default", None), ("finetuned", ckpt)]:
            p = predict_fast(t, mp, seed)
            rows.append({"holdout": args.holdout, "tag": tag, "Species": t["species"], "arm": arm,
                         "n_pres": len(t["Xp"]), **B.calc_metrics(t["y_test"], p)})
        r = rows[-2:]
        print(f"  {t['species']} n_p={len(t['Xp'])}: default={r[0]['AUC']:.3f} finetuned={r[1]['AUC']:.3f}", flush=True)
    df = pd.DataFrame(rows)
    f = OUT / "finetune_results.csv"
    df.to_csv(f, mode="a", header=not f.exists(), index=False)
    piv = df.pivot_table(index="Species", columns="arm", values="AUC")
    print(f"[{tag}] mean AUC default={piv['default'].mean():.4f} finetuned={piv['finetuned'].mean():.4f} "
          f"wins={int((piv.finetuned > piv.default).sum())}/{len(piv)}", flush=True)


if __name__ == "__main__":
    main()
