# Presence-absence-supervised fine-tuning of TabPFN 3.5 (leave-one-region-out).
#
# Unlike finetune_sdm.py (context and query both drawn from presence-background data), each
# training task here uses a species' presence-background sample as CONTEXT (all presences, at
# most 2,000, plus the same number of random background points) and that species' independent
# presence-absence survey sites as QUERY, with the true presence/absence labels as the loss
# target. The model thus learns to go from biased presence-only evidence to true occurrence.
# Only the five training regions' surveys are used; the held-out region is never seen.
# NOTE: this uses other regions' presence-absence data, which the published models did not.
#
# Evaluation on the held-out region: "Fast" (5 balanced bags), categoricals as numeric codes,
# three arms: default weights, presence-background fine-tuned (finetune_sdm.py), PA fine-tuned.
# Usage: python finetune_pa.py --holdout AWT [--epochs 10] [--lr 1e-5]
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import benchmark_v40_valavi as B
import tabpfn.finetuning.finetuned_base as FB
from finetune_sdm import MAX_PRES, OUT, load_tasks, predict_fast
from tabpfn.constants import ModelVersion
from tabpfn.finetuning import FinetunedTabPFNClassifier
from tabpfn.model_loading import save_tabpfn_model
from valavi_config import REGIONS

MAX_QUERY = 2000  # survey sites per task and epoch (all presences kept up to half of this)


def pa_tasks(train_tasks, seed):
    """One task per species: rows = [context (presence-background) ; query (PA sites)]."""
    rng = np.random.default_rng(seed)
    Xs, ys, n_ctx_by_len = [], [], {}
    for t in train_tasks:
        y_pa = t["y_test"]
        if y_pa.min() == y_pa.max():
            continue
        ip = rng.permutation(len(t["Xp"]))[:MAX_PRES]
        ib = rng.choice(len(t["Xb"]), len(ip), replace=False)
        Xc = np.vstack([t["Xp"][ip], t["Xb"][ib]])
        yc = np.r_[np.ones(len(ip)), np.zeros(len(ib))]
        pos, neg = np.flatnonzero(y_pa == 1), np.flatnonzero(y_pa == 0)
        pos = rng.permutation(pos)[: MAX_QUERY // 2]
        neg = rng.permutation(neg)[: MAX_QUERY - len(pos)]
        q = rng.permutation(np.r_[pos, neg])
        # the split function only sees the array, so total lengths must identify the task
        while len(Xc) + len(q) in n_ctx_by_len and len(q) > 2:
            q = q[:-1]
        if len(Xc) + len(q) in n_ctx_by_len:
            continue
        X = np.vstack([Xc, t["Xt"][q]]).astype(np.float32)
        y = np.r_[yc, y_pa[q]].astype(np.int64)
        n_ctx_by_len[len(X)] = len(Xc)
        Xs.append(X)
        ys.append(y)
    return Xs, ys, n_ctx_by_len


def finetune_pa(train_tasks, epochs, lr, out_path, seed=0):
    orig = FB.get_preprocessed_dataset_chunks

    def patched(calling_instance, X_raw, y_raw, split_fn, max_data_size, model_type, **kw):
        Xs, ys, n_ctx_by_len = pa_tasks(train_tasks, kw.get("data_shuffle_seed", 0))

        def split(X, y, stratify=None):
            n = n_ctx_by_len[len(X)]
            # context = balanced presence-background sample (sanity check: half presences)
            assert abs(float(np.mean(np.asarray(y[:n]))) - 0.5) < 1e-6, "row order changed"
            return X[:n], X[n:], y[:n], y[n:]

        kw.update(shuffle=False)
        return orig(calling_instance, Xs, ys, split, None, model_type, **kw)

    FB.get_preprocessed_dataset_chunks = patched
    try:
        ft = FinetunedTabPFNClassifier(
            device="cuda", epochs=epochs, learning_rate=lr, validation_split_ratio=None, early_stopping=False,
            n_estimators_finetune=2, n_estimators_final_inference=2, random_state=seed,
            save_checkpoint_interval=None, model_version=ModelVersion.V3_5,
            extra_classifier_kwargs={"ignore_pretraining_limits": True})
        t = train_tasks[0]
        X0 = np.vstack([t["Xp"][:50], t["Xb"][:50]]).astype(np.float32)
        ft.fit(X0, np.r_[np.ones(len(t["Xp"][:50])), np.zeros(50)].astype(np.int64))
    finally:
        FB.get_preprocessed_dataset_chunks = orig
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_tabpfn_model(model=ft.finetuned_estimator_, save_path=out_path)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max-eval-species", type=int, default=0)
    args = ap.parse_args()
    tag = f"{args.holdout}_e{args.epochs}_lr{args.lr:g}"
    t0 = time.time()
    train_tasks = load_tasks([r for r in REGIONS if r != args.holdout])
    test_tasks = load_tasks([args.holdout])
    if args.max_eval_species:
        test_tasks = test_tasks[: args.max_eval_species]
    print(f"[PA {tag}] fine-tuning on {len(train_tasks)} species", flush=True)
    ckpt_pa = finetune_pa(train_tasks, args.epochs, args.lr, OUT / f"tabpfn35_paft_{tag}.ckpt")
    ckpt_pb = OUT / f"tabpfn35_ft_{args.holdout}_e10_lr1e-05.ckpt"
    print(f"[PA {tag}] fine-tuned in {(time.time() - t0) / 60:.1f} min -> {ckpt_pa}", flush=True)

    rows = []
    for i, t in enumerate(test_tasks):
        torch.cuda.empty_cache()
        seed = [B.SEED, REGIONS.index(args.holdout), i]
        for arm, mp in [("default", None), ("pb_finetuned", ckpt_pb), ("pa_finetuned", ckpt_pa)]:
            p = predict_fast(t, mp, seed)
            rows.append({"holdout": args.holdout, "tag": tag, "Species": t["species"], "arm": arm,
                         "n_pres": len(t["Xp"]), **B.calc_metrics(t["y_test"], p)})
        r = rows[-3:]
        print(f"  {t['species']} n_p={len(t['Xp'])}: default={r[0]['AUC']:.3f} pb_ft={r[1]['AUC']:.3f} "
              f"pa_ft={r[2]['AUC']:.3f}", flush=True)
    df = pd.DataFrame(rows)
    f = OUT / "finetune_pa_results.csv"
    df.to_csv(f, mode="a", header=not f.exists(), index=False)
    piv = df.pivot_table(index="Species", columns="arm", values="AUC")
    print(f"[PA {tag}] mean AUC default={piv['default'].mean():.4f} pb_ft={piv['pb_finetuned'].mean():.4f} "
          f"pa_ft={piv['pa_finetuned'].mean():.4f} | pa>pb {int((piv.pa_finetuned > piv.pb_finetuned).sum())}/{len(piv)}",
          flush=True)


if __name__ == "__main__":
    main()
