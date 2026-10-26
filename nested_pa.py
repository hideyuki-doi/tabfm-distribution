# Nested leave-one-region-out selection of the number of fine-tuning epochs for
# presence-absence-supervised fine-tuning (finetune_pa.py).
#
# Phase A: for every pair of regions {R, Q}, fine-tune on the other four regions for 10 epochs,
#          saving a checkpoint after each epoch, and score epochs 0 (no fine-tuning), 1, 2, 3, 5, 7
#          and 10 on R and on Q (light evaluation: 2 bags, at most 3,000 test sites per species).
# Phase B: for outer region R, choose the epoch with the best mean inner AUC over the five
#          regions Q != R, using only runs trained without R (and without Q) -> no test data of R.
# Phase C: fine-tune on all five regions with that epoch count and evaluate on R with the same
#          protocol as finetune_pa.py (Fast, 5 bags, all test sites).
# Usage: python nested_pa.py phaseA | phaseB | phaseC
import glob
import itertools
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

import benchmark_v40_valavi as B
import finetune_pa as FPA
from finetune_sdm import OUT, load_tasks, predict_fast
from valavi_config import REGIONS

NEST = OUT / "nested_pa"
EPOCHS = [0, 1, 2, 3, 5, 7, 10]
MAX_EVAL_SITES = 3000
INNER = NEST / "inner_scores.csv"
CHOICE = NEST / "chosen_epochs.csv"
FINAL = OUT / "finetune_pa_nested_results.csv"


def light_eval(tasks, model_path, seed):
    rng = np.random.default_rng(seed)
    aucs = []
    for i, t in enumerate(tasks):
        n = len(t["y_test"])
        idx = rng.choice(n, MAX_EVAL_SITES, replace=False) if n > MAX_EVAL_SITES else np.arange(n)
        if t["y_test"][idx].min() == t["y_test"][idx].max():
            idx = np.arange(n)
        sub = {**t, "Xt": t["Xt"][idx], "y_test": t["y_test"][idx]}
        p = predict_fast(sub, model_path, [seed, i], bags=2)
        aucs.append(roc_auc_score(sub["y_test"], p))
    return float(np.mean(aucs))


def finetune_with_checkpoints(train_tasks, out_dir, epochs=10, lr=1e-5):
    """finetune_pa.finetune_pa with a checkpoint after every epoch."""
    import tabpfn.finetuning.finetuned_base as FB
    from tabpfn.constants import ModelVersion
    from tabpfn.finetuning import FinetunedTabPFNClassifier
    orig = FB.get_preprocessed_dataset_chunks

    def patched(calling_instance, X_raw, y_raw, split_fn, max_data_size, model_type, **kw):
        Xs, ys, n_ctx_by_len = FPA.pa_tasks(train_tasks, kw.get("data_shuffle_seed", 0))

        def split(X, y, stratify=None):
            n = n_ctx_by_len[len(X)]
            assert abs(float(np.mean(np.asarray(y[:n]))) - 0.5) < 1e-6, "row order changed"
            return X[:n], X[n:], y[:n], y[n:]

        kw.update(shuffle=False)
        return orig(calling_instance, Xs, ys, split, None, model_type, **kw)

    FB.get_preprocessed_dataset_chunks = patched
    try:
        ft = FinetunedTabPFNClassifier(
            device="cuda", epochs=epochs, learning_rate=lr, validation_split_ratio=None, early_stopping=False,
            n_estimators_finetune=2, n_estimators_final_inference=2, random_state=0,
            save_checkpoint_interval=1, model_version=ModelVersion.V3_5,
            extra_classifier_kwargs={"ignore_pretraining_limits": True})
        t = train_tasks[0]
        X0 = np.vstack([t["Xp"][:50], t["Xb"][:50]]).astype(np.float32)
        ft.fit(X0, np.r_[np.ones(len(t["Xp"][:50])), np.zeros(50)].astype(np.int64), output_dir=Path(out_dir))
    finally:
        FB.get_preprocessed_dataset_chunks = orig
    ck = {}
    for f in glob.glob(os.path.join(out_dir, "checkpoint_*_*.pth")):
        e = os.path.basename(f).rsplit("_", 1)[1].split(".")[0]
        if e.isdigit():
            ck[int(e)] = f
    return ck


def phase_a():
    NEST.mkdir(parents=True, exist_ok=True)
    tasks = {r: load_tasks([r]) for r in REGIONS}
    done = set()
    if INNER.exists():
        prev = pd.read_csv(INNER)
        done = set(zip(prev.pair, prev.eval_region, prev.epoch))
    # epoch 0 = default weights, independent of the pair
    for r in REGIONS:
        if ("none", r, 0) not in done:
            a = light_eval(tasks[r], None, B.SEED)
            pd.DataFrame([{"pair": "none", "eval_region": r, "epoch": 0, "auc": a}]).to_csv(
                INNER, mode="a", header=not INNER.exists(), index=False)
            print(f"default {r}: {a:.4f}", flush=True)
    for R, Q in itertools.combinations(REGIONS, 2):
        pair = f"{R}-{Q}"
        if all((pair, x, e) in done for x in (R, Q) for e in EPOCHS if e > 0):
            continue
        t0 = time.time()
        train = [t for g in REGIONS if g not in (R, Q) for t in tasks[g]]
        out_dir = NEST / f"run_{pair}"
        ck = finetune_with_checkpoints(train, str(out_dir))
        rows = []
        for e in [e for e in EPOCHS if e > 0]:
            for x in (R, Q):
                torch.cuda.empty_cache()
                rows.append({"pair": pair, "eval_region": x, "epoch": e, "auc": light_eval(tasks[x], ck[e], B.SEED)})
        pd.DataFrame(rows).to_csv(INNER, mode="a", header=not INNER.exists(), index=False)
        shutil.rmtree(out_dir, ignore_errors=True)
        print(f"[{pair}] {(time.time() - t0) / 60:.0f} min | " +
              " ".join(f"{r['eval_region']}@{r['epoch']}={r['auc']:.3f}" for r in rows), flush=True)


def phase_b():
    s = pd.read_csv(INNER)
    out = []
    for R in REGIONS:
        rows = []
        for Q in [q for q in REGIONS if q != R]:
            pair = "-".join(sorted([R, Q], key=REGIONS.index))
            for e in EPOCHS:
                sel = s[(s.eval_region == Q) & (s.epoch == e) & ((s.pair == pair) if e > 0 else (s.pair == "none"))]
                rows.append({"Q": Q, "epoch": e, "auc": sel.auc.mean()})
        t = pd.DataFrame(rows).groupby("epoch").auc.mean()
        best = int(t.idxmax())
        out.append({"outer": R, "best_epoch": best, **{f"inner_auc_e{e}": round(v, 4) for e, v in t.items()}})
        print(f"{R}: best epoch {best} | " + " ".join(f"e{e}={v:.4f}" for e, v in t.items()))
    pd.DataFrame(out).to_csv(CHOICE, index=False)


def phase_c():
    ch = pd.read_csv(CHOICE).set_index("outer").best_epoch
    prev = pd.read_csv(OUT / "finetune_pa_results.csv")  # default and pb_finetuned arms (same seeds)
    rows = []
    for R in REGIONS:
        e = int(ch[R])
        test = load_tasks([R])
        if e == 0:
            ckpt = None
        else:
            train = load_tasks([r for r in REGIONS if r != R])
            ckpt = FPA.finetune_pa(train, e, 1e-5, OUT / f"tabpfn35_paft_{R}_e{e}_lr1e-05_nested.ckpt")
        for i, t in enumerate(test):
            torch.cuda.empty_cache()
            p = predict_fast(t, ckpt, [B.SEED, REGIONS.index(R), i])
            rows.append({"holdout": R, "Species": t["species"], "arm": f"pa_nested_e{e}", "n_pres": len(t["Xp"]),
                         **B.calc_metrics(t["y_test"], p)})
        print(f"{R}: epoch {e} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(FINAL, index=False)
    df["arm"] = "pa_nested"
    allr = pd.concat([prev, df]).pivot_table(index=["holdout", "Species"], columns="arm", values="AUC")
    print(allr.groupby(level=0).mean().round(3))
    print(allr.mean().round(4))


if __name__ == "__main__":
    {"phaseA": phase_a, "phaseB": phase_b, "phaseC": phase_c}[sys.argv[1]]()
