# Fine-tuning TabPFN 3.5 on simulated (virtual) species, leave-one-region-out.
#
# Landscapes = the 50,000 background points (real covariate values and coordinates) of the
# five training regions; the held-out region is not used in any way. Each epoch draws new
# virtual species. For each one:
#   - suitability: random responses to 1-4 continuous covariates (Gaussian or logistic),
#     optional random effects of a categorical covariate, random steepness and a prevalence
#     of 1-40%; true occurrence z ~ Bernoulli(p) at every landscape point;
#   - sampling bias: none, environmental (exp(beta * covariate)) or spatial (Gaussian kernels
#     around 1-5 random hubs), with random strength;
#   - presence-only records: 5-1,000 occupied points drawn with probability ~ sampling intensity;
#   - context = those presences + the same number of uniform background points (as at inference);
#     query = 1,000 uniform survey sites with the TRUE occurrence z as target.
# Settings fixed in advance: 1,000 species per epoch, 5 epochs, learning rate 1e-5.
# Evaluation: as finetune_pa.py (Fast, 5 bags, categorical covariates as numeric codes).
# Usage: python finetune_vs.py --holdout AWT
import argparse
import os
import time

import numpy as np
import pandas as pd
import torch

import benchmark_v40_valavi as B
import tabpfn.finetuning.finetuned_base as FB
from finetune_sdm import OUT, load_tasks, predict_fast
from tabpfn.constants import ModelVersion
from tabpfn.finetuning import FinetunedTabPFNClassifier
from tabpfn.model_loading import save_tabpfn_model
from valavi_config import CATEGORICAL, HERE, REGIONS, VARS

N_SPECIES = 1000
N_QUERY = 1000
BG_DIR = os.path.join(HERE, "data", "background50k")


def target_group_intensity(region, xy):
    """Survey effort surface of a TRAINING region: Gaussian kernel density of the pooled
    presence-only records of all its species (target-group idea, Phillips et al. 2009),
    evaluated at the landscape points and scaled to mean 1."""
    from sklearn.neighbors import KernelDensity
    po = pd.read_csv(os.path.join(B.DATA_DIR, "train_po", f"{region}train_po.csv"))[["x", "y"]].values.astype(float)
    bw = 0.03 * np.ptp(xy, axis=0).max()
    kd = KernelDensity(bandwidth=bw).fit(po)
    d = np.exp(kd.score_samples(xy))
    return d / d.mean()


def load_landscapes(regions, tg=False):
    out = []
    for r in regions:
        bg = pd.read_csv(os.path.join(BG_DIR, f"{r}bg50k.csv"))
        env = bg[VARS[r]]
        P = B.Prep(env, r)
        X = P.coded(env)  # z-scored continuous + categorical codes (same coding as the benchmark)
        xy = bg[["x", "y"]].values.astype(float)
        out.append({"region": r, "X": X, "n_cont": len(P.cont), "cat_idx": P.cat_idx, "xy": xy,
                    "tg": target_group_intensity(r, xy) if tg else None})
    return out


def virtual_species(L, rng):
    X, n, nc = L["X"], len(L["X"]), L["n_cont"]
    eta = np.zeros(n)
    for j in rng.choice(nc, size=rng.integers(1, min(4, nc) + 1), replace=False):
        x = X[:, j]
        if rng.random() < 0.7:  # unimodal
            opt = rng.uniform(-1.5, 1.5)
            width = np.exp(rng.uniform(np.log(0.3), np.log(2.0)))
            eta += -0.5 * ((x - opt) / width) ** 2
        else:  # monotone
            eta += rng.choice([-1, 1]) * rng.uniform(0.5, 3.0) * (x - rng.uniform(-1.5, 1.5))
    if L["cat_idx"] and rng.random() < 0.3:
        c = X[:, rng.choice(L["cat_idx"])].astype(int)
        eta += rng.normal(0, 1, c.max() + 2)[c + 1]
    eta = (eta - eta.mean()) / (eta.std() + 1e-9) * rng.uniform(1.0, 4.0)
    prev = np.exp(rng.uniform(np.log(0.01), np.log(0.4)))
    lo, hi = -20.0, 20.0
    for _ in range(40):  # intercept for the target prevalence
        a = (lo + hi) / 2
        if (1 / (1 + np.exp(-(a + eta)))).mean() > prev:
            hi = a
        else:
            lo = a
    p = 1 / (1 + np.exp(-(a + eta)))
    z = (rng.random(n) < p).astype(np.int64)
    # sampling bias
    u = rng.random()
    if L.get("tg") is not None and u < 0.4:  # real survey-effort surface of this region
        s = (L["tg"] + 1e-3) ** rng.uniform(0.5, 2.0)
    elif L.get("tg") is not None and u < 0.55:
        s = np.ones(n)
    elif L.get("tg") is None and u < 0.25:
        s = np.ones(n)
    elif u < 0.6 if L.get("tg") is None else u < 0.775:
        j = rng.integers(nc)
        s = np.exp(rng.choice([-1, 1]) * rng.uniform(0.5, 2.5) * np.clip(X[:, j], -3, 3))
    else:
        xy = L["xy"]
        ext = np.ptp(xy, axis=0).max()
        hubs = xy[rng.choice(n, rng.integers(1, 6), replace=False)]
        h = rng.uniform(0.05, 0.3) * ext
        d2 = ((xy[:, None, :] - hubs[None, :, :]) ** 2).sum(-1)
        s = np.exp(-d2 / (2 * h * h)).sum(1) + rng.uniform(0.0, 0.1)
    occ = np.flatnonzero(z == 1)
    if len(occ) < 5:
        return None
    n_p = int(min(np.exp(rng.uniform(np.log(5), np.log(1000))), len(occ)))
    w = s[occ] / s[occ].sum()
    ip = rng.choice(occ, n_p, replace=False, p=w)
    ib = rng.choice(n, n_p, replace=False)
    iq = rng.choice(n, N_QUERY, replace=False)
    if z[iq].min() == z[iq].max():
        return None
    Xc = np.vstack([X[ip], X[ib]])
    yc = np.r_[np.ones(n_p), np.zeros(n_p)]
    return Xc, yc, X[iq], z[iq]


def pb_task(t, rng):
    """Real presence-background task as in finetune_sdm.py: a balanced sample split 70/30
    (stratified, so the context stays balanced) into context and query."""
    n_p = min(len(t["Xp"]), 2000)
    ip = rng.permutation(len(t["Xp"]))[:n_p]
    ib = rng.choice(len(t["Xb"]), n_p, replace=False)
    k = max(1, int(round(0.7 * n_p)))
    if n_p - k < 1:
        return None
    Xc = np.vstack([t["Xp"][ip[:k]], t["Xb"][ib[:k]]])
    yc = np.r_[np.ones(k), np.zeros(k)]
    # a few extra background queries (0-99) so that task lengths rarely coincide
    extra = rng.choice(np.setdiff1d(np.arange(len(t["Xb"])), ib), rng.integers(0, 100), replace=False)
    Xq = np.vstack([t["Xp"][ip[k:]], t["Xb"][ib[k:]], t["Xb"][extra]])
    yq = np.r_[np.ones(n_p - k), np.zeros(n_p - k + len(extra))].astype(np.int64)
    perm = rng.permutation(len(Xq))
    return Xc, yc, Xq[perm], yq[perm]


def vs_tasks(landscapes, seed, real_tasks=None):
    """Real presence-background tasks (optional) followed by N_SPECIES virtual species.
    The split function identifies a task by its total length, so lengths must be unique:
    a colliding real task is skipped; a virtual task's query is trimmed by a few rows."""
    rng = np.random.default_rng(seed)
    Xs, ys, n_ctx_by_len = [], [], {}

    def add(sp, trim):
        Xc, yc, Xq, yq = sp
        k = len(Xq)
        while len(Xc) + k in n_ctx_by_len:
            if not trim or k <= len(Xq) - 50:
                return False
            k -= 1
        if k < 2 or yq[:k].min() == yq[:k].max():
            return False
        Xs.append(np.vstack([Xc, Xq[:k]]).astype(np.float32))
        ys.append(np.r_[yc, yq[:k]].astype(np.int64))
        n_ctx_by_len[len(Xs[-1])] = len(Xc)
        return True

    for t in real_tasks or []:
        sp = pb_task(t, rng)
        if sp is not None:
            add(sp, trim=True)
    n_real = len(Xs)
    while len(Xs) - n_real < N_SPECIES:
        sp = virtual_species(landscapes[rng.integers(len(landscapes))], rng)
        if sp is not None:
            add(sp, trim=True)
    return Xs, ys, n_ctx_by_len


def finetune_vs(landscapes, epochs, lr, out_path, seed=0, real_tasks=None):
    orig = FB.get_preprocessed_dataset_chunks

    def patched(calling_instance, X_raw, y_raw, split_fn, max_data_size, model_type, **kw):
        Xs, ys, n_ctx_by_len = vs_tasks(landscapes, kw.get("data_shuffle_seed", 0), real_tasks)

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
            n_estimators_finetune=2, n_estimators_final_inference=2, random_state=seed,
            save_checkpoint_interval=None, model_version=ModelVersion.V3_5,
            extra_classifier_kwargs={"ignore_pretraining_limits": True})
        X0 = landscapes[0]["X"][:100].astype(np.float32)
        ft.fit(X0, np.r_[np.ones(50), np.zeros(50)].astype(np.int64))
    finally:
        FB.get_preprocessed_dataset_chunks = orig
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_tabpfn_model(model=ft.finetuned_estimator_, save_path=out_path)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max-eval-species", type=int, default=0)
    ap.add_argument("--mix", action="store_true", help="add the real presence-background tasks of the training regions")
    ap.add_argument("--tg", action="store_true", help="add target-group (real survey effort) sampling bias")
    args = ap.parse_args()
    tag = f"{args.holdout}_e{args.epochs}_lr{args.lr:g}"
    kind = ("mix" if args.mix else "vs") + ("tg" if args.tg else "")
    t0 = time.time()
    train_regions = [r for r in REGIONS if r != args.holdout]
    landscapes = load_landscapes(train_regions, tg=args.tg)
    real = load_tasks(train_regions) if args.mix else None
    ckpt = finetune_vs(landscapes, args.epochs, args.lr, OUT / f"tabpfn35_{kind}ft_{tag}.ckpt", real_tasks=real)
    print(f"[{kind} {tag}] fine-tuned in {(time.time() - t0) / 60:.1f} min -> {ckpt}", flush=True)
    test = load_tasks([args.holdout])
    if args.max_eval_species:
        test = test[: args.max_eval_species]
    rows = []
    for i, t in enumerate(test):
        torch.cuda.empty_cache()
        p = predict_fast(t, ckpt, [B.SEED, REGIONS.index(args.holdout), i])
        rows.append({"holdout": args.holdout, "tag": tag, "Species": t["species"], "arm": f"{kind}_finetuned",
                     "n_pres": len(t["Xp"]), **B.calc_metrics(t["y_test"], p)})
    df = pd.DataFrame(rows)
    f = OUT / f"finetune_{kind}_results.csv"
    df.to_csv(f, mode="a", header=not f.exists(), index=False)
    prev = pd.read_csv(OUT / "finetune_pa_results.csv")
    prev = prev[prev.holdout == args.holdout]
    piv = pd.concat([prev, df]).pivot_table(index="Species", columns="arm", values="AUC")
    piv = piv.loc[df.Species]
    arm = f"{kind}_finetuned"
    print(f"[{kind} {tag}] mean AUC default={piv['default'].mean():.4f} pb_ft={piv['pb_finetuned'].mean():.4f} "
          f"{kind}_ft={piv[arm].mean():.4f} | >pb {int((piv[arm] > piv.pb_finetuned).sum())}/{len(piv)} "
          f">default {int((piv[arm] > piv['default']).sum())}/{len(piv)}", flush=True)


if __name__ == "__main__":
    main()
