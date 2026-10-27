# Spatially separated evaluation as in Valavi et al. (2023, Global Ecology and Biogeography 32:369-383).
#
# For every species: training = its presence-only records + the target-group background (TGB) of
# its biological group (data/valavi2023/tgb_samples); testing = the independent presence-absence
# sites. Training and testing points are assigned to the published spatial blocks
# (data/valavi2023/blocks) and blocks to 5 folds at random, keeping the most even of 100 random
# allocations (port of Valavi's spatial_block(), seed 32). Fold f trains on training points outside
# fold f and tests on PA sites inside fold f. Random CV ("rcv") draws the same numbers of training
# presences/background and test presences/absences at random, as in Valavi's code.
# Species are skipped as in the paper (any empty fold class, < 5 training presences in a fold,
# < 10 test presences in total).
#
# Usage:
#   python spatial_cv.py tasks            # build and cache the fold tasks
#   python spatial_cv.py r [--workers 14] # reference models in R (R/valavi_models.R)
#   python spatial_cv.py gpu              # TabPFN 3.5 (default and fine-tuned), Fast
import argparse
import os
import pickle
import shutil
import subprocess
import time
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

import benchmark_v40_valavi as B
from valavi_config import CATEGORICAL, DATA_DIR, HERE, REGIONS, VARS

V23 = os.path.join(HERE, "data", "valavi2023")
OUT = os.path.join(HERE, "results", "v50")
TASKS = os.path.join(OUT, "tasks.pkl")
R_MODELS = ["GAM", "Lasso", "BRT", "RF", "RFdown", "MaxNet", "MaxEnt", "Ensemble"]
# Valavi et al. (2023) omitted vegsys in NSW (R/nceas_model_vars.R)
VARS23 = {**VARS, "NSW": [v for v in VARS["NSW"] if v != "vegsys"]}
K = 5


def assign_blocks(xy, blocks, rng):
    import geopandas as gpd
    pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(xy[:, 0], xy[:, 1]), crs=blocks.crs)
    j = gpd.sjoin(pts, blocks[["geometry"]].reset_index(names="block"), how="left", predicate="intersects")
    j = j.iloc[rng.permutation(len(j))]          # points on block edges: keep one block at random
    j = j[~j.index.duplicated()].sort_index()
    return j["block"].to_numpy()                 # NaN = outside all blocks


def spatial_folds(b_train, y_train, b_test, y_test, n_blocks, seed=32, iterations=100):
    rng = np.random.default_rng(seed)
    best, best_min, best_sd = None, -1, np.inf
    for _ in range(iterations):
        folds = np.zeros(n_blocks, int)
        num = n_blocks // K
        folds[: num * K] = rng.permutation(np.repeat(np.arange(1, K + 1), num))
        if n_blocks % K:
            folds[num * K:] = rng.choice(np.arange(1, K + 1), n_blocks % K, replace=False)
        ftr = np.where(np.isnan(b_train), 0, folds[np.nan_to_num(b_train).astype(int)])
        fte = np.where(np.isnan(b_test), 0, folds[np.nan_to_num(b_test).astype(int)])
        table = []
        for p in range(1, K + 1):
            tr, te = (ftr != p) & (ftr > 0), fte == p
            table.append([np.sum(y_train[tr] == 0), np.sum(y_train[tr] == 1),
                          np.sum(y_test[te] == 0), np.sum(y_test[te] == 1)])
        table = np.array(table)
        if table.min() >= best_min and table.std(ddof=1) < best_sd:
            best, best_min, best_sd = (ftr, fte, table), table.min(), table.std(ddof=1)
    return best


def build_tasks():
    import geopandas as gpd
    os.makedirs(OUT, exist_ok=True)
    tasks, skipped = [], []
    for r in REGIONS:
        vars_ = VARS23[r]
        po = pd.read_csv(os.path.join(DATA_DIR, "train_po", f"{r}train_po.csv"))
        blocks = gpd.read_file(os.path.join(V23, "blocks", f"{r}.gpkg"))
        for pa_path in sorted(__import__("glob").glob(os.path.join(DATA_DIR, "test_pa", f"{r}test_pa*.csv"))):
            pa = pd.read_csv(pa_path)
            env = pd.read_csv(pa_path.replace("test_pa", "test_env"))
            grp = str(pa["group"].iloc[0])
            tgb = pd.read_csv(os.path.join(V23, "tgb_samples", r, grp, "tgbsample.csv"))
            for sp in [c for c in pa.columns if c not in ("group", "siteid", "x", "y")]:
                pres = po[po.spid == sp]
                if len(pres) < 3:
                    continue
                train = pd.concat([pres, tgb], ignore_index=True)
                y_tr = np.r_[np.ones(len(pres), int), np.zeros(len(tgb), int)]
                y_te = pa[sp].to_numpy().astype(int)
                rng = np.random.default_rng(32)
                b_tr = assign_blocks(train[["x", "y"]].to_numpy(float), blocks, rng)
                b_te = assign_blocks(pa[["x", "y"]].to_numpy(float), blocks, rng)
                ftr, fte, table = spatial_folds(b_tr, y_tr, b_te, y_te, len(blocks))
                if table.min() < 1 or (table[:, 1] < 5).any() or table[:, 3].sum() < 10:
                    skipped.append((r, sp))
                    continue
                rng_r = np.random.default_rng(33)
                pr_tr, bg_tr = np.flatnonzero(y_tr == 1), np.flatnonzero(y_tr == 0)
                pr_te, ab_te = list(np.flatnonzero(y_te == 1)), list(np.flatnonzero(y_te == 0))
                for f in range(1, K + 1):
                    tr = np.flatnonzero((ftr != f) & (ftr > 0))
                    te = np.flatnonzero(fte == f)
                    n_tr1, n_tr0, n_te0, n_te1 = table[f - 1][1], table[f - 1][0], table[f - 1][2], table[f - 1][3]
                    tr_r = np.r_[rng_r.choice(pr_tr, n_tr1, replace=False), rng_r.choice(bg_tr, n_tr0, replace=False)]
                    ts_p = rng_r.choice(pr_te, n_te1, replace=False)
                    ts_a = rng_r.choice(ab_te, n_te0, replace=False)
                    pr_te = [i for i in pr_te if i not in set(ts_p)]
                    ab_te = [i for i in ab_te if i not in set(ts_a)]
                    te_r = np.r_[ts_p, ts_a]
                    for cv, a, b in [("spatial", tr, te), ("random", tr_r, te_r)]:
                        tasks.append({"region": r, "species": sp, "group": grp, "cv": cv, "fold": f,
                                      "train": train.iloc[a][vars_].reset_index(drop=True),
                                      "y_train": y_tr[a], "test": env.iloc[b][vars_].reset_index(drop=True),
                                      "y_test": y_te[b]})
        print(f"{r}: tasks so far {len(tasks)}", flush=True)
    with open(TASKS, "wb") as fh:
        pickle.dump(tasks, fh)
    n_sp = len({(t['region'], t['species']) for t in tasks})
    print(f"{n_sp} species kept, {len(skipped)} skipped, {len(tasks)} fold tasks -> {TASKS}")
    pd.DataFrame(skipped, columns=["Region", "Species"]).to_csv(os.path.join(OUT, "skipped_species.csv"), index=False)


def load_tasks():
    with open(TASKS, "rb") as fh:
        return pickle.load(fh)


def row(t, model, p, secs):
    return {"Model": model, **B.calc_metrics(t["y_test"], p), "Time": secs, "Region": t["region"],
            "Species": t["species"], "Group": t["group"], "cv": t["cv"], "fold": t["fold"],
            "n_train_pres": int(t["y_train"].sum()), "n_test": len(t["y_test"]),
            "prevalence": float(t["y_test"].mean())}


def run_r(workers):
    out = os.path.join(OUT, "results_r.csv")
    done = set()
    if os.path.exists(out):
        prev = pd.read_csv(out)
        done = set(zip(prev.Region, prev.Species, prev.cv, prev.fold))
    tasks = [t for t in load_tasks() if (t["region"], t["species"], t["cv"], t["fold"]) not in done]
    print(f"R: {len(tasks)} fold tasks, {workers} workers", flush=True)

    def one(t):
        jd = os.path.join(OUT, "rjobs", f"{t['region']}_{t['species']}_{t['cv']}_{t['fold']}")
        os.makedirs(jd, exist_ok=True)
        tr = t["train"].copy()
        tr.insert(0, "occ", t["y_train"])
        tr.to_csv(os.path.join(jd, "train.csv"), index=False)
        t["test"].to_csv(os.path.join(jd, "test.csv"), index=False)
        subprocess.run(["Rscript", os.path.join(HERE, "R", "valavi_models.R"), jd, t["region"], ",".join(R_MODELS)],
                       capture_output=True, text=True, env={**os.environ, "OMP_NUM_THREADS": "1"})
        pf, tf = os.path.join(jd, "preds.csv"), os.path.join(jd, "times.csv")
        preds = pd.read_csv(pf) if os.path.exists(pf) else pd.DataFrame()
        times = pd.read_csv(tf).set_index("model").time if os.path.exists(tf) else pd.Series(dtype=float)
        rows = [row(t, {"RFdown": "RF down-sampled"}.get(m, m), preds[m].to_numpy() if m in preds else None,
                    float(times.get(m, np.nan))) for m in R_MODELS]
        shutil.rmtree(jd, ignore_errors=True)
        return rows

    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(one, t) for t in tasks]
        for k, f in enumerate(as_completed(futs), 1):
            rows = f.result()
            pd.DataFrame(rows).to_csv(out, mode="a", header=not os.path.exists(out), index=False)
            if k % 50 == 0:
                print(f"  {k}/{len(tasks)} ({(time.time() - t0) / 60:.0f} min)", flush=True)
    print("R DONE", flush=True)


def run_gpu():
    import torch
    from scipy.special import expit, logit
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion
    out = os.path.join(OUT, "results_gpu.csv")
    done = set()
    if os.path.exists(out):
        prev = pd.read_csv(out)
        done = set(zip(prev.Region, prev.Species, prev.cv, prev.fold))
    tasks = [t for t in load_tasks() if (t["region"], t["species"], t["cv"], t["fold"]) not in done]
    print(f"GPU: {len(tasks)} fold tasks", flush=True)
    for k, t in enumerate(tasks, 1):
        P = B.Prep(t["train"], t["region"])
        X, Xt, y = P.coded(t["train"]), P.coded(t["test"]), t["y_train"]
        Xp, Xb = X[y == 1], X[y == 0]
        n_p = len(Xp)
        rows = []
        for model, ckpt, cat in [("TabPFN3.5_Fast", None, P.cat_idx),
                                 ("TabPFN3.5ft_Fast", B.FT_CKPT.format(region=t["region"]), None)]:
            t0 = time.time()
            # species seed from crc32, which is stable across processes (str hash() is salted per process)
            rng = np.random.default_rng([B.SEED, REGIONS.index(t["region"]), t["fold"], zlib.crc32(t["species"].encode()) % 10_000])
            logits = []
            for _ in range(5):
                ip = rng.choice(n_p, n_p, replace=True)
                ib = rng.choice(len(Xb), n_p, replace=n_p > len(Xb))
                kw = dict(device="cuda", random_state=int(rng.integers(1 << 31)),
                          categorical_features_indices=cat or None, ignore_pretraining_limits=True)
                if ckpt:
                    kw["model_path"] = ckpt
                tf = TabPFNClassifier.create_default_for_version(ModelVersion.V3_5, **kw).fit(
                    np.vstack([Xp[ip], Xb[ib]]), np.r_[np.ones(n_p), np.zeros(n_p)])
                logits.append(logit(np.clip(tf.predict_proba(Xt)[:, 1], 1e-6, 1 - 1e-6)))
            rows.append(row(t, model, expit(np.mean(logits, 0)), time.time() - t0))
        torch.cuda.empty_cache()
        pd.DataFrame(rows).to_csv(out, mode="a", header=not os.path.exists(out), index=False)
        if k % 50 == 0:
            print(f"  {k}/{len(tasks)}", flush=True)
    print("GPU DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["tasks", "r", "gpu"])
    ap.add_argument("--workers", type=int, default=14)
    a = ap.parse_args()
    {"tasks": build_tasks, "r": lambda: run_r(a.workers), "gpu": run_gpu}[a.step]()
