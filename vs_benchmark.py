# Virtual-species benchmark with known truth (results/v60).
#
# Why: on real data the true distribution is unknown, so we cannot tell whether a model recovers
# the niche or only ranks survey sites well. Here species are simulated on the real environmental
# landscapes of the six NCEAS regions (the 50,000 background points of each region), so the true
# probability of occurrence and the true response to each covariate are known.
#
# Design (fixed before any model was run):
#   - 10 virtual species per region (60 species). Each responds additively on the logit scale to
#     2-3 continuous covariates; each response is unimodal (Gaussian, p = 0.7) or monotone
#     (logistic). Categorical and the remaining covariates are irrelevant but given to all models.
#     Prevalence (mean true probability over the landscape) is log-uniform in 0.03-0.30.
#   - Presence records are drawn from occupied points, with uniform sampling (half of the species)
#     or environmental sampling bias exp(b * z) on a covariate not in the niche, b in [1, 2]
#     (the other half). Nested samples of n = 10, 30, 100 and 300 presences (240 datasets).
#   - Landscape split: 5,000 test points, 20,000 background points (disjoint), both random.
#     At test points we record the true probability and one Bernoulli draw (true presence/absence).
#   - Partial-dependence rows: 100 random test points x 12 quantiles of each niche covariate,
#     appended to the test set, so every model is also asked to predict response curves.
# Models: the R reference models of Valavi et al. (R/valavi_models.R: GAM, Lasso, BRT, RF,
#   RF down-sampled, MaxNet, MaxEnt, Ensemble) and TabPFN 3.5 with default and leave-one-region-out
#   fine-tuned weights (the checkpoint of the region the species lives in, which never saw that
#   region), 20 balanced bags (the first 5 give the "Fast" setting).
# Metrics: AUC against the Bernoulli draws, Spearman correlation with the true probability,
#   Pearson correlation of logits with the true logit, and response-curve recovery (Pearson r
#   between each model's partial dependence and the true additive component, averaged over the
#   niche covariates).
#
# Usage: python vs_benchmark.py simulate | r [--workers 12] | gpu | analyse
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

from valavi_config import CATEGORICAL, HERE, REGIONS, VARS

# VS_FAMILY=additive (results/v60, main design) or complex (results/v61: non-additive niches with
# interactions and thresholds, added in revision to test whether the additive design favours regression)
FAMILY = os.environ.get("VS_FAMILY", "additive")
OUT = os.path.join(HERE, "results", "v60" if FAMILY == "additive" else "v61")
DATA = os.path.join(OUT, "data")
BG_DIR = os.path.join(HERE, "data", "background50k")
N_SPECIES, SIZES, N_TEST, N_BG = 10, [10, 30, 100, 300], 5000, 20000
N_PD_POINTS, N_GRID = 100, 12
SEED = 60


# ------------------------------------------------------------------ simulation
def simulate():
    os.makedirs(DATA, exist_ok=True)
    meta = []
    for ri, r in enumerate(REGIONS):
        bg = pd.read_csv(os.path.join(BG_DIR, f"{r}bg50k.csv"))
        cov = bg[VARS[r]].reset_index(drop=True)
        cont = [v for v in VARS[r] if v not in CATEGORICAL.get(r, [])]
        z = (cov[cont] - cov[cont].mean()) / cov[cont].std()
        n = len(cov)
        k = 0
        rng = np.random.default_rng([SEED, ri])
        while k < N_SPECIES:
            if FAMILY == "complex":
                niche, spec, fun = complex_niche(rng, cont)
                eta = fun(z)
                scale = rng.uniform(1.5, 3.0) / eta.std()
                eta = eta * scale
                comps = None
            else:
                niche = list(rng.choice(cont, size=rng.integers(2, 4), replace=False))
                comps, spec = {}, []
            if FAMILY != "complex":
                for v in niche:
                    x = z[v].values
                    if rng.random() < 0.7:
                        opt, w = rng.uniform(-1.5, 1.5), np.exp(rng.uniform(np.log(0.4), np.log(1.5)))
                        comps[v] = -0.5 * ((x - opt) / w) ** 2
                        spec.append({"var": v, "type": "gaussian", "opt": opt, "width": w})
                    else:
                        sl, c0 = rng.choice([-1, 1]) * rng.uniform(1, 3), rng.uniform(-1, 1)
                        comps[v] = sl * (x - c0)
                        spec.append({"var": v, "type": "monotone", "slope": sl, "centre": c0})
                eta = sum(comps.values())
                scale = rng.uniform(1.5, 3.0) / eta.std()
                eta = eta * scale
                comps = {v: c * scale for v, c in comps.items()}
            prev = np.exp(rng.uniform(np.log(0.03), np.log(0.30)))
            lo, hi = -30.0, 30.0
            for _ in range(60):
                a = (lo + hi) / 2
                lo, hi = (lo, a) if expit(a + eta).mean() > prev else (a, hi)
            p = expit(a + eta)
            occ = rng.random(n) < p
            perm = rng.permutation(n)
            te, bgi, rest = perm[:N_TEST], perm[N_TEST:N_TEST + N_BG], perm[N_TEST + N_BG:]
            # presences come from the part of the landscape not used for testing
            pool = np.r_[bgi, rest]
            occ_pool = pool[occ[pool]]
            if len(occ_pool) < max(SIZES) or occ[te].sum() < 20:
                continue
            biased = k % 2 == 1
            if biased:
                others = [v for v in cont if v not in niche]
                bv = rng.choice(others) if others else niche[0]
                b = rng.choice([-1, 1]) * rng.uniform(1, 2)
                w_s = np.exp(b * z[bv].values[occ_pool])
            else:
                bv, b, w_s = None, 0.0, np.ones(len(occ_pool))
            pres = rng.choice(occ_pool, size=max(SIZES), replace=False, p=w_s / w_s.sum())  # nested: first n
            # partial-dependence rows
            base = rng.choice(te, N_PD_POINTS, replace=False)
            pd_rows, pd_meta = [], []
            for v in niche:
                grid = np.quantile(cov[v].values, np.linspace(0.02, 0.98, N_GRID))
                for gi, gv in enumerate(grid):
                    blk = cov.iloc[base].copy()
                    blk[v] = gv
                    pd_rows.append(blk)
                    pd_meta += [(v, gi, gv)] * N_PD_POINTS
            pdx = pd.concat(pd_rows, ignore_index=True)
            truth_pd = {}
            if FAMILY == "complex":
                # partial dependence of the TRUE logit, computed on the same rows as the models'
                zpd = (pdx[cont] - cov[cont].mean()) / cov[cont].std()
                eta_pd = fun(zpd) * scale
                pv = np.array([m[0] for m in pd_meta]); pg = np.array([m[1] for m in pd_meta])
                for v in niche:
                    sel = pv == v
                    truth_pd[v] = pd.Series(eta_pd[sel]).groupby(pg[sel]).mean().sort_index().tolist()
            else:
                for v in niche:
                    grid = np.quantile(cov[v].values, np.linspace(0.02, 0.98, N_GRID))
                    zg = (grid - cov[v].mean()) / cov[v].std()
                    s = next(x for x in spec if x["var"] == v)
                    f = (-0.5 * ((zg - s["opt"]) / s["width"]) ** 2 if s["type"] == "gaussian"
                         else s["slope"] * (zg - s["centre"])) * scale
                    truth_pd[v] = f.tolist()
            sid = f"{r}_vs{k:02d}"
            np.savez_compressed(os.path.join(DATA, f"{sid}.npz"), pres=pres, bg=bgi, te=te,
                                p_true=p[te], y_true=occ[te].astype(np.int8),
                                pd_var=np.array([m[0] for m in pd_meta]), pd_grid=np.array([m[1] for m in pd_meta]))
            pdx.to_csv(os.path.join(DATA, f"{sid}_pd.csv"), index=False)
            meta.append({"sid": sid, "region": r, "niche": niche, "spec": spec, "scale": scale, "prevalence": prev,
                         "biased": biased, "bias_var": bv, "bias_b": b, "truth_pd": truth_pd})
            print(f"{sid}: niche={niche} prev={prev:.3f} biased={biased} test_prev={occ[te].mean():.3f}", flush=True)
            k += 1
    json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"), indent=1, default=float)


def complex_niche(rng, cont):
    """Non-additive niche on two covariates (plus, with probability 0.5, an additive Gaussian third):
    a tilted bivariate Gaussian (correlated tolerances), a joint threshold (both conditions needed)
    or a multiplicative interaction (saddle). Returns (niche, spec, f) with f(z frame) -> eta."""
    kind = ["bivariate", "threshold", "product"][rng.integers(3)]
    v1, v2 = [str(v) for v in rng.choice(cont, size=2, replace=False)]
    if kind == "bivariate":
        c1, c2 = rng.uniform(-1.2, 1.2, 2)
        s1, s2 = np.exp(rng.uniform(np.log(0.5), np.log(1.5), 2))
        rho = rng.choice([-1, 1]) * rng.uniform(0.5, 0.85)
        par = {"c1": c1, "c2": c2, "s1": s1, "s2": s2, "rho": rho}

        def f2(z):
            a, b = (z[v1].values - c1) / s1, (z[v2].values - c2) / s2
            return -0.5 * (a * a - 2 * rho * a * b + b * b) / (1 - rho * rho)
    elif kind == "threshold":
        t1, t2 = rng.uniform(-1, 1, 2)
        d1, d2 = rng.choice([-1, 1], 2)
        par = {"t1": t1, "t2": t2, "d1": int(d1), "d2": int(d2)}

        def f2(z):
            return 4 * expit(6 * d1 * (z[v1].values - t1)) * expit(6 * d2 * (z[v2].values - t2))
    else:
        beta = rng.choice([-1, 1]) * rng.uniform(1, 2)
        gam = rng.uniform(-1, 1)
        par = {"beta": beta, "gamma": gam}

        def f2(z):
            return beta * z[v1].values * z[v2].values + gam * z[v1].values
    niche, spec = [v1, v2], [{"type": kind, "vars": [v1, v2], **{k: float(x) for k, x in par.items()}}]
    others = [v for v in cont if v not in niche]
    third = None
    if others and rng.random() < 0.5:
        v3 = str(rng.choice(others))
        o3, w3 = rng.uniform(-1.5, 1.5), np.exp(rng.uniform(np.log(0.4), np.log(1.5)))
        third = (v3, o3, w3)
        niche.append(v3)
        spec.append({"type": "gaussian", "var": v3, "opt": float(o3), "width": float(w3)})

    def f(z):
        e = f2(z)
        if third is not None:
            e = e - 0.5 * ((z[third[0]].values - third[1]) / third[2]) ** 2
        return e
    return niche, spec, f


def load(sid, n):
    r = sid.split("_")[0]
    d = np.load(os.path.join(DATA, f"{sid}.npz"))
    bg = pd.read_csv(os.path.join(BG_DIR, f"{r}bg50k.csv"))[VARS[r]]
    pdx = pd.read_csv(os.path.join(DATA, f"{sid}_pd.csv"))
    return {"region": r, "pres": bg.iloc[d["pres"][:n]].reset_index(drop=True),
            "bg": bg.iloc[d["bg"]].reset_index(drop=True),
            "test": pd.concat([bg.iloc[d["te"]], pdx], ignore_index=True),
            "n_test": len(d["te"]), "p_true": d["p_true"], "y_true": d["y_true"],
            "pd_var": d["pd_var"], "pd_grid": d["pd_grid"]}


def jobs():
    meta = json.load(open(os.path.join(OUT, "meta.json")))
    return [(m["sid"], n) for m in meta for n in SIZES]


# ------------------------------------------------------------------ R models
R_MODELS = ["GAM", "Lasso", "BRT", "RF", "RFdown", "MaxNet", "MaxEnt", "Ensemble"]


def run_r(job):
    sid, n = job
    out = os.path.join(OUT, "preds", f"{sid}_n{n}_r.npz")
    if os.path.exists(out):
        return job, 0.0, "cached"
    d = load(sid, n)
    jd = os.path.join(OUT, "rjobs", f"{sid}_n{n}")
    os.makedirs(jd, exist_ok=True)
    train = pd.concat([d["pres"], d["bg"]], ignore_index=True)
    train.insert(0, "occ", np.r_[np.ones(len(d["pres"]), int), np.zeros(len(d["bg"]), int)])
    train.to_csv(os.path.join(jd, "train.csv"), index=False)
    d["test"].to_csv(os.path.join(jd, "test.csv"), index=False)
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MODEL_TIME_LIMIT": "3600"}
    t0 = time.time()
    r = subprocess.run(["Rscript", os.path.join(HERE, "R", "valavi_models.R"), jd, d["region"], ",".join(R_MODELS)],
                       capture_output=True, text=True, env=env)
    pf = os.path.join(jd, "preds.csv")
    if not os.path.exists(pf):
        return job, time.time() - t0, "FAILED " + (r.stderr.strip().splitlines() or ["?"])[-1][:200]
    preds = pd.read_csv(pf)
    np.savez_compressed(out, **{c: preds[c].values.astype(np.float32) for c in preds.columns})
    shutil.rmtree(jd, ignore_errors=True)
    return job, time.time() - t0, "ok " + ",".join(preds.columns)


def stage_r(workers):
    os.makedirs(os.path.join(OUT, "preds"), exist_ok=True)
    js = jobs()
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(run_r, j) for j in js]
        for i, f in enumerate(as_completed(futs)):
            job, dt, msg = f.result()
            print(f"[R {i + 1}/{len(js)}] {job} {dt / 60:.1f} min {msg}", flush=True)
    print("ALLDONE", flush=True)


# ------------------------------------------------------------------ TabPFN
BAGS = 20


BG_RATIO = 1


def stage_gpu(reverse=False):
    import torch
    import benchmark_v40_valavi as B
    os.makedirs(os.path.join(OUT, "preds"), exist_ok=True)
    js = jobs()[::-1] if reverse == "reverse" else jobs()
    if reverse == "shuffle":
        js = [js[i] for i in np.random.default_rng(7).permutation(len(js))]
    if BG_RATIO != 1:  # sensitivity analysis only where background per bag is scarce
        js = [j for j in js if j[1] in (10, 30)]
    for i, (sid, n) in enumerate(js):
        out = os.path.join(OUT, "preds", f"{sid}_n{n}_gpu.npz" if BG_RATIO == 1 else f"{sid}_n{n}_gpu_bg{BG_RATIO}.npz")
        if os.path.exists(out) or os.path.exists(out + ".lock"):
            continue
        open(out + ".lock", "w").close()
        t0 = time.time()
        d = load(sid, n)
        r = d["region"]
        train = pd.concat([d["pres"], d["bg"]], ignore_index=True)
        P = B.Prep(train, r)
        Xp, Xb, Xt = P.coded(d["pres"]), P.coded(d["bg"]), P.coded(d["test"])
        res = {}
        arms = (("TabPFN3.5", None, P.cat_idx), ("TabPFN3.5ft", B.FT_CKPT.format(region=r), None))
        if BG_RATIO != 1:  # sensitivity analysis: more background points per bag, fine-tuned model only
            arms = ((f"TabPFN3.5ft_bg{BG_RATIO}", B.FT_CKPT.format(region=r), None),)
        for name, ckpt, cats in arms:
            rng = np.random.default_rng([SEED, REGIONS.index(r), int(sid[-2:]), n])
            logits = []
            for _ in range(BAGS):
                ip = rng.choice(n, n, replace=True)
                ib = rng.choice(len(Xb), n * BG_RATIO, replace=True)
                X = np.vstack([Xp[ip], Xb[ib]])
                y = np.r_[np.ones(n), np.zeros(n * BG_RATIO)]
                tf = B.make_tabpfn("v3.5", int(rng.integers(1 << 31)), cats, ckpt).fit(X, y)
                logits.append(logit(np.clip(tf.predict_proba(Xt)[:, 1], 1e-6, 1 - 1e-6)))
                torch.cuda.empty_cache()
            L = np.array(logits)
            if BG_RATIO == 1:
                res[name + "_Fast"] = expit(L[:5].mean(0)).astype(np.float32)
            res[name] = expit(L.mean(0)).astype(np.float32)
        np.savez_compressed(out, **res)
        os.remove(out + ".lock")
        print(f"[GPU {i + 1}/{len(js)}] {sid} n={n} {time.time() - t0:.0f} s", flush=True)
    print("ALLDONE", flush=True)


# ------------------------------------------------------------------ analysis
def stage_analyse():
    meta = {m["sid"]: m for m in json.load(open(os.path.join(OUT, "meta.json")))}
    rows = []
    for sid, n in jobs():
        d = np.load(os.path.join(DATA, f"{sid}.npz"))
        preds = {}
        for f in glob.glob(os.path.join(OUT, "preds", f"{sid}_n{n}_*.npz")):
            z = np.load(f)
            preds.update({k: z[k] for k in z.files})
        nt = len(d["te"])
        pt, yt = d["p_true"], d["y_true"]
        m = meta[sid]
        rows.append({"sid": sid, "region": m["region"], "n": n, "biased": m["biased"], "Model": "Truth",
                     "AUC": roc_auc_score(yt, pt), "spearman": 1.0, "pearson_logit": 1.0, "curve_r": 1.0})
        for model, p in preds.items():
            p = np.clip(p.astype(float), 1e-6, 1 - 1e-6)
            pe, pp = p[:nt], p[nt:]
            const = np.ptp(pe) == 0
            row = {"sid": sid, "region": m["region"], "n": n, "biased": m["biased"], "Model": model,
                   "AUC": 0.5 if const else roc_auc_score(yt, pe),
                   "spearman": 0.0 if const else spearmanr(pe, pt)[0],
                   "pearson_logit": 0.0 if const else pearsonr(logit(pe), logit(pt))[0]}
            # response curves: mean logit over PD points at each grid value
            rs = []
            for v in m["niche"]:
                sel = d["pd_var"] == v
                lg = pd.Series(logit(pp[sel])).groupby(d["pd_grid"][sel]).mean().sort_index().values
                tr = np.array(m["truth_pd"][v])
                rs.append(0.0 if np.ptp(lg) == 0 else pearsonr(lg, tr)[0])
            row["curve_r"] = float(np.mean(rs))
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "results.csv"), index=False)
    print(df.groupby(["Model", "n"])[["AUC", "spearman", "curve_r"]].mean().unstack("n").round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--order", default="forward")
    ap.add_argument("--bg-ratio", type=int, default=1)
    a = ap.parse_args()
    BG_RATIO = a.bg_ratio
    {"simulate": simulate, "r": lambda: stage_r(a.workers), "gpu": lambda: stage_gpu(a.order), "analyse": stage_analyse}[a.stage]()
