# ==============================================================================
# Version: Benchmark v40.0 "Valavi-compatible"
# Target: NVIDIA GB10 (DGX Spark)
#
# Protocol follows Valavi et al. (2022) Ecol. Monogr. 92:e01486 so that results are
# directly comparable with their 21 models:
#   - 225 species (the one species with 2 presence records is excluded)
#   - 50,000 random background points per region (make_background.py)
#   - uncorrelated covariates per region, categorical covariates as factors (valavi_config.py)
#   - environmental covariates only (no coordinates); continuous covariates z-scored
#     with training (presence + background) mean / sd
#   - evaluation on the independent presence-absence data: AUC_ROC, AUC_PRG, COR (+ TSS, AP)
#
# Reference models re-implemented with Valavi's settings (Table 2 / Appendix S2):
#   MaxEnt (maxent.jar defaults, no threshold features -> elapid), RF, RF down-sampled,
#   BRT (gbm.step settings -> LightGBM), XGBoost (eta 0.001, nrounds by 5-fold CV),
#   Lasso (orthogonal quadratic terms, down-weighting, CV lambda), Ensemble
#   (mean of 0-1 rescaled Lasso, MaxEnt, BRT, RF down-sampled; Valavi also had GAM).
# Foundation models: TabPFN 3.5 / 2.5 (balanced bagging as in v35), TabICL.
#
# Per-species test predictions are saved (results/v40/preds) for ensembles and re-analysis.
# ==============================================================================

import argparse
import datetime
import glob
import os
import random
import time
import traceback
import warnings

import numpy as np
import pandas as pd
import torch
from scipy.special import expit, logit
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

import valavi_stats as VS
from valavi_config import CATEGORICAL, DATA_DIR, HERE, REGIONS, VARS

warnings.filterwarnings("ignore")

BG_DIR = os.path.join(HERE, "data", "background50k")
OUT_DIR = os.path.join(HERE, "results", "v40")
PRED_DIR = os.path.join(OUT_DIR, "preds")
SEED = 42
N_JOBS = int(os.environ.get("N_JOBS", "16"))
TABPFN_BAGS = 100       # "Stable" (as v35)
TABPFN_FAST_BAGS = 5    # "Fast"   (as v35)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RUN_TAG = "run"  # set in main(); prediction files are per run so CPU/GPU runs never collide

CPU_MODELS = ["MaxEnt", "RF", "RF down-sampled", "BRT", "Lasso", "Ensemble", "XGBoost"]
GPU_MODELS = ["TabPFN3.5_Fast", "TabPFN2.5_Fast", "TabICL", "TabPFN3.5_Stable", "TabPFN2.5_Stable",
              "TabPFN3.5ft_Fast", "TabPFN3.5ft_Stable"]
# leave-one-region-out fine-tuned TabPFN 3.5 weights (finetune_sdm.py): species of region R are
# predicted with the checkpoint trained on the other five regions only
FT_CKPT = os.path.join(HERE, "results", "v40", "finetune", "tabpfn35_ft_{region}_e10_lr1e-05.ckpt")
ALL_MODELS = CPU_MODELS + GPU_MODELS
ENSEMBLE_PARTS = ["Lasso", "MaxEnt", "BRT", "RF down-sampled"]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------
def calc_metrics(y, p):
    if p is None or np.isnan(p).any():
        return {"AUC": np.nan, "PRG": np.nan, "COR": np.nan, "TSS": np.nan, "AP": np.nan}
    if np.ptp(p) == 0:  # constant prediction (e.g. intercept-only Lasso): no discrimination
        return {"AUC": 0.5, "PRG": VS.auc_prg(y, p), "COR": 0.0, "TSS": 0.0, "AP": float(np.mean(y))}
    fpr, tpr, _ = roc_curve(y, p)
    return {
        "AUC": roc_auc_score(y, p),
        "PRG": VS.auc_prg(y, p),
        "COR": pearsonr(y, p)[0],
        "TSS": float(np.max(tpr - fpr)),
        "AP": average_precision_score(y, p),
    }


# ---------------------------------------------------------
# Data
# ---------------------------------------------------------
def load_region(region):
    vars_ = VARS[region]
    po = pd.read_csv(os.path.join(DATA_DIR, "train_po", f"{region}train_po.csv"))
    bg = pd.read_csv(os.path.join(BG_DIR, f"{region}bg50k.csv"))[vars_]
    out = []
    for pa_path in sorted(glob.glob(os.path.join(DATA_DIR, "test_pa", f"{region}test_pa*.csv"))):
        pa = pd.read_csv(pa_path)
        env = pd.read_csv(pa_path.replace("test_pa", "test_env"))
        assert (pa["siteid"].values == env["siteid"].values).all(), pa_path
        group = str(pa["group"].iloc[0])
        for sp in [c for c in pa.columns if c not in ("group", "siteid", "x", "y")]:
            pres = po.loc[po["spid"] == sp, vars_]
            if len(pres) < 3:  # Valavi et al.: 225 species (one with 2 records dropped)
                continue
            y = pa[sp].values
            out.append({"region": region, "species": sp, "group": group, "pres": pres.reset_index(drop=True),
                        "bg": bg, "test": env[vars_].reset_index(drop=True), "y_test": y.astype(int)})
    return out


class Prep:
    """Covariate handling fitted on training data (presences + background)."""

    def __init__(self, train: pd.DataFrame, region):
        self.cat = [v for v in CATEGORICAL.get(region, []) if v in train.columns]
        self.cont = [v for v in train.columns if v not in self.cat]
        self.mean = train[self.cont].mean()
        self.sd = train[self.cont].std().replace(0, 1)
        self.levels = {c: np.sort(train[c].dropna().unique()) for c in self.cat}
        self.mode = {c: train[c].mode().iloc[0] for c in self.cat}

    def _fill(self, df):
        df = df.copy()
        df[self.cont] = df[self.cont].fillna(self.mean)
        for c in self.cat:
            df[c] = df[c].fillna(self.mode[c])
        return df

    def raw(self, df):
        """Unscaled continuous + integer-coded categoricals (MaxEnt scales internally)."""
        df = self._fill(df)
        return np.column_stack([df[self.cont].values] + [self._codes(df, c) for c in self.cat]).astype(float)

    def coded(self, df):
        """z-scored continuous + integer-coded categoricals (TabPFN / TabICL)."""
        df = self._fill(df)
        z = ((df[self.cont] - self.mean) / self.sd).values
        return np.column_stack([z] + [self._codes(df, c) for c in self.cat]).astype(float)

    def onehot(self, df):
        """z-scored continuous + one-hot categoricals (R model.matrix style)."""
        df = self._fill(df)
        z = ((df[self.cont] - self.mean) / self.sd).values
        oh = [(df[c].values[:, None] == self.levels[c][None, :]).astype(float) for c in self.cat]
        return np.column_stack([z] + oh)

    def _codes(self, df, c):
        codes = np.searchsorted(self.levels[c], df[c].values)
        codes = np.clip(codes, 0, len(self.levels[c]) - 1)
        known = self.levels[c][codes] == df[c].values
        return np.where(known, codes, -1)  # unseen level -> -1

    @property
    def cat_idx(self):
        return list(range(len(self.cont), len(self.cont) + len(self.cat)))


def orth_poly2(x_train, *xs):
    """R's poly(x, 2) (orthogonal quadratic basis) fitted on x_train, applied to xs."""
    xbar = x_train.mean()
    X = np.column_stack([np.ones_like(x_train), x_train - xbar, (x_train - xbar) ** 2])
    Q, R = np.linalg.qr(X)
    Z = Q * np.diag(R)
    norm2 = (Z ** 2).sum(axis=0)
    alpha = ((x_train - xbar)[:, None] * Z ** 2).sum(axis=0) / norm2 + xbar
    norm2 = np.r_[1.0, norm2]

    def apply(x):
        out = np.zeros((len(x), 3))
        out[:, 0] = 1
        out[:, 1] = x - alpha[0]
        out[:, 2] = (x - alpha[1]) * out[:, 1] - (norm2[2] / norm2[1]) * out[:, 0]
        return out[:, 1:] / np.sqrt(norm2[2:])

    return [apply(x) for x in (x_train,) + xs]


# ---------------------------------------------------------
# Models
# ---------------------------------------------------------
def maxent_features(n_p):
    # maxent.jar "auto" features, without threshold (Valavi et al. Table 2)
    if n_p < 10:
        return ["linear"]
    if n_p < 15:
        return ["linear", "quadratic"]
    if n_p < 80:
        return ["linear", "quadratic", "hinge"]
    return ["linear", "quadratic", "hinge", "product"]


def make_tabpfn(version, seed, cat_idx, model_path=None):
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion
    kw = {"model_path": model_path} if model_path else {}
    return TabPFNClassifier.create_default_for_version(
        ModelVersion(version), device=DEVICE, random_state=seed,
        categorical_features_indices=cat_idx or None, ignore_pretraining_limits=True, **kw)


def run_models(d, models, rng, region):
    pres, bg, test, y_test = d["pres"], d["bg"], d["test"], d["y_test"]
    train = pd.concat([pres, bg], ignore_index=True)
    y = np.r_[np.ones(len(pres)), np.zeros(len(bg))]
    n_p, n_b = len(pres), len(bg)
    wt = np.where(y == 1, 1.0, n_p / n_b)  # down-weighting (Valavi: DW)
    P = Prep(train, region)

    def maxent():
        import elapid
        m = elapid.MaxentModel(feature_types=maxent_features(n_p), beta_multiplier=1.0,
                               transform="cloglog", clamp=True, n_cpus=N_JOBS, random_state=SEED)
        m.fit(P.raw(train), y, categorical=P.cat_idx or None)
        return m.predict(P.raw(test))

    def rf():
        m = RandomForestClassifier(n_estimators=500, max_features="sqrt", n_jobs=N_JOBS, random_state=SEED)
        return m.fit(P.onehot(train), y).predict_proba(P.onehot(test))[:, 1]

    def rf_down():
        from imblearn.ensemble import BalancedRandomForestClassifier
        # each tree: n_p presences + n_p background, drawn with replacement (sampsize = c(n_p, n_p))
        m = BalancedRandomForestClassifier(n_estimators=1000, max_features="sqrt", sampling_strategy="all",
                                           replacement=True, bootstrap=False, n_jobs=N_JOBS, random_state=SEED)
        return m.fit(P.onehot(train), y).predict_proba(P.onehot(test))[:, 1]

    def brt():
        import lightgbm as lgb
        # gbm.step: bernoulli, tc = 1 if n_p < 50 else 5, lr 0.001, bag.fraction 0.75,
        # n.minobsinnode 10, down-weights, n.trees chosen by 5-fold CV (max 10,000)
        tc = 1 if n_p < 50 else 5
        params = dict(objective="binary", learning_rate=0.001, num_leaves=tc + 1, max_depth=-1,
                      min_data_in_leaf=10, min_sum_hessian_in_leaf=0.0, bagging_fraction=0.75, bagging_freq=1,
                      feature_fraction=1.0, lambda_l2=0.0, verbose=-1, num_threads=N_JOBS, seed=SEED,
                      deterministic=True)
        Xtr = P.onehot(train)
        ds = lgb.Dataset(Xtr, y, weight=wt, free_raw_data=False)
        cv = lgb.cv(params, ds, num_boost_round=10000, nfold=5, stratified=True, seed=SEED,
                    callbacks=[lgb.early_stopping(500, verbose=False)])
        best = max(50, len(cv["valid binary_logloss-mean"]))
        m = lgb.train(params, ds, num_boost_round=best)
        return m.predict(P.onehot(test))

    def xgboost():
        import xgboost as xgb
        # caret grid: nrounds 500..15,000 by 500; eta 0.001, max_depth 5, subsample 0.75,
        # colsample_bytree 0.8, min_child_weight 1, gamma 0; 5-fold CV on ROC; no weights
        params = dict(objective="binary:logistic", eta=0.001, max_depth=5, subsample=0.75,
                      colsample_bytree=0.8, min_child_weight=1, gamma=0, eval_metric="auc",
                      tree_method="hist", nthread=N_JOBS, seed=SEED)
        dtr = xgb.DMatrix(P.onehot(train), label=y)
        cv = xgb.cv(params, dtr, num_boost_round=15000, nfold=5, stratified=True, seed=SEED,
                    early_stopping_rounds=1500, verbose_eval=False)
        auc = cv["test-auc-mean"].values
        grid = np.arange(500, 15001, 500)
        grid = grid[grid <= len(auc)] if len(auc) >= 500 else np.array([len(auc)])
        best = int(grid[np.argmax(auc[grid - 1])])
        m = xgb.train(params, dtr, num_boost_round=best)
        return m.predict(xgb.DMatrix(P.onehot(test)))

    def lasso():
        # glmnet(alpha = 1) on orthogonal quadratic terms of continuous covariates + factors,
        # down-weighted, lambda by 10-fold CV (lambda.min)
        cols_tr, cols_te = [], []
        tr, te = P._fill(train), P._fill(test)
        for v in P.cont:
            a, b = orth_poly2(tr[v].values.astype(float), te[v].values.astype(float))
            cols_tr.append(a)
            cols_te.append(b)
        oh_tr, oh_te = P.onehot(train)[:, len(P.cont):], P.onehot(test)[:, len(P.cont):]
        Xtr, Xte = np.column_stack(cols_tr + [oh_tr]), np.column_stack(cols_te + [oh_te])
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-12
        m = LogisticRegressionCV(Cs=np.logspace(-4, 4, 30), penalty="l1", solver="liblinear",
                                 cv=StratifiedKFold(10, shuffle=True, random_state=SEED),
                                 scoring="neg_log_loss", max_iter=5000, n_jobs=N_JOBS, random_state=SEED)
        m.fit((Xtr - mu) / sd, y, sample_weight=wt)
        return m.predict_proba((Xte - mu) / sd)[:, 1]

    def tabpfn_bagged(version, n_bags, tta, finetuned=False):
        Xp, Xb, Xt = P.coded(pres), P.coded(bg), P.coded(test)
        cont = np.zeros(Xt.shape[1], bool)
        cont[: len(P.cont)] = True
        Xt_list = [Xt]
        if tta:  # noise on continuous covariates only
            noisy = Xt.copy()
            noisy[:, cont] += rng.normal(0, 0.01, (len(Xt), cont.sum()))
            Xt_list.append(noisy)
        logits = []
        for _ in range(n_bags):
            ip = rng.choice(n_p, n_p, replace=True)
            ib = rng.choice(n_b, n_p, replace=True)
            X = np.vstack([Xp[ip], Xb[ib]])
            yy = np.r_[np.ones(n_p), np.zeros(n_p)]
            if finetuned:  # fine-tuned with categoricals as numeric codes -> same at inference
                tf = make_tabpfn(version, int(rng.integers(1 << 31)), None, FT_CKPT.format(region=region))
            else:
                tf = make_tabpfn(version, int(rng.integers(1 << 31)), P.cat_idx)
            tf.fit(X, yy)
            for X_t in Xt_list:
                logits.append(logit(np.clip(tf.predict_proba(X_t)[:, 1], 1e-6, 1 - 1e-6)))
        return expit(np.mean(logits, axis=0))

    def tabicl():
        from tabicl import TabICLClassifier
        ip = rng.choice(n_p, n_p, replace=True)
        ib = rng.choice(n_b, n_p, replace=True)
        X = np.vstack([P.coded(pres)[ip], P.coded(bg)[ib]])
        m = TabICLClassifier(device=DEVICE, random_state=SEED).fit(X, np.r_[np.ones(n_p), np.zeros(n_p)])
        return m.predict_proba(P.coded(test))[:, 1]

    runners = {
        "MaxEnt": maxent, "RF": rf, "RF down-sampled": rf_down, "BRT": brt, "XGBoost": xgboost, "Lasso": lasso,
        "TabICL": tabicl,
        "TabPFN3.5_Fast": lambda: tabpfn_bagged("v3.5", TABPFN_FAST_BAGS, tta=False),
        "TabPFN3.5_Stable": lambda: tabpfn_bagged("v3.5", TABPFN_BAGS, tta=True),
        "TabPFN2.5_Fast": lambda: tabpfn_bagged("v2.5", TABPFN_FAST_BAGS, tta=False),
        "TabPFN2.5_Stable": lambda: tabpfn_bagged("v2.5", TABPFN_BAGS, tta=True),
        "TabPFN3.5ft_Fast": lambda: tabpfn_bagged("v3.5", TABPFN_FAST_BAGS, tta=False, finetuned=True),
        "TabPFN3.5ft_Stable": lambda: tabpfn_bagged("v3.5", TABPFN_BAGS, tta=True, finetuned=True),
    }

    pred_path = os.path.join(PRED_DIR, f"{d['region']}_{d['species']}_{RUN_TAG}.npz")
    preds = dict(np.load(pred_path)) if os.path.exists(pred_path) else {}
    rows = []
    for name in models:
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        t0 = time.time()
        err = None
        try:
            if name == "Ensemble":
                parts = [preds[m] for m in ENSEMBLE_PARTS if m in preds]
                if len(parts) < len(ENSEMBLE_PARTS):
                    raise RuntimeError("ensemble parts missing")
                p = np.mean([(q - q.min()) / (np.ptp(q) or 1) for q in parts], axis=0)
            else:
                p = np.asarray(runners[name](), dtype=float)
            preds[name] = p  # full precision, so metrics recomputed from saved predictions match exactly
        except Exception as e:
            print(f"  [{d['region']}/{d['species']}] {name} error: {e!r}")
            traceback.print_exc()
            p, err = None, repr(e)[:300]
        rows.append({"Model": name, **calc_metrics(y_test, p), "Time": time.time() - t0, "Error": err})
    preds.pop("y_test", None)
    np.savez_compressed(pred_path, y_test=y_test.astype(np.int8), **preds)
    return rows


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    global TABPFN_BAGS, RUN_TAG
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", default=",".join(REGIONS))
    ap.add_argument("--models", default=",".join(CPU_MODELS))
    ap.add_argument("--max-species", type=int, default=0, help="per region; 0 = all")
    ap.add_argument("--bags", type=int, default=TABPFN_BAGS)
    ap.add_argument("--tag", required=True, help="results/v40/results_<tag>.csv")
    args = ap.parse_args()

    TABPFN_BAGS = args.bags
    RUN_TAG = args.tag
    models = args.models.split(",")
    assert not (set(models) - set(ALL_MODELS)), f"unknown: {set(models) - set(ALL_MODELS)}"
    os.makedirs(PRED_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"results_{args.tag}.csv")

    seed_everything(SEED)
    print(f"=== BENCHMARK v40 (Valavi-compatible) START {datetime.datetime.now():%Y-%m-%d %H:%M:%S} ===")
    print(f"device={DEVICE}  n_jobs={N_JOBS}  bags={TABPFN_BAGS}  models={models}  -> {out}", flush=True)

    done = set()
    if os.path.exists(out):
        # later rows supersede earlier ones; failed fits (no AUC) are not counted as done and are retried
        prev = pd.read_csv(out).drop_duplicates(["Region", "Species", "Model"], keep="last")
        prev = prev[prev["AUC"].notna()]
        done = set(zip(prev["Region"], prev["Species"], prev["Model"]))
        print(f"resume: {len(done)} rows done")

    t_start = time.time()
    for region in args.regions.split(","):
        species = load_region(region)
        if args.max_species:
            species = species[: args.max_species]
        for i, d in enumerate(species):
            todo = [m for m in models if (region, d["species"], m) not in done]
            if not todo:
                continue
            # the models of one call draw from one random stream in the order given, so a species that is only
            # partly done is re-run with the whole model list to give every model the same draws as before
            todo = models
            rng = np.random.default_rng([SEED, REGIONS.index(region), i])
            t0 = time.time()
            rows = run_models(d, todo, rng, region)
            for r in rows:
                r.update({"Region": region, "Species": d["species"], "Group": d["group"],
                          "n_pres": len(d["pres"]), "n_test": len(d["y_test"]),
                          "prevalence": float(d["y_test"].mean())})
            pd.DataFrame(rows).to_csv(out, mode="a", header=not os.path.exists(out), index=False)
            summ = " ".join(f"{r['Model']}={r['AUC']:.3f}({r['Time']:.0f}s)" for r in rows)
            print(f"[{region} {i + 1}/{len(species)} {d['species']} n_p={len(d['pres'])}] "
                  f"{time.time() - t0:.0f}s | {summ}", flush=True)
    print(f"=== DONE in {(time.time() - t_start) / 60:.1f} min ===")


if __name__ == "__main__":
    main()
