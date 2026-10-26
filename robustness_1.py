# Analyses added in response to (self-)review:
#  1. uncertainty of FT - comparator differences: species bootstrap stratified by region,
#     linear mixed model with region and species effects, and region-level consistency
#  2. does the advantage grow as presences decrease? regression of the paired difference on
#     log10(n presences) with region fixed effects
#  3. calibration: Miller's calibration slope (logistic regression of observed presence on the
#     logit of the prediction) at the independent survey sites
#  4. categorical coding: TabPFN 3.5 Fast with categorical features (benchmark) vs numeric codes
#     (default arm of the fine-tuning experiments), same species
import glob
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

import plot_v40 as P
from valavi_config import HERE

OUT = os.path.join(HERE, "results", "v40", "review")
os.makedirs(OUT, exist_ok=True)
FT = "TabPFN3.5ft_Stable"
COMP = ["MaxEnt", "MaxNet", "RF down-sampled", "BRT", "GAM", "Ensemble", "TabPFN3.5_Stable"]

df = P.load(0.9)
auc = P.pivot(df, "AUC")
npres = df.drop_duplicates(["Region", "Species"]).set_index(["Region", "Species"]).n_pres.reindex(auc.index)

# 1. uncertainty
rng = np.random.default_rng(1)
rows = []
for m in COMP:
    d = (auc[FT] - auc[m])
    boots = []
    for _ in range(5000):
        s = pd.concat([g.sample(len(g), replace=True, random_state=int(rng.integers(1 << 31)))
                       for _, g in d.groupby(level=0)])
        boots.append(s.mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    long = pd.DataFrame({"d": d.values, "region": d.index.get_level_values(0)})
    mm = smf.mixedlm("d ~ 1", long, groups="region").fit(reml=True)
    est, se = mm.fe_params["Intercept"], mm.bse_fe["Intercept"]
    reg = d.groupby(level=0).mean()
    rows.append({"comparator": m, "mean_diff": d.mean(), "boot_lo": lo, "boot_hi": hi,
                 "lmm_est": est, "lmm_lo": est - 1.96 * se, "lmm_hi": est + 1.96 * se,
                 "regions_positive": int((reg > 0).sum()), "species_positive": int((d > 0).sum())})
u = pd.DataFrame(rows).round(4)
u.to_csv(os.path.join(OUT, "uncertainty.csv"), index=False)
print(u.to_string(index=False))

# 2. sample size
rows = []
for m in ["MaxEnt", "MaxNet", "RF down-sampled", "BRT", "Ensemble"]:
    dd = pd.DataFrame({"d": (auc[FT] - auc[m]).values, "lp": np.log10(npres.values),
                       "region": auc.index.get_level_values(0)})
    f = smf.ols("d ~ lp + C(region)", dd).fit()
    rows.append({"comparator": m, "slope_per_log10": f.params["lp"], "se": f.bse["lp"], "P": f.pvalues["lp"]})
s = pd.DataFrame(rows).round(4)
s.to_csv(os.path.join(OUT, "sample_size.csv"), index=False)
print(s.to_string(index=False))

# 3. calibration slope
PRED = os.path.join(HERE, "results", "v40", "preds")
models = ["GAM", "Lasso", "BRT", "RF", "RF down-sampled", "MaxNet", "MaxEnt", "Ensemble", "XGBoost",
          "TabPFN3.5_Stable", "TabPFN3.5ft_Stable", "TabPFN2.5_Stable", "TabICL"]
cal = []
for (r, sp) in auc.index:
    preds = {}
    for f in glob.glob(os.path.join(PRED, f"{r}_{sp}_*.npz")):
        z = np.load(f)
        y = z["y_test"]
        preds.update({k: z[k] for k in z.files if k != "y_test"})
    for m in models:
        if m not in preds:
            continue
        p = np.clip(preds[m].astype(float), 1e-6, 1 - 1e-6)
        x = np.log(p / (1 - p))
        if np.ptp(x) == 0 or y.min() == y.max():
            continue
        try:
            fit = sm.Logit(y, sm.add_constant(x)).fit(disp=0, maxiter=200)
            cal.append({"Region": r, "Species": sp, "Model": m, "slope": fit.params[1]})
        except Exception:
            pass
cal = pd.DataFrame(cal)
cs = cal.groupby("Model").slope.agg(["median", lambda v: np.percentile(v, 25), lambda v: np.percentile(v, 75), "count"])
cs.columns = ["median", "q25", "q75", "n"]
cs = cs.sort_values("median").round(3)
cs.to_csv(os.path.join(OUT, "calibration_slope.csv"))
print(cs.to_string())

# 4. categorical coding
fp = pd.read_csv(os.path.join(HERE, "results", "v40", "finetune", "finetune_pa_results.csv"))
dflt = fp[fp.arm == "default"].set_index(["holdout", "Species"]).AUC
dflt.index.names = ["Region", "Species"]
both = pd.concat([auc["TabPFN3.5_Fast"].rename("categorical"), dflt.rename("numeric")], axis=1).dropna()
print("categorical vs numeric coding (TabPFN 3.5 Fast):", both.mean().round(4).to_dict(), "n =", len(both),
      "| by region:", (both.categorical - both.numeric).groupby(level=0).mean().round(4).to_dict())
both.to_csv(os.path.join(OUT, "categorical_coding.csv"))
