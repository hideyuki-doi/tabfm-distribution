# Round-3 (self-)review checks on the virtual species (additive design, results/v60) and real data
import glob
import json
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.special import logit
from scipy.stats import spearmanr, wilcoxon

from valavi_config import CATEGORICAL, HERE, VARS

V = os.path.join(HERE, "results", "v60")
meta = {m["sid"]: m for m in json.load(open(os.path.join(V, "meta.json")))}
MODELS = ["TabPFN3.5ft", "TabPFN3.5", "MaxEnt", "MaxNet", "GAM", "Lasso", "BRT", "RFdown", "Ensemble"]
bgcache = {}

rows_env, rows_cal = [], []
for sid, m in meta.items():
    r = m["region"]
    if r not in bgcache:
        bgcache[r] = pd.read_csv(os.path.join(HERE, "data", "background50k", f"{r}bg50k.csv"))
    bg = bgcache[r]
    d = np.load(os.path.join(V, "data", f"{sid}.npz"))
    te = d["te"]; pt = d["p_true"]; yt = d["y_true"]; nt = len(te)
    for n in [10, 30, 100, 300]:
        preds = {}
        for f in glob.glob(os.path.join(V, "preds", f"{sid}_n{n}_*.npz")):
            z = np.load(f); preds.update({k: z[k] for k in z.files})
        pres = bg.iloc[d["pres"][:n]]
        # environmental envelope of the presences on the niche covariates
        inside = np.ones(nt, bool)
        for v in m["niche"]:
            lo, hi = pres[v].min(), pres[v].max()
            inside &= (bg[v].values[te] >= lo) & (bg[v].values[te] <= hi)
        for mod in MODELS:
            p = np.clip(preds[mod][:nt].astype(float), 1e-6, 1 - 1e-6)
            rows_env.append({"sid": sid, "n": n, "biased": m["biased"], "Model": mod, "frac_inside": inside.mean(),
                             "rho_all": spearmanr(p, pt)[0],
                             "rho_inside": spearmanr(p[inside], pt[inside])[0] if inside.sum() > 50 else np.nan})
            x = logit(p)
            try:
                rows_cal.append({"sid": sid, "n": n, "Model": mod,
                                 "slope": sm.Logit(yt, sm.add_constant(x)).fit(disp=0).params[1]})
            except Exception:
                pass
E = pd.DataFrame(rows_env)
print("(a) fraction of test points inside the presence envelope, by n:", E.groupby("n").frac_inside.mean().round(2).to_dict())
for col in ["rho_all", "rho_inside"]:
    piv = E.pivot_table(index=["sid", "n"], columns="Model", values=col)
    out = {m: round((piv["TabPFN3.5ft"] - piv[m]).mean(), 4) for m in ["MaxEnt", "Lasso", "GAM", "Ensemble"]}
    print(f"    FT minus comparator, {col}:", out)
piv_in = E.pivot_table(index=["sid", "n"], columns="Model", values="rho_inside").dropna()
for m in ["MaxEnt", "Lasso", "GAM", "Ensemble"]:
    dd = piv_in["TabPFN3.5ft"] - piv_in[m]
    print(f"    inside: FT-{m} {dd.mean():+.4f} P={wilcoxon(dd).pvalue:.2g} n={len(dd)}")
print("    means inside:", piv_in.mean().round(3).to_dict())
E.to_csv(os.path.join(V, "envelope.csv"), index=False)

# (b) bias entanglement: correlation between the bias covariate and the niche covariates
res = pd.read_csv(os.path.join(V, "results.csv"))
rows = []
for sid, m in meta.items():
    if not m["biased"]:
        continue
    bg = bgcache[m["region"]]
    c = max(abs(np.corrcoef(bg[m["bias_var"]], bg[v])[0, 1]) for v in m["niche"])
    rows.append({"sid": sid, "corr": c})
B = pd.DataFrame(rows).set_index("sid")
piv = res[res.biased].pivot_table(index=["sid", "n"], columns="Model", values="curve_r")
dd = (piv["TabPFN3.5ft"] - piv["MaxEnt"]).rename("d").reset_index()
dd["corr"] = dd.sid.map(B["corr"])
f = smf.ols("d ~ corr + np.log10(n)", dd).fit()
print("(b) biased species: FT-MaxEnt curve deficit vs max |corr(bias var, niche var)|: slope",
      round(f.params["corr"], 3), "P", round(f.pvalues["corr"], 4), "| corr range", B["corr"].round(2).describe()[["min", "50%", "max"]].to_dict())
piv2 = res[res.biased].pivot_table(index=["sid", "n"], columns="Model", values="spearman")
dd2 = (piv2["TabPFN3.5ft"] - piv2["MaxEnt"]).rename("d").reset_index(); dd2["corr"] = dd2.sid.map(B["corr"])
f2 = smf.ols("d ~ corr + np.log10(n)", dd2).fit()
print("    same for Spearman: slope", round(f2.params["corr"], 3), "P", round(f2.pvalues["corr"], 4))

# (c) calibration slope against true presence/absence in virtual species
C = pd.DataFrame(rows_cal)
print("(c) VS calibration slope (median) by model:", C.groupby("Model").slope.median().round(2).to_dict())
print("    by n for FT and MaxEnt:", C[C.Model.isin(["TabPFN3.5ft", "MaxEnt"])].groupby(["Model", "n"]).slope.median().round(2).to_dict())

# (d) real data: MaxEnt/MaxNet calibration on their natural (cloglog) scale
import plot_v40 as P
df = P.load(0.9)
auc = P.pivot(df, "AUC")
cal = []
for (r, sp) in auc.index:
    z = np.load(os.path.join(HERE, "results", "v40", "preds", f"{r}_{sp}_r.npz"))
    y = z["y_test"]
    for mod in ["MaxEnt", "MaxNet"]:
        p = np.clip(z[mod].astype(float), 1e-6, 1 - 1e-6)
        for lab_, x in (("logit", logit(p)), ("cloglog", np.log(-np.log(1 - p)))):
            try:
                cal.append({"Model": mod, "scale": lab_, "slope": sm.Logit(y, sm.add_constant(x)).fit(disp=0).params[1]})
            except Exception:
                pass
print("(d) real data, MaxEnt/MaxNet slope by predictor scale:", pd.DataFrame(cal).groupby(["Model", "scale"]).slope.median().round(3).to_dict())
