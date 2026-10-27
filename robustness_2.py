# Round-2 (self-)review checks
import glob
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import wilcoxon

import plot_v40 as P
import valavi_stats as VS
import make_main_figs as M
from valavi_config import DATA_DIR, HERE, REGIONS

df = P.load(0.9)
auc = P.pivot(df, "AUC")
FT, TP, FTF = "TabPFN3.5ft_Stable", "TabPFN3.5_Stable", "TabPFN3.5ft_Fast"
ph = VS.posthoc_aligned_ranks(auc[M.MODELS])

def boot(d, n=5000, seed=1):
    rng = np.random.default_rng(seed)
    b = [pd.concat([g.sample(len(g), replace=True, random_state=int(rng.integers(1 << 31))) for _, g in d.groupby(level=0)]).mean() for _ in range(n)]
    return np.percentile(b, [2.5, 97.5])

print("(a) TabPFN 3.5 (no FT) vs comparators, Shaffer P (13-model family):")
for m in ["MaxEnt", "MaxNet", "RF down-sampled", "BRT", "Ensemble"]:
    d = auc[TP] - auc[m]
    print(f"   vs {m}: diff {d.mean():+.4f} boot95 {boot(d).round(4)} Shaffer P {ph.loc[TP, m]:.3f}")
print("(b) Fast FT (5 bags) vs comparators (paired Wilcoxon, bootstrap):")
for m in ["MaxEnt", "MaxNet", "RF down-sampled", "BRT", "Ensemble"]:
    d = auc[FTF] - auc[m]
    print(f"   vs {m}: {d.mean():+.4f} boot95 {boot(d).round(4)} Wilcoxon P {wilcoxon(d).pvalue:.3g} wins {(d>0).sum()}")
# (c) variance partition of AUC: species vs model
long = auc[M.SINGLE].stack().rename("auc").reset_index()
long["sp"] = long.Region + "_" + long.Species
import statsmodels.formula.api as smf
f = smf.ols("auc ~ C(sp) + C(Model)", long).fit()
an = sm.stats.anova_lm(f, typ=2)
print("(c) variance partition (SS share):", (an.sum_sq / an.sum_sq.sum()).round(3).to_dict())
f2 = smf.ols("auc ~ C(Region) + C(Model)", long).fit(); an2 = sm.stats.anova_lm(f2, typ=2)
print("    region vs model SS share:", (an2.sum_sq / an2.sum_sq.sum()).round(3).to_dict())
# (d) calibration slope: Fast vs Stable
PRED = os.path.join(HERE, "results", "v40", "preds")
rows = []
for (r, sp) in auc.index:
    preds = {}
    y = None
    for fn in glob.glob(os.path.join(PRED, f"{r}_{sp}_gpu_*.npz")):
        z = np.load(fn); y = z["y_test"]; preds.update({k: z[k] for k in z.files if k != "y_test"})
    for m in ["TabPFN3.5_Fast", "TabPFN3.5_Stable", "TabPFN3.5ft_Fast", "TabPFN3.5ft_Stable"]:
        p = np.clip(preds[m].astype(float), 1e-6, 1 - 1e-6); x = np.log(p / (1 - p))
        try:
            rows.append({"m": m, "slope": sm.Logit(y, sm.add_constant(x)).fit(disp=0).params[1]})
        except Exception:
            pass
print("(d) calibration slope median:", pd.DataFrame(rows).groupby("m").slope.median().round(3).to_dict())
# (e) spatial CV: FT - default by region and design
d = pd.concat([pd.read_csv(os.path.join(HERE, "results", "v50", f"results_{x}.csv")) for x in ("gpu",)])
sp = d.groupby(["cv", "Region", "Species", "Model"]).AUC.mean().unstack("Model")
e = (sp["TabPFN3.5ft_Fast"] - sp["TabPFN3.5_Fast"]).groupby(level=["cv", "Region"]).mean().unstack("Region").round(4)
print("(e) FT - default under CV designs (TGB background):\n", e)
# (f) 226 vs 225 species
po = pd.concat([pd.read_csv(os.path.join(DATA_DIR, "train_po", f"{r}train_po.csv")) for r in REGIONS])
allsp = set(po.spid.unique()); used = set(auc.index.get_level_values(1))
print("(f) species in PO data not used:", sorted(allsp - used), "| n PO species", len(allsp))
