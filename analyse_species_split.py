# Compare species-split fine-tuning (Dinnage & Warren-style) with leave-one-region-out (LORO)
# fine-tuning, per species and with the same evaluation seeds.
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from valavi_config import HERE

F = os.path.join(HERE, "results", "v40", "finetune")
lo = pd.read_csv(os.path.join(F, "finetune_pa_results.csv"))  # default, pb_finetuned, pa_finetuned (LORO)
ss = pd.read_csv(os.path.join(F, "finetune_species_split_results.csv"))
d = pd.concat([lo, ss]).pivot_table(index=["holdout", "Species"], columns="arm", values="AUC")
arms = ["default", "pb_finetuned", "pa_finetuned", "pb_species", "pa_species"]
d = d[arms].dropna()
rows = []
for a in arms[1:]:
    x = d[a] - d["default"]
    rows.append({"arm": a, "mean_AUC": d[a].mean(), "gain_vs_default": x.mean(),
                 "P_vs_default": wilcoxon(x).pvalue, "species_improved": int((x > 0).sum())})
t = pd.DataFrame(rows).round(4)
print(len(d), "species"); print(t.to_string(index=False))
for a, b in [("pa_species", "pa_finetuned"), ("pb_species", "pb_finetuned"), ("pa_species", "pb_species")]:
    x = d[a] - d[b]
    print(f"{a} - {b}: {x.mean():+.4f}  P = {wilcoxon(x).pvalue:.2g}")
reg = (d[arms[1:]].sub(d["default"], axis=0)).groupby(level=0).mean().round(4)
print(reg)
out = os.path.join(HERE, "results", "v40", "review")
t.to_csv(os.path.join(out, "species_split_summary.csv"), index=False)
reg.to_csv(os.path.join(out, "species_split_by_region.csv"))
d.to_csv(os.path.join(out, "species_split_per_species.csv"))
