# Ensembles that add TabPFN to the Ensemble of Valavi et al. (2022): mean of 0-1 rescaled
# predictions of Lasso, GAM, MaxEnt, BRT, RF down-sampled (+ one TabPFN model).
# Reads the saved per-species predictions; writes results/v40/results_derived.csv
import glob
import os

import numpy as np
import pandas as pd

import benchmark_v40_valavi as B
from valavi_config import HERE

PRED = os.path.join(HERE, "results", "v40", "preds")
PARTS = ["Lasso", "GAM", "MaxEnt", "BRT", "RF down-sampled"]
EXTRA = {"Ensemble+TabPFN3.5": ("gpu_stable35", "TabPFN3.5_Stable"),
         "Ensemble+TabPFN3.5ft": ("gpu_ft_stable", "TabPFN3.5ft_Stable")}


def rescale(q):
    return (q - q.min()) / (np.ptp(q) or 1)


def main():
    meta = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(HERE, "results", "v40", "results_r.csv"))])
    meta = meta.drop_duplicates(["Region", "Species"], keep="last").set_index(["Region", "Species"])
    rows = []
    for f in sorted(glob.glob(os.path.join(PRED, "*_r.npz"))):
        region, sp = os.path.basename(f)[:-6].split("_", 1)
        r = dict(np.load(f))
        if not all(p in r for p in PARTS):
            continue
        y = r["y_test"].astype(int)
        for name, (tag, model) in EXTRA.items():
            g = os.path.join(PRED, f"{region}_{sp}_{tag}.npz")
            if not os.path.exists(g):
                continue
            t = dict(np.load(g))
            if model not in t:
                continue
            p = np.mean([rescale(r[m].astype(float)) for m in PARTS] + [rescale(t[model].astype(float))], axis=0)
            m = meta.loc[(region, sp)]
            rows.append({"Model": name, **B.calc_metrics(y, p), "Time": 0.0, "Error": None, "Region": region,
                         "Species": sp, "Group": m.Group, "n_pres": m.n_pres, "n_test": m.n_test,
                         "prevalence": m.prevalence})
    out = os.path.join(HERE, "results", "v40", "results_derived.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(pd.DataFrame(rows).groupby("Model").AUC.agg(["count", "mean"]).round(4))


if __name__ == "__main__":
    main()
