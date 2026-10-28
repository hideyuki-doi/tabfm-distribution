# Fig 5A: spatial-block vs random cross-validation within the presence-background data
# (Valavi et al. 2023 design). Per-species AUC = mean over the 5 folds.
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plot_v40 as P
from valavi_config import HERE

RES = os.path.join(HERE, "results", "v50")
LAB = {"TabPFN3.5": "TabPFN 3.5", "TabPFN3.5ft": "TabPFN 3.5-FT"}


def main():
    P.style()
    d = pd.concat([pd.read_csv(os.path.join(RES, f"results_{f}.csv")) for f in ("r", "gpu")])
    d = d.drop_duplicates(["Region", "Species", "cv", "fold", "Model"], keep="last")
    sp = d.groupby(["cv", "Region", "Species", "Model"]).AUC.mean().reset_index()
    stat = sp.groupby(["Model", "cv"]).AUC.agg(["mean", "sem"]).unstack("cv")
    order = stat[("mean", "spatial")].sort_values().index
    stat = stat.loc[order]
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    y = np.arange(len(order))
    ax.hlines(y, stat[("mean", "spatial")], stat[("mean", "random")], color=P.GRID, lw=3, zorder=1)
    for cv, c, mk in (("random", "#9aa5b1", "o"), ("spatial", "#1f6f8b", "s")):
        ax.errorbar(stat[("mean", cv)], y, xerr=stat[("sem", cv)], fmt=mk, color=c, ms=6, capsize=2,
                    label={"random": "Random 5-fold CV", "spatial": "Spatial-block 5-fold CV"}[cv], zorder=2)
    ax.set_yticks(y, [LAB.get(m, m) for m in order])
    for t in ax.get_yticklabels():
        if "TabPFN" in t.get_text():
            t.set_fontweight("bold")
    ax.set_xlabel("AUC$_{ROC}$ at survey sites in held-out folds (mean ± SE, 178 species)")
    ax.legend(frameon=False, loc="lower right", fontsize=8.5)
    fig.tight_layout()
    P.save(fig, "Fig5A_spatial_vs_random_cv")


if __name__ == "__main__":
    main()
