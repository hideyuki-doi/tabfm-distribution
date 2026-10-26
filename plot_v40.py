# ==============================================================================
# Figures for Benchmark v40 in the style of Valavi et al. (2022) Ecol. Monogr. 92:e01486
#   Fig2  circular heatmap of AUC_ROC (all models x all species)
#   Fig3  mean AUC_ROC vs mean COR (+/- SE from a mixed model), 3 performance groups
#   Fig4  Friedman aligned-rank post-hoc p-values (Shaffer) + clustering dendrogram
#   Fig7  average ranks: AUC_ROC vs COR (A) and AUC_ROC vs AUC_PRG (B)
#   Fig8  % of species where a model is top 1 / 2 / 3
#   Fig9  AUC_ROC for species with < 30 and >= 30 presences
#   Fig10 mean AUC_ROC per region, with the 21 published models of Valavi et al.
#   FigS1 replication check: published vs re-fitted reference models (per region)
# Usage: python plot_v40.py [--min-coverage 0.95]
# ==============================================================================
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage

import valavi_stats as VS
from valavi_config import HERE

OUT = os.path.join(HERE, "figures", "v40")
RES = os.path.join(HERE, "results", "v40")
REF = os.path.join(HERE, "reference", "valavi2022_fig10_auc_by_region.csv")
N_SP_REGION = {"SWI": 30, "SA": 30, "NZ": 52, "NSW": 53, "AWT": 40, "CAN": 20}  # 225 species

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
LABEL = {"TabPFN3.5_Fast": "TabPFN 3.5 Fast", "TabPFN3.5_Stable": "TabPFN 3.5 Stable",
         "TabPFN3.5ft_Fast": "TabPFN 3.5-FT Fast", "TabPFN3.5ft_Stable": "TabPFN 3.5-FT Stable",
         "Ensemble+TabPFN3.5": "Ensemble + TabPFN 3.5", "Ensemble+TabPFN3.5ft": "Ensemble + TabPFN 3.5-FT",
         "TabPFN2.5_Fast": "TabPFN 2.5 Fast", "TabPFN2.5_Stable": "TabPFN 2.5 Stable"}
FAMILY = {"RF": "Tree-based", "RF down-sampled": "Tree-based", "BRT": "Tree-based", "XGBoost": "Tree-based",
          "GAM": "Regression", "Lasso": "Regression", "MaxNet": "Regression", "MaxEnt": "Regression",
          "Ensemble": "Other", "TabICL": "Foundation", "TabPFN3.5_Fast": "Foundation",
          "TabPFN3.5_Stable": "Foundation", "TabPFN2.5_Fast": "Foundation", "TabPFN2.5_Stable": "Foundation",
          "TabPFN3.5ft_Fast": "Foundation", "TabPFN3.5ft_Stable": "Foundation",
          "Ensemble+TabPFN3.5": "Other", "Ensemble+TabPFN3.5ft": "Other"}
# colour = model family; the two TabPFN generations and TabICL get their own hue
COLOR = {"Tree-based": "#2a78d6", "Regression": "#1baf7a", "Other": "#52514e"}
MODEL_COLOR = {"TabPFN3.5_Fast": "#eb6834", "TabPFN3.5_Stable": "#eb6834",
               "TabPFN2.5_Fast": "#4a3aa7", "TabPFN2.5_Stable": "#4a3aa7", "TabICL": "#e87ba4",
               "TabPFN3.5ft_Fast": "#b23a0e", "TabPFN3.5ft_Stable": "#b23a0e",
               "Ensemble+TabPFN3.5": "#8a4a2a", "Ensemble+TabPFN3.5ft": "#8a4a2a"}
# same model names as in the published Fig. 10
PUBLISHED_NAME = {"RF down-sampled": "RF down-sampled"}


def lab(m):
    return LABEL.get(m, m)


def col(m):
    return MODEL_COLOR.get(m, COLOR.get(FAMILY.get(m, "Other"), INK2))


def style():
    plt.rcParams.update({
        "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.7, "axes.axisbelow": True, "figure.facecolor": "white", "savefig.dpi": 300,
    })


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"  - figures/v40/{name}.png")


def load(min_cov):
    files = [f for f in sorted(glob.glob(os.path.join(RES, "results_*.csv"))) if "test" not in f]
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df.drop_duplicates(["Region", "Species", "Model"], keep="last")
    n_sp = df.groupby(["Region", "Species"]).ngroups
    cov = df.dropna(subset=["AUC"]).groupby("Model").size() / n_sp
    keep = cov[cov >= min_cov].index.tolist()
    print(f"species={n_sp}; coverage: " + ", ".join(f"{m}={c:.0%}" for m, c in cov.sort_values().items()))
    df = df[df.Model.isin(keep)]
    global RAW
    RAW = df.copy()
    # complete cases only, so every model is evaluated on the same species
    piv = df.pivot_table(index=["Region", "Species"], columns="Model", values="AUC")
    ok = piv.dropna().index
    df = df.set_index(["Region", "Species"]).loc[ok].reset_index()
    print(f"complete species: {len(ok)}")
    return df


def pivot(df, metric):
    return df.pivot_table(index=["Region", "Species"], columns="Model", values=metric)


def model_order(df):
    """Models sorted by overall mean AUC (lowest first), as in the paper."""
    return pivot(df, "AUC").mean().sort_values().index.tolist()


# ---------------------------------------------------------
# Fig 2: circular heatmap
# ---------------------------------------------------------
def fig2_circular(df, order):
    auc = pivot(df, "AUC")[order]
    npres = df.drop_duplicates(["Region", "Species"]).set_index(["Region", "Species"]).n_pres
    present = set(auc.index.get_level_values(0))
    regions = [r for r in ["NZ", "SA", "SWI", "AWT", "CAN", "NSW"] if r in present]  # clockwise, as in the paper
    gap = 3  # empty columns between regions
    cols = []
    for r in regions:
        sub = auc.loc[r]
        cols += [(r, s) for s in sub.mean(axis=1).sort_values(ascending=False).index] + [None] * gap
    n_col, n_ring = len(cols), len(order)
    r0 = 4.0  # inner radius (in ring units)
    cmap = plt.get_cmap("Spectral", 10)
    norm = BoundaryNorm(np.linspace(0, 1, 11), 10)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    width = 2 * np.pi / n_col
    for j, c in enumerate(cols):
        if c is None:
            continue
        th = j * width
        vals = auc.loc[c].values
        ax.bar(np.full(n_ring, th), np.ones(n_ring), width=width, bottom=r0 + np.arange(n_ring),
               color=cmap(norm(vals)), edgecolor="white", linewidth=0.15, align="edge")
        npr = npres.loc[c]
        h = np.log10(npr)
        colr = "#f6c89f" if npr < 100 else ("#ef6f6c" if npr <= 1000 else "#6b1d1d")
        ax.bar(th, h * 0.9, width=width, bottom=r0 + n_ring + 0.4, color=colr, edgecolor="white", linewidth=0.15,
               align="edge")
    # region labels
    j = 0
    for r in regions:
        n = (auc.loc[r].shape[0])
        mid = (j + n / 2) * width
        ax.text(mid, r0 + n_ring + 4.6, r, ha="center", va="center", fontsize=13, fontweight="bold",
                rotation=-np.degrees(mid) if np.cos(mid) >= 0 else 180 - np.degrees(mid))
        j += n + gap
    # ring numbers in the last gap (between NSW and NZ)
    for i in range(n_ring):
        ax.text((n_col - gap / 2) * width, r0 + i + 0.5, str(i + 1), ha="center", va="center", fontsize=5.5,
                color=INK2)
    ax.set_ylim(0, r0 + n_ring + 5.5)
    ax.axis("off")
    # model key in the centre
    key = "\n".join(f"{i + 1:>2}- {lab(m)}" for i, m in enumerate(order))
    ax.text(0, 0, key, ha="center", va="center", fontsize=6.5, family="monospace", color=INK)
    cax = fig.add_axes([0.38, 0.085, 0.24, 0.015])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal",
                      ticks=[0, 0.25, 0.5, 0.75, 1])
    cb.set_label("AUC$_{ROC}$")
    fig.text(0.5, 0.015, "outer bars: log$_{10}$(no. of presence records);  light < 100, pink 100–1,000, "
             "dark > 1,000", ha="center", fontsize=8, color=INK2)
    save(fig, "Fig2_circular_AUC")


# ---------------------------------------------------------
# Fig 3: mean AUC vs mean COR with mixed-model SE, three groups
# ---------------------------------------------------------
def mixed_model_means(df, metric, order):
    """Model means and SEs from metric ~ 0 + model + (1|species) + (1|model:region)
    (frequentist analogue of the paper's Bayesian GLMM)."""
    import statsmodels.formula.api as smf
    d = df[["Model", "Region", "Species", metric]].dropna().copy()
    d["sp"] = d.Region + "_" + d.Species
    d["mr"] = d.Model + "_" + d.Region
    d["g"] = 1
    m = smf.mixedlm(f"{metric} ~ 0 + C(Model)", d, groups="g",
                    vc_formula={"sp": "0 + C(sp)", "mr": "0 + C(mr)"}).fit(reml=True)
    est = pd.Series({k.replace("C(Model)[", "").rstrip("]"): v for k, v in m.fe_params.items()})
    se = pd.Series({k.replace("C(Model)[", "").rstrip("]"): v for k, v in m.bse_fe.items()})
    return est[order], se[order]


def performance_groups(mean_auc, mean_cor, k=3):
    Z = linkage(np.column_stack([mean_auc, mean_cor]), method="complete")
    g = fcluster(Z, k, criterion="maxclust")
    # relabel 1 = low ... k = high by mean AUC
    rank = pd.Series(mean_auc.values, index=g).groupby(level=0).mean().rank().astype(int)
    return pd.Series([rank[x] for x in g], index=mean_auc.index), Z


def fig3_auc_cor(df, order):
    auc, auc_se = mixed_model_means(df, "AUC", order)
    cor, cor_se = mixed_model_means(df, "COR", order)
    groups, _ = performance_groups(auc, cor)
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    x0, x1 = auc.min() - 0.012, auc.max() + 0.015
    y0, y1 = cor.min() - 0.012, cor.max() + 0.015
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    slope, icpt = np.polyfit(auc.values, cor.values, 1)
    xs = np.array([x0, x1])
    ax.plot(xs, icpt + slope * xs, color="#9b9a95", lw=1.2, zorder=1)
    # dashed group boundaries: perpendicular (on screen) to the LS line, midway between groups
    nx = lambda x: (x - x0) / (x1 - x0)
    ny = lambda y: (y - y0) / (y1 - y0)
    u = np.array([1.0, slope * (x1 - x0) / (y1 - y0)])
    u /= np.linalg.norm(u)
    proj = nx(auc.values) * u[0] + ny(cor.values) * u[1]
    for gk in range(1, groups.max()):
        b = (proj[groups.values == gk].max() + proj[groups.values == gk + 1].min()) / 2
        t = np.linspace(-2, 2, 50)
        px, py = b * u[0] - t * u[1], b * u[1] + t * u[0]
        ax.plot(x0 + px * (x1 - x0), y0 + py * (y1 - y0), ls="--", color="#9b9a95", lw=1, zorder=1)
    texts = []
    for m in order:
        c = col(m)
        ax.errorbar(auc[m], cor[m], xerr=auc_se[m], yerr=cor_se[m], fmt="none", ecolor="#d9d8d3", lw=1, zorder=2)
        ax.scatter(auc[m], cor[m], s=40, color=c, zorder=3, edgecolors="white", linewidths=1)
        texts.append(ax.text(auc[m], cor[m], lab(m), fontsize=8.5, color=c, zorder=4))
    from adjustText import adjust_text
    adjust_text(texts, ax=ax, x=auc.values, y=cor.values, expand=(1.3, 1.6),
                arrowprops=dict(arrowstyle="-", color="#9b9a95", lw=0.6))
    ax.set_xlabel("AUC$_{ROC}$")
    ax.set_ylabel("COR")
    ax.grid(False)
    ax.set_title("Mean AUC$_{ROC}$ vs. mean COR (± SE, mixed model; dashed lines: low / moderate / high groups)",
                 fontsize=10, loc="left")
    save(fig, "Fig3_AUC_vs_COR")
    return auc, cor, groups


# ---------------------------------------------------------
# Fig 4: post-hoc p-values + dendrogram
# ---------------------------------------------------------
def fig4_posthoc(df, order, auc, cor):
    data = pivot(df, "AUC")[order]
    stat, p, _ = VS.friedman_aligned_ranks(data)
    P = VS.posthoc_aligned_ranks(data, correct="shaffer")
    _, Z = performance_groups(auc, cor)
    fig = plt.figure(figsize=(9, 9.5))
    axd = fig.add_axes([0.2, 0.78, 0.7, 0.15])
    dn = dendrogram(Z, labels=[lab(m) for m in order], ax=axd, color_threshold=0, above_threshold_color=INK,
                    no_labels=True)
    axd.axis("off")
    leaf = [order[i] for i in dn["leaves"]]
    ax = fig.add_axes([0.2, 0.2, 0.7, 0.58])
    M = P.loc[leaf, leaf].values.astype(float)
    np.fill_diagonal(M, np.nan)
    cmap = plt.get_cmap("magma_r").copy()
    cmap.set_bad("#7f7f7f")
    ax.imshow(M, cmap=cmap, vmin=0, vmax=1, origin="lower", aspect="auto")
    for i in range(len(leaf)):
        for j in range(len(leaf)):
            if i != j:
                v = M[i, j]
                ax.text(j, i, f"{v:.2g}" if v < 0.995 else "1", ha="center", va="center", fontsize=6.5,
                        color="white" if v > 0.5 else ("#c9c8c3" if v < 0.005 else INK))
    ax.set_xticks(range(len(leaf)), [lab(m) for m in leaf], rotation=55, ha="right")
    ax.set_yticks(range(len(leaf)), [lab(m) for m in leaf])
    ax.grid(False)
    fig.suptitle(f"Post-hoc Friedman aligned-rank test, Shaffer-adjusted P (AUC$_{{ROC}}$); "
                 f"Friedman aligned rank T = {stat:.0f}, P = {p:.1e}", fontsize=9, x=0.2, ha="left", y=0.97)
    save(fig, "Fig4_posthoc_pvalues_AUC")
    # also COR and PRG tables for the text
    for met in ["COR", "PRG"]:
        VS.posthoc_aligned_ranks(pivot(df, met)[order]).round(4).to_csv(os.path.join(OUT, f"posthoc_{met}.csv"))
    P.round(4).to_csv(os.path.join(OUT, "posthoc_AUC.csv"))


# ---------------------------------------------------------
# Fig 7: average ranks
# ---------------------------------------------------------
def fig7_ranks(df, order):
    rk = {m: pivot(df, m)[order].rank(axis=1, ascending=False).mean() for m in ["AUC", "COR", "PRG"]}
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 11))
    for ax, ym, tag in zip(axes, ["COR", "PRG"], ["A", "B"]):
        x, y = rk["AUC"], rk[ym]
        for m in order:
            ax.scatter(x[m], y[m], s=30, color=col(m), zorder=3)
            ax.annotate(lab(m), (x[m], y[m]), xytext=(4, 4), textcoords="offset points", fontsize=8.5, color=col(m))
        mid = (len(order) + 1) / 2
        ax.axvline(mid, ls="--", color="#bdbcb7", lw=1)
        ax.axhline(mid, ls="--", color="#bdbcb7", lw=1)
        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.set_xlabel("Average rank, AUC$_{ROC}$")
        ax.set_ylabel("Average rank, " + ("COR" if ym == "COR" else "AUC$_{PRG}$"))
        ax.text(0.04, 0.9, tag, transform=ax.transAxes, fontsize=14, fontweight="bold")
        ax.grid(False)
    fig.suptitle("Average ranks across species (better models toward top right)", x=0.02, ha="left", fontsize=11)
    fig.tight_layout()
    save(fig, "Fig7_average_ranks")


# ---------------------------------------------------------
# Fig 8: % of species top 1 / 2 / 3
# ---------------------------------------------------------
def fig8_topk(df, order):
    fig, axes = plt.subplots(3, 1, figsize=(0.62 * len(order) + 2, 7.5), sharex=True)
    for ax, met, nm in zip(axes, ["AUC", "PRG", "COR"], ["AUC$_{ROC}$", "AUC$_{PRG}$", "COR"]):
        r = pivot(df, met)[order].rank(axis=1, ascending=False, method="min")
        M = np.array([(r <= k).mean() * 100 for k in (1, 2, 3)])
        ax.imshow(M, cmap="viridis_r", aspect="auto", vmin=0, vmax=max(35, M.max()))
        for i in range(3):
            for j in range(len(order)):
                ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=7.5,
                        color="white" if M[i, j] > 0.45 * max(35, M.max()) else INK)
        ax.set_yticks(range(3), ["Top 1", "Top 2", "Top 3"])
        ax.grid(False)
        ax.yaxis.set_label_position("right")
        ax.set_ylabel(nm, rotation=270, labelpad=14)
    axes[-1].set_xticks(range(len(order)), [lab(m) for m in order], rotation=55, ha="right")
    fig.suptitle("Percentage of species for which a model is the top model or among the top two / three",
                 x=0.02, ha="left", fontsize=10)
    fig.tight_layout()
    save(fig, "Fig8_top_models")


# ---------------------------------------------------------
# Fig 9: < 30 vs >= 30 presences
# ---------------------------------------------------------
def fig9_low_occ(df, order):
    piv = pivot(df, "AUC")[order]
    npres = df.drop_duplicates(["Region", "Species"]).set_index(["Region", "Species"]).n_pres.reindex(piv.index)
    fams = ["Tree-based", "Regression", "Other", "Foundation"]
    fams = [f for f in fams if any(FAMILY.get(m) == f for m in order)]
    widths = [sum(FAMILY.get(m) == f for m in order) for f in fams]
    fig, axes = plt.subplots(1, len(fams), figsize=(0.55 * len(order) + 2.5, 4.8), sharey=True,
                             gridspec_kw={"width_ratios": widths}, squeeze=False)
    axes = axes[0]
    lo, hi = npres < 30, npres >= 30
    for ax, f in zip(axes, fams):
        ms = [m for m in piv[lo].median().sort_values().index if FAMILY.get(m) == f]
        x = np.arange(len(ms))
        for sel, c, nm, dx in [(lo, "#d99412", f"< 30 presences (n={lo.sum()})", -0.12),
                               (hi, "#1a9fa8", f"≥ 30 presences (n={hi.sum()})", 0.12)]:
            mu = piv.loc[sel, ms].mean()
            se = piv.loc[sel, ms].std() / np.sqrt(sel.sum())
            ax.errorbar(x + dx, mu, yerr=se, fmt="o", color=c, ms=5, capsize=2.5, label=nm)
        ax.set_xticks(x, [lab(m) for m in ms], rotation=55, ha="right")
        ax.set_title(f, fontsize=10, fontweight="bold")
        ax.set_xlim(-0.6, len(ms) - 0.4)
    axes[0].set_ylabel("AUC$_{ROC}$ (mean ± SE)")
    axes[0].legend(frameon=False, loc="lower left", fontsize=8.5)
    fig.tight_layout()
    save(fig, "Fig9_low_vs_high_presences")


# ---------------------------------------------------------
# Fig 10: region heatmap with the 21 published models
# ---------------------------------------------------------
def fig10_region(df, order):
    pub = pd.read_csv(REF, comment="#").set_index("Model")
    fm = [m for m in order if FAMILY.get(m) == "Foundation" or m.startswith("Ensemble+")]
    # foundation models: all species they were run on (not only the complete-case subset)
    ours = pivot(RAW, "AUC")[fm].groupby(level="Region").mean().T.reindex(columns=pub.columns)
    tab = pd.concat([pub, ours.loc[fm].rename(index=lambda m: lab(m) + " †")])
    w = pd.Series(N_SP_REGION)
    overall = (tab[w.index] * w).sum(axis=1) / w.sum()
    tab = tab.loc[overall.sort_values().index]
    regions = tab.mean().sort_values(ascending=False).index.tolist()
    M = tab[regions].T.values
    fig, ax = plt.subplots(figsize=(0.62 * len(tab) + 1.5, 4.6))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("yb", ["#fde725", "#8f8a6a", "#2b3a67", "#001b5e"])
    im = ax.imshow(M, cmap=cmap, vmin=0.53, vmax=0.83, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isnan(M[i, j]):
                continue
            ax.text(j, i, f"{M[i, j]:.3f}".lstrip("0"), ha="center", va="center", fontsize=7,
                    color="white" if M[i, j] > 0.66 else INK)
    xl = list(tab.index)
    ax.set_xticks(range(len(xl)), xl, rotation=55, ha="right")
    for t in ax.get_xticklabels():
        if t.get_text().endswith("†"):
            t.set_color(MODEL_COLOR.get(next((m for m in fm if lab(m) + " †" == t.get_text()), ""), INK))
            t.set_fontweight("bold")
    ax.set_yticks(range(len(regions)), regions)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.01)
    cb.set_label("AUC$_{ROC}$")
    ax.set_title("Mean AUC$_{ROC}$ per region. Unmarked: 21 models published in Valavi et al. (2022, Fig. 10); "
                 "† this study (same data and protocol).\nModels sorted by overall AUC$_{ROC}$ (species-weighted), "
                 "regions by mean AUC$_{ROC}$.", fontsize=8.5, loc="left", pad=8)
    save(fig, "Fig10_region_AUC_with_published")
    out = tab.copy()
    out["Overall"] = overall.loc[tab.index]
    out.round(4).to_csv(os.path.join(OUT, "region_auc_with_published.csv"))


def figS1_replication(df):
    pub = pd.read_csv(REF, comment="#").set_index("Model")
    ours = pivot(df, "AUC").groupby(level="Region").mean().T.reindex(columns=pub.columns)
    shared = [m for m in ours.index if m in pub.index]
    if not shared:
        return
    fig, ax = plt.subplots(figsize=(5.8, 5.6))
    lim = [0.5, 0.85]
    ax.plot(lim, lim, color="#9b9a95", lw=1)
    rows = []
    for m in shared:
        x, y = pub.loc[m, pub.columns], ours.loc[m, pub.columns]
        ax.scatter(x, y, s=28, color=col(m), label=m, alpha=0.85, edgecolors="white", linewidths=0.6)
        rows.append({"Model": m, "MAD": float(np.mean(np.abs(x - y))), "r": float(np.corrcoef(x, y)[0, 1])})
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Published mean AUC$_{ROC}$ per region (Valavi et al. 2022)")
    ax.set_ylabel("This study (re-fitted, same protocol)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    t = pd.DataFrame(rows)
    ax.set_title(f"Replication check: {len(shared)} models × 6 regions; mean |Δ| = {t.MAD.mean():.3f}",
                 fontsize=9.5, loc="left")
    t.round(4).to_csv(os.path.join(OUT, "replication_check.csv"), index=False)
    save(fig, "FigS1_replication_check")


def summary(df, order):
    rows = []
    for m in order:
        r = {"Model": m}
        for met in ["AUC", "PRG", "COR", "TSS"]:
            r[met] = pivot(df, met)[m].mean()
            r[f"rank_{met}"] = pivot(df, met)[order].rank(axis=1, ascending=False).mean()[m]
        r["Time_s"] = df[df.Model == m].Time.mean()
        rows.append(r)
    t = pd.DataFrame(rows).sort_values("AUC", ascending=False).round(4)
    t.to_csv(os.path.join(OUT, "summary_v40.csv"), index=False)
    print(t.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-coverage", type=float, default=0.95)
    args = ap.parse_args()
    style()
    os.makedirs(OUT, exist_ok=True)
    df = load(args.min_coverage)
    order = model_order(df)
    print("Figures:")
    fig2_circular(df, order)
    auc, cor, _ = fig3_auc_cor(df, order)
    fig4_posthoc(df, order, auc, cor)
    fig7_ranks(df, order)
    fig8_topk(df, order)
    fig9_low_occ(df, order)
    fig10_region(df, order)
    figS1_replication(df)
    summary(df, order)


if __name__ == "__main__":
    main()
