# Main-text figures (Figs 1, 2, 4) and several SI figures, built from the v40 (independent
# presence-absence evaluation, 225 species) and v50 (spatial-block CV, 178 species) results.
# Fast variants and Ensemble+ combinations are left to the supplement.
import os

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import plot_v40 as P
import valavi_stats as VS
from valavi_config import HERE

OUT = os.path.join(HERE, "figures", "main")
FT, TP = "TabPFN3.5ft_Stable", "TabPFN3.5_Stable"
SINGLE = ["GAM", "Lasso", "MaxNet", "MaxEnt", "BRT", "RF", "RF down-sampled", "XGBoost",
          "TabICL", "TabPFN2.5_Stable", TP, FT]
MODELS = SINGLE + ["Ensemble"]
LAB = {TP: "TabPFN 3.5", FT: "TabPFN 3.5-FT", "TabPFN2.5_Stable": "TabPFN 2.5",
       "TabPFN3.5_Fast": "TabPFN 3.5", "TabPFN3.5ft_Fast": "TabPFN 3.5-FT",
       "TabPFN3.5": "TabPFN 3.5", "TabPFN3.5ft": "TabPFN 3.5-FT"}
C_FT, C_TP, C_REF, C_ENS = "#d1495b", "#edae49", "#5b6770", "#9aa5b1"


def lab(m):
    return LAB.get(m, m)


def colour(m):
    return {FT: C_FT, TP: C_TP, "TabPFN3.5ft_Fast": C_FT, "TabPFN3.5_Fast": C_TP, "TabPFN3.5ft": C_FT, "TabPFN3.5": C_TP,
            "Ensemble": C_ENS}.get(m, C_REF)


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  - figures/main/{name}.png")


def tag(ax, s, x=-0.02):
    ax.text(x, 1.02, s, transform=ax.transAxes, fontsize=13, fontweight="bold", ha="right", va="bottom")


def circular(df, models, name, out_dir=None):
    """Per-species AUC as concentric rings (one ring per model, 1 = innermost = best mean AUC),
    species as spokes grouped by region. Ring numbers sit in the gap between NSW and NZ and the
    model key is placed outside the circle."""
    from matplotlib.colors import BoundaryNorm
    L = P.lab if any(m.endswith("_Fast") for m in models) else lab
    auc = P.pivot(df, "AUC")[models]
    order = auc.mean().sort_values(ascending=False).index.tolist()
    auc = auc[order]
    npres = df.drop_duplicates(["Region", "Species"]).set_index(["Region", "Species"]).n_pres
    regions = ["NZ", "SA", "SWI", "AWT", "CAN", "NSW"]
    gap, key_gap = 3, 9
    cols = []
    for r in regions:
        sub = auc.loc[r]
        cols += [(r, sp) for sp in sub.mean(axis=1).sort_values(ascending=False).index]
        cols += [None] * (key_gap if r == regions[-1] else gap)
    n_col, n_ring = len(cols), len(order)
    r0 = 3.0
    cmap = plt.get_cmap("Spectral", 10)
    norm = BoundaryNorm(np.linspace(0, 1, 11), 10)
    fig = plt.figure(figsize=(12.5, 9.2))
    ax = fig.add_axes([0.0, 0.06, 0.66, 0.9], projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    # rotate so that the key gap is centred at the top
    width = 2 * np.pi / n_col
    ax.set_theta_offset(np.pi / 2 + (key_gap / 2) * width)
    for j, c in enumerate(cols):
        if c is None:
            continue
        th = j * width
        vals = auc.loc[c].values
        ax.bar(np.full(n_ring, th), np.ones(n_ring), width=width, bottom=r0 + np.arange(n_ring),
               color=cmap(norm(vals)), edgecolor="white", linewidth=0.15, align="edge")
        npr = npres.loc[c]
        colr = "#f6c89f" if npr < 100 else ("#ef6f6c" if npr <= 1000 else "#6b1d1d")
        ax.bar(th, np.log10(npr) * 0.9, width=width, bottom=r0 + n_ring + 0.4, color=colr,
               edgecolor="white", linewidth=0.15, align="edge")
    j = 0
    for r in regions:
        n = auc.loc[r].shape[0]
        mid = (j + n / 2) * width
        ax.text(mid, r0 + n_ring + 4.4, r, ha="center", va="center", fontsize=13, fontweight="bold")
        j += n + gap
    th_key = (n_col - key_gap / 2) * width
    for i in range(n_ring):
        ax.text(th_key, r0 + i + 0.5, str(i + 1), ha="center", va="center", fontsize=8.5, color=P.INK,
                fontweight="bold" if "TabPFN3.5" in order[i] else "normal")
    ax.set_ylim(0, r0 + n_ring + 5.2)
    ax.axis("off")
    # key outside the circle
    kx = fig.add_axes([0.67, 0.25, 0.32, 0.6])
    kx.axis("off")
    means = auc.mean()
    for i, m in enumerate(order):
        y = 1 - i / max(n_ring, 12)
        kx.text(0.0, y, f"{i + 1}", ha="right", va="center", fontsize=10, fontweight="bold")
        kx.text(0.05, y, L(m), ha="left", va="center", fontsize=10,
                fontweight="bold" if "TabPFN3.5" in m else "normal")
        kx.text(0.85, y, f"{means[m]:.3f}", ha="right", va="center", fontsize=10, color=P.INK2)
    kx.text(0.0, 1 + 1.3 / max(n_ring, 12), "Ring  Model", ha="right", va="center", fontsize=9, color=P.INK2)
    kx.text(0.85, 1 + 1.3 / max(n_ring, 12), "mean AUC", ha="right", va="center", fontsize=9, color=P.INK2)
    kx.text(0.05, 1 + 1.3 / max(n_ring, 12), "", fontsize=9)
    kx.set_ylim(-0.1, 1.12)
    kx.set_xlim(-0.1, 1)
    cax = fig.add_axes([0.70, 0.14, 0.24, 0.022])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal",
                      ticks=[0, 0.25, 0.5, 0.75, 1])
    cb.set_label("AUC$_{ROC}$ (independent survey data)")
    fig.text(0.70, 0.06, "Outer bars: log$_{10}$(number of presence records)\nlight < 100; pink 100–1,000; dark > 1,000",
             fontsize=8.5, color=P.INK2)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(out_dir, f"{name}.{ext}"), bbox_inches="tight", dpi=300)
        plt.close(fig)
        print(f"  - {out_dir}/{name}.png")
    else:
        save(fig, name)


def rank_axes(df, models, axes, tags):
    from adjustText import adjust_text
    rk = {m: P.pivot(df, m)[models].rank(axis=1, ascending=False).mean() for m in ["AUC", "COR", "PRG"]}
    for ax, ym, tg in zip(axes, ["COR", "PRG"], tags):
        x, y = rk["AUC"], rk[ym]
        texts = []
        for m in models:
            ax.scatter(x[m], y[m], s=46 if "TabPFN3.5" in m else 34, color=colour(m), zorder=3,
                       edgecolor="white", linewidth=0.5)
            texts.append(ax.text(x[m], y[m], lab(m), fontsize=8.5, color=P.INK,
                                 fontweight="bold" if "TabPFN3.5" in m else "normal"))
        mid = (len(models) + 1) / 2
        ax.axvline(mid, ls="--", color="#bdbcb7", lw=1)
        ax.axhline(mid, ls="--", color="#bdbcb7", lw=1)
        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.set_xlabel("Average rank, AUC$_{ROC}$ (1 = best)")
        ax.set_ylabel("Average rank, " + ("COR" if ym == "COR" else "AUC$_{PRG}$"))
        adjust_text(texts, ax=ax, x=x[models].values, y=y[models].values, expand=(1.4, 1.9), force_text=(0.4, 0.8),
                    arrowprops=dict(arrowstyle="-", color="#bdbcb7", lw=0.6))
        tag(ax, tg, -0.12)


def fig1(df, auc):
    est, se = auc[MODELS].mean(), auc[MODELS].sem()
    order = est.sort_values().index.tolist()
    shaffer = VS.posthoc_aligned_ranks(auc[MODELS])[FT]
    fig = plt.figure(figsize=(11, 10))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.15], height_ratios=[1, 1.05], hspace=0.32, wspace=0.55)
    a, b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    axc, axd = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    y = np.arange(len(order))
    for i, m in enumerate(order):
        a.errorbar(est[m], i, xerr=se[m], fmt="o", color=colour(m), ms=7 if "TabPFN3.5" in m else 5.5,
                   capsize=2, mfc="white" if m == "Ensemble" else colour(m))
    a.set_yticks(y, [lab(m) + (" (reference)" if m == "Ensemble" else "") for m in order])
    a.set_xlabel("Mean AUC$_{ROC}$ (± SE across 225 species)")
    tag(a, "A", -0.33)
    # B: paired differences TabPFN 3.5-FT minus each model
    others = [m for m in order if m != FT][::-1]
    others = others[::-1]
    for i, m in enumerate(others):
        d = auc[FT] - auc[m]
        ci = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
        b.errorbar(d.mean(), i, xerr=ci, fmt="o", color=colour(m), ms=6, capsize=2,
                   mfc="white" if m == "Ensemble" else colour(m))
        p = shaffer[m]
        star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        b.text(1.01, i, f"{int((d > 0).sum())}/225  {star}", transform=b.get_yaxis_transform(),
               va="center", fontsize=8.5, color=P.INK2)
    b.axvline(0, color=P.INK2, lw=1)
    b.set_yticks(np.arange(len(others)), [lab(m) for m in others])
    b.set_xlabel("AUC difference, TabPFN 3.5-FT minus model (mean, 95% CI)")
    b.text(1.01, len(others) - 0.3, "species won\n& Shaffer P", transform=b.get_yaxis_transform(),
           fontsize=8, color=P.INK2, va="bottom")
    tag(b, "B", -0.3)
    rank_axes(df, SINGLE, [axc, axd], ["C", "D"])
    save(fig, "Fig2_overall_and_ranks")


def fig2(df, auc):
    best = auc[SINGLE].eq(auc[SINGLE].max(axis=1), axis=0)
    top = best.div(best.sum(axis=1), axis=0).mean() * 100  # ties share the credit
    top = top.sort_values()
    n = df.drop_duplicates(["Region", "Species"]).set_index(["Region", "Species"]).n_pres.reindex(auc.index)
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1, 1.2]})
    a.barh(np.arange(len(top)), top.values, color=[colour(m) for m in top.index], height=0.65)
    a.set_yticks(np.arange(len(top)), [lab(m) for m in top.index])
    for i, v in enumerate(top.values):
        a.text(v + 0.3, i, f"{v:.1f}", va="center", fontsize=8.5)
    a.set_xlabel("Species for which the model is best (%)\namong single models")
    tag(a, "A", -0.3)
    REGC = {"AWT": "#1b9e77", "CAN": "#d95f02", "NSW": "#7570b3", "NZ": "#e7298a", "SA": "#66a61e", "SWI": "#e6ab02"}
    d = auc[FT] - auc["MaxEnt"]
    lp = np.log10(n.values.astype(float))
    for r, c in REGC.items():
        sel = d.index.get_level_values(0) == r
        b.scatter(n.values[sel], d.values[sel], s=16, color=c, alpha=0.8, label=r, edgecolor="none")
    import statsmodels.formula.api as smf
    dd = pd.DataFrame({"d": d.values, "lp": lp, "region": d.index.get_level_values(0)})
    f = smf.ols("d ~ lp + C(region)", dd).fit()
    xs = np.linspace(lp.min(), lp.max(), 50)
    b.plot(10 ** xs, d.mean() + f.params["lp"] * (xs - lp.mean()), color=P.INK, lw=1.4)
    b.axhline(0, color=P.INK2, lw=0.8, ls="--")
    b.set_xscale("log")
    b.set_xlabel("Number of presence records")
    b.set_ylabel("AUC difference, TabPFN 3.5-FT minus MaxEnt")
    b.text(0.98, 0.04, f"slope per log$_{{10}}$ unit = {f.params['lp']:+.4f} (P = {f.pvalues['lp']:.2f})",
           transform=b.transAxes, ha="right", fontsize=8.5, color=P.INK2)
    b.legend(frameon=False, fontsize=8, ncol=6, loc="lower center", bbox_to_anchor=(0.5, 1.0), handletextpad=0.2,
             columnspacing=0.8)
    tag(b, "B", -0.12)
    b.text(-0.12, 1.02, "", transform=b.transAxes)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, "figures", "supp", f"FigS10_species_level.{ext}"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("  - figures/supp/FigS10_species_level.png")


def fig3(auc):
    pub = pd.read_csv(P.REF, comment="#").set_index("Model")
    ours = auc[[TP, FT]].groupby(level="Region").mean().T.reindex(columns=pub.columns)
    ours.index = [lab(m) + " †" for m in ours.index]
    tab = pd.concat([pub, ours])
    w = pd.Series(P.N_SP_REGION)[tab.columns]
    tab["Overall"] = (tab[w.index] * w).sum(axis=1) / w.sum()
    tab = tab.sort_values("Overall")
    z = tab.apply(lambda c: (c - c.mean()) / c.std())
    fig, ax = plt.subplots(figsize=(6.4, 7.6))
    ax.imshow(z.values, cmap="viridis", aspect="auto")
    for i in range(tab.shape[0]):
        for j in range(tab.shape[1]):
            best = tab.iloc[i, j] == tab.iloc[:, j].max()
            ax.text(j, i, f"{tab.iloc[i, j]:.3f}".lstrip("0"), ha="center", va="center", fontsize=7.5,
                    color="black" if z.iloc[i, j] > 0.3 else "white", fontweight="bold" if best else "normal")
    ax.set_xticks(range(tab.shape[1]), tab.columns)
    ax.set_yticks(range(tab.shape[0]), tab.index)
    for t in ax.get_yticklabels():
        if "†" in t.get_text():
            t.set_fontweight("bold")
    ax.xaxis.tick_top()
    ax.grid(False)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    save(fig, "Fig3_regions_vs_published")


def fig5():
    """Spatial-block vs random CV (SI). The two designs are offset vertically so their error bars do not overlap."""
    d = pd.concat([pd.read_csv(os.path.join(HERE, "results", "v50", f"results_{f}.csv")) for f in ("r", "gpu")])
    d = d.drop_duplicates(["Region", "Species", "cv", "fold", "Model"], keep="last")
    sp = d.groupby(["cv", "Region", "Species", "Model"]).AUC.mean().reset_index()
    stat = sp.groupby(["Model", "cv"]).AUC.agg(["mean", "sem"]).unstack("cv")
    order = stat[("mean", "spatial")].sort_values().index
    stat = stat.loc[order]
    fig, a = plt.subplots(figsize=(6.4, 5.6))
    y = np.arange(len(order))
    dy = {"random": 0.18, "spatial": -0.18}
    for i, m in enumerate(order):
        a.plot([stat.loc[m, ("mean", "random")], stat.loc[m, ("mean", "spatial")]], [i + dy["random"], i + dy["spatial"]],
               color=P.GRID, lw=1.2, zorder=1)
    for cv, mk, nm in (("random", "o", "Random 5-fold CV"), ("spatial", "s", "Spatial-block 5-fold CV")):
        for i, m in enumerate(order):
            a.errorbar(stat.loc[m, ("mean", cv)], i + dy[cv], xerr=stat.loc[m, ("sem", cv)], fmt=mk, ms=6, capsize=2.5,
                       color=colour(m), mfc="white" if cv == "random" else colour(m), mew=1.4, zorder=2,
                       label=nm if i == 0 else None)
    a.set_yticks(y, [lab(m) for m in order])
    a.set_ylim(-0.6, len(order) - 0.4)
    for t in a.get_yticklabels():
        if "TabPFN" in t.get_text():
            t.set_fontweight("bold")
    a.set_xlabel("AUC$_{ROC}$ at survey sites in held-out folds\n(mean ± SE, 178 species)")
    leg = a.legend(frameon=False, fontsize=8.5, loc="lower right")
    for h in leg.legend_handles:
        h.set_color(C_REF)
    fig.tight_layout()
    out = os.path.join(HERE, "figures", "supp")
    os.makedirs(out, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"FigS14_spatial_cv.{ext}"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("  - figures/supp/FigS14_spatial_cv.png")


def fig5_finetuning():
    F = os.path.join(HERE, "results", "v40", "finetune")
    d = pd.concat([pd.read_csv(os.path.join(F, f)) for f in
                   ["finetune_pa_results.csv", "finetune_vs_results.csv", "finetune_mix_results.csv",
                    "finetune_mixtg_results.csv", "finetune_pa_nested_results.csv",
                    "finetune_species_split_results.csv"]])
    d["arm"] = d["arm"].str.replace(r"pa_nested_e\d+", "pa_nested", regex=True)
    p = d.drop_duplicates(["holdout", "Species", "arm"]).pivot_table(index=["holdout", "Species"], columns="arm",
                                                                     values="AUC")
    arms = [("pb_finetuned", "PB (main)", "#d1495b"), ("vs_finetuned", "Virtual species", "#2a78d6"),
            ("mix_finetuned", "Mixed (VS + PB)", "#1baf7a"), ("mixtg_finetuned", "Mixed + survey bias", "#4a3aa7"),
            ("pa_finetuned", "PA, 10 epochs", "#9b9a95"), ("pa_nested", "PA, nested", "#c9c8c3")]
    regions = ["AWT", "CAN", "NSW", "NZ", "SA", "SWI", "All"]
    fig = plt.figure(figsize=(12, 4.6))
    a = fig.add_axes([0.06, 0.2, 0.6, 0.72])
    w = 0.12
    for k, (arm, nm, c) in enumerate(arms):
        vals = [(p.loc[r, arm] - p.loc[r, "default"]).mean() for r in regions[:-1]] + [(p[arm] - p["default"]).mean()]
        a.bar(np.arange(len(regions)) + (k - (len(arms) - 1) / 2) * w, vals, width=w * 0.9, color=c, label=nm)
    a.axhline(0, color=P.INK2, lw=1)
    a.axvline(len(regions) - 1.5, color="#c9c8c3", lw=1, ls="--")
    a.set_xticks(range(len(regions)), regions)
    a.set_ylabel("Change in mean AUC vs TabPFN 3.5\n(no fine-tuning)")
    a.legend(frameon=False, ncol=3, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    tag(a, "A", -0.06)
    b = fig.add_axes([0.75, 0.2, 0.24, 0.72])
    comp = [("pb_finetuned", "PB, by region"), ("pa_finetuned", "PA, by region"),
            ("pb_species", "PB, by species"), ("pa_species", "PA, by species")]
    rng = np.random.default_rng(3)
    for i, (arm, nm) in enumerate(comp):
        x = (p[arm] - p["default"]).dropna()
        bs = [x.sample(len(x), replace=True, random_state=int(rng.integers(1 << 31))).mean() for _ in range(2000)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        col = "#d1495b" if arm.startswith("pb") else "#7d7c78"
        mk = "o" if "species" not in arm else "s"
        b.errorbar(x.mean(), i, xerr=[[x.mean() - lo], [hi - x.mean()]], fmt=mk, color=col, ms=7, capsize=3,
                   mfc=col if "species" not in arm else "white", mew=1.5)
    b.axvline(0, color=P.INK2, lw=1)
    b.axhline(1.5, color="#c9c8c3", lw=1, ls="--")
    b.set_yticks(range(len(comp)), [c[1] for c in comp])
    b.invert_yaxis()
    b.set_xlabel("Gain in mean AUC (95% CI)")
    tag(b, "B", -0.45)
    save(fig, "Fig4_finetuning_strategies")


def fig_s9_calibration():
    cal = pd.read_csv(os.path.join(HERE, "results", "v40", "review", "calibration_slope.csv"), index_col=0)
    cal = cal.sort_values("median")
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    y = np.arange(len(cal))
    ax.hlines(y, cal.q25, cal.q75, color=[colour(m) for m in cal.index], lw=3)
    ax.scatter(cal["median"], y, color=[colour(m) for m in cal.index], zorder=3, s=36)
    ax.axvline(1, color=P.INK2, ls="--", lw=1)
    ax.set_yticks(y, [lab(m) for m in cal.index])
    ax.set_xlabel("Calibration slope at survey sites (median and interquartile range)")
    fig.tight_layout()
    out = os.path.join(HERE, "figures", "supp")
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"FigS9_calibration_slope.{ext}"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("  - figures/supp/FigS9_calibration_slope.png")


def main():
    P.style()
    df = P.load(0.9)
    auc = P.pivot(df, "AUC")
    circular(df, SINGLE, "Fig1_circular_AUC")
    allm = [m for m in auc.columns if not m.startswith("Ensemble")]
    circular(df, allm, "FigS2_circular_AUC_all", out_dir=os.path.join(HERE, "figures", "supp"))
    fig1(df, auc)
    fig2(df, auc)
    fig3(auc)
    fig5()
    fig_s9_calibration()
    fig5_finetuning()


if __name__ == "__main__":
    main()
