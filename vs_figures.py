# Figures and statistics for the virtual-species benchmarks:
#   results/v60 (additive species-environment relationships) and results/v61 (complex: interactions,
#   correlated tolerances and joint thresholds).
# Main figure: rows = design, columns = Spearman with truth, AUC, response-curve recovery.
# SI figure: example response curves (additive without bias, additive with bias, complex).
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import wilcoxon

import make_main_figs as M
import plot_v40 as P
from valavi_config import HERE

DIRS = {"additive": os.path.join(HERE, "results", "v60"), "complex": os.path.join(HERE, "results", "v61")}
SHOW = ["TabPFN3.5ft", "TabPFN3.5", "MaxEnt", "MaxNet", "BRT", "RFdown", "GAM", "Ensemble"]
LAB = {"TabPFN3.5ft": "TabPFN 3.5-FT", "TabPFN3.5": "TabPFN 3.5", "RFdown": "RF down-sampled"}
COL = {"TabPFN3.5ft": M.C_FT, "TabPFN3.5": M.C_TP, "Ensemble": M.C_ENS, "MaxEnt": "#2a78d6", "MaxNet": "#7fb3e8",
       "BRT": "#1baf7a", "RFdown": "#4a3aa7", "GAM": "#8a8984"}


def summary(df, out):
    sizes = sorted(df.n.unique())
    rows, tests = [], []
    for met in ["AUC", "spearman", "curve_r"]:
        piv = df.pivot_table(index=["sid", "n"], columns="Model", values=met)
        for n in sizes:
            p = piv.xs(n, level="n")
            for m in p.columns:
                rows.append({"metric": met, "n": n, "Model": m, "mean": p[m].mean(), "se": p[m].sem()})
        for n in sizes + ["all"]:
            p = piv if n == "all" else piv.xs(n, level="n")
            for m in [c for c in p.columns if c not in ("TabPFN3.5ft", "Truth")]:
                d = (p["TabPFN3.5ft"] - p[m]).dropna()
                if n == "all":  # the four sample sizes of a species are not independent: test species means
                    d = d.groupby(level="sid").mean()
                tests.append({"metric": met, "n": n, "vs": m, "diff": d.mean(), "wins": int((d > 0).sum()),
                              "N": len(d), "P": wilcoxon(d).pvalue if (d != 0).any() else 1.0})
    s, t = pd.DataFrame(rows), pd.DataFrame(tests)
    s.to_csv(os.path.join(out, "summary.csv"), index=False)
    t.to_csv(os.path.join(out, "tests.csv"), index=False)
    return s, t


def figure_main():
    P.style()
    fig = plt.figure(figsize=(12, 8.6))
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.3)
    titles = {"spearman": "Correlation with true probability\n(Spearman)", "AUC": "AUC against true\npresence and absence",
              "curve_r": "Response-curve recovery\n(Pearson r with true response)"}
    for row, fam in enumerate(["additive", "complex"]):
        df = pd.read_csv(os.path.join(DIRS[fam], "results.csv"))
        df = df[~df.Model.str.contains("bg5")]
        s, _ = summary(df, DIRS[fam])
        for k, met in enumerate(["spearman", "AUC", "curve_r"]):
            ax = fig.add_subplot(gs[row, k])
            for m in SHOW[::-1]:
                q = s[(s.metric == met) & (s.Model == m)].sort_values("n")
                ax.errorbar(q.n, q["mean"], yerr=q.se, color=COL[m], marker="o", ms=4.5 if "TabPFN" in m else 3.5,
                            lw=2.2 if "TabPFN" in m else 1.2, capsize=2, label=LAB.get(m, m))
            if met == "AUC":
                q = s[(s.metric == met) & (s.Model == "Truth")].sort_values("n")
                ax.plot(q.n, q["mean"], color=P.INK2, ls="--", lw=1, label="True probability")
            ax.set_xscale("log")
            ax.set_xticks([10, 30, 100, 300], ["10", "30", "100", "300"])
            ax.set_xlabel("Number of presence records")
            ax.set_title(titles[met], fontsize=10)
            M.tag(ax, "ABCDEF"[row * 3 + k], -0.12)
            if k == 0:
                ax.set_ylabel("Additive relationships" if fam == "additive" else "Complex relationships",
                              fontsize=10.5, fontweight="bold")
            if row == 0 and k == 2:
                ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, "figures", "main", f"Fig_virtual_species.{ext}"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("  - figures/main/Fig_virtual_species.png")


def curves(ax_row, fam, sid, n, title):
    out = DIRS[fam]
    meta = {m["sid"]: m for m in json.load(open(os.path.join(out, "meta.json")))}
    m = meta[sid]
    d = np.load(os.path.join(out, "data", f"{sid}.npz"))
    preds = {}
    for f in ["r", "gpu"]:
        z = np.load(os.path.join(out, "preds", f"{sid}_n{n}_{f}.npz"))
        preds.update({k_: z[k_] for k_ in z.files})
    nt = len(d["te"])
    cov = pd.read_csv(os.path.join(HERE, "data", "background50k", f"{m['region']}bg50k.csv"))
    for j, ax in enumerate(ax_row):
        if j >= len(m["niche"]):
            ax.axis("off")
            continue
        v = m["niche"][j]
        grid = np.quantile(cov[v].values, np.linspace(0.02, 0.98, 12))
        tr = np.array(m["truth_pd"][v])
        ax.plot(grid, tr - tr.mean(), color=P.INK, lw=3, label="Truth")
        for mm in ["TabPFN3.5ft", "MaxEnt", "BRT", "GAM"]:
            p = np.clip(preds[mm][nt:].astype(float), 1e-6, 1 - 1e-6)
            sel = d["pd_var"] == v
            lg = pd.Series(logit(p[sel])).groupby(d["pd_grid"][sel]).mean().sort_index().values
            ax.plot(grid, lg - lg.mean(), color=COL[mm], lw=1.8, label=LAB.get(mm, mm))
        ax.set_xlabel(v)
        if j == 0:
            ax.set_ylabel("Partial effect (logit, centred)")
            ax.set_title(title, loc="left", fontsize=9.5)
    return m


def pick(fam, biased, n=30):
    out = DIRS[fam]
    meta = {m["sid"]: m for m in json.load(open(os.path.join(out, "meta.json")))}
    df = pd.read_csv(os.path.join(out, "results.csv"))
    sub = df[(df.n == n) & (df.Model == "TabPFN3.5ft")].copy()
    sub = sub[sub.sid.map(lambda x: meta[x]["biased"] == biased and len(meta[x]["niche"]) >= 2)]
    med = sub.curve_r.median()
    return sub.iloc[(sub.curve_r - med).abs().argsort().iloc[0]].sid


def figure_examples():
    P.style()
    fig, axes = plt.subplots(3, 3, figsize=(12, 11))
    plan = [("additive", False, "Additive relationship, no sampling bias"),
            ("additive", True, "Additive relationship, environmental sampling bias"),
            ("complex", False, "Complex relationship, no sampling bias")]
    for i, (fam, b, t) in enumerate(plan):
        sid = pick(fam, b)
        curves(axes[i], fam, sid, 30, f"{t} ({sid}, 30 records)")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, "figures", "supp", f"FigS_vs_examples.{ext}"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("  - figures/supp/FigS_vs_examples.png")


if __name__ == "__main__":
    for sub in ("main", "supp"):
        os.makedirs(os.path.join(HERE, "figures", sub), exist_ok=True)
    figure_main()
    figure_examples()
    for fam, out in DIRS.items():
        t = pd.read_csv(os.path.join(out, "tests.csv"))
        print("==", fam)
        print(t[t.n == "all"].round(4).to_string(index=False))
