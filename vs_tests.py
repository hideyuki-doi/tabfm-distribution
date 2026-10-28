# Species-level tests for the virtual-species benchmarks (results/v60 additive, results/v61 complex).
# The four sample sizes of one species are not independent, so comparisons pooled over sample sizes
# average the paired difference within each species first and then test across species (paired
# Wilcoxon signed-rank test). Comparisons at a single sample size have one dataset per species.
# Output: results/v6x/tests_species.csv and the statistics quoted in the manuscript.
import json
import os

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import wilcoxon

from valavi_config import HERE

DIRS = {"additive": os.path.join(HERE, "results", "v60"), "complex": os.path.join(HERE, "results", "v61")}
FT, TP = "TabPFN3.5ft", "TabPFN3.5"
COMP = ["MaxEnt", "MaxNet", "Lasso", "GAM", "BRT", "RFdown", "RF", "Ensemble", TP]


def species_test(df, a, b, metric, ns=(10, 30, 100, 300)):
    """Mean over the given sample sizes of the paired difference a - b, per species, then Wilcoxon."""
    piv = df[df.n.isin(ns)].pivot_table(index=["sid", "n"], columns="Model", values=metric)
    d = (piv[a] - piv[b]).groupby(level="sid").mean().dropna()
    return {"a": a, "b": b, "metric": metric, "n": "+".join(map(str, ns)), "diff": d.mean(),
            "species_a_better": int((d > 0).sum()), "species": len(d), "P": wilcoxon(d).pvalue}


def load(fam):
    df = pd.read_csv(os.path.join(DIRS[fam], "results.csv"))
    return df[~df.Model.str.contains("bg5")]


def main():
    out = {}
    for fam, path in DIRS.items():
        df = load(fam)
        rows = []
        for subset, sub in [("all", df), ("unbiased", df[~df.biased]), ("biased", df[df.biased])]:
            for metric in ["spearman", "AUC", "curve_r"]:
                for a in [FT, TP]:
                    for b in COMP:
                        if a == b:
                            continue
                        rows.append({"design": fam, "subset": subset, **species_test(sub, a, b, metric)})
                        for n in (10, 30, 100, 300):
                            rows.append({"design": fam, "subset": subset, **species_test(sub, a, b, metric, (n,))})
        t = pd.DataFrame(rows)
        t.to_csv(os.path.join(path, "tests_species.csv"), index=False)
        out[fam] = t

    def get(fam, subset, a, b, metric="spearman", n="10+30+100+300"):
        t = out[fam]
        return t[(t.subset == subset) & (t.a == a) & (t.b == b) & (t.metric == metric) & (t.n == n)].iloc[0]

    def show(label, r):
        print(f"{label}: diff {r['diff']:+.4f}, a better for {r['species_a_better']}/{r['species']} species, P = {r['P']:.2g}")

    print("== additive design, all 60 species (Spearman, pooled over n within species)")
    for b in ["MaxEnt", "Lasso", "GAM", "Ensemble", "BRT", "RFdown"]:
        show(f"   FT - {b}", get("additive", "all", FT, b))
    print("== complex design, 30 unbiased species: untuned TabPFN 3.5 vs")
    for b in ["MaxNet", "MaxEnt", "Ensemble", "Lasso", "GAM", "BRT"]:
        show(f"   TP - {b}", get("complex", "unbiased", TP, b))
    print("== complex design, fine-tuning effect (FT - TP), 60 species, by n")
    for n in ["10", "30", "100", "300", "100+300"]:
        r = species_test(load("complex"), FT, TP, "spearman", tuple(int(x) for x in n.split("+")))
        show(f"   n = {n}", r)
    print("== biased species (30 per design): foundation models vs MaxEnt")
    for fam in DIRS:
        for a in [FT, TP]:
            show(f"   {fam} {a} - MaxEnt", get(fam, "biased", a, "MaxEnt"))
    print("== complex design, all 60 species: FT vs Lasso / GAM (AUC, curve_r)")
    for b in ["Lasso", "GAM"]:
        for metric in ["AUC", "curve_r"]:
            show(f"   FT - {b} {metric}", get("complex", "all", FT, b, metric))

    # bias and covariate correlation, additive design, species level
    fam = "additive"
    meta = {m["sid"]: m for m in json.load(open(os.path.join(DIRS[fam], "meta.json")))}
    bg = {}
    corr = {}
    for sid, m in meta.items():
        if not m["biased"]:
            continue
        r = m["region"]
        if r not in bg:
            bg[r] = pd.read_csv(os.path.join(HERE, "data", "background50k", f"{r}bg50k.csv"))
        corr[sid] = max(abs(np.corrcoef(bg[r][m["bias_var"]], bg[r][v])[0, 1]) for v in m["niche"])
    df = load(fam)
    print("== bias correlation (30 biased additive species; species means over n)")
    for metric in ["curve_r", "spearman"]:
        piv = df[df.biased].pivot_table(index=["sid", "n"], columns="Model", values=metric)
        d = (piv[FT] - piv["MaxEnt"]).groupby(level="sid").mean().rename("d").to_frame()
        d["corr"] = pd.Series(corr)
        f = smf.ols("d ~ corr", d).fit()
        print(f"   {metric}: slope {f.params['corr']:+.3f}, P = {f.pvalues['corr']:.2g} (n = {len(d)} species)")


if __name__ == "__main__":
    main()
