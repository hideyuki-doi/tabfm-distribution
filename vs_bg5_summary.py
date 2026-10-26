# Sensitivity analysis: TabPFN 3.5-FT with 5x background points per bag (n = 10 and 30 only)
import pandas as pd
from scipy.stats import wilcoxon
df = pd.read_csv("results/v60/results.csv")
d = df[df.n.isin([10, 30])]
out = []
for met in ["spearman", "AUC", "curve_r"]:
    p = d.pivot_table(index=["sid", "n", "biased"], columns="Model", values=met)
    for n in [10, 30]:
        q = p.xs(n, level="n")
        x = q["TabPFN3.5ft_bg5"] - q["TabPFN3.5ft"]
        out.append({"metric": met, "n": n, "ft": q["TabPFN3.5ft"].mean(), "ft_bg5": q["TabPFN3.5ft_bg5"].mean(),
                    "MaxEnt": q["MaxEnt"].mean(), "diff": x.mean(), "P": wilcoxon(x).pvalue})
    for b in [False, True]:
        q = p.xs(b, level="biased")
        out.append({"metric": met, "n": f"biased={b}", "ft": q["TabPFN3.5ft"].mean(), "ft_bg5": q["TabPFN3.5ft_bg5"].mean(),
                    "MaxEnt": q["MaxEnt"].mean(), "diff": (q["TabPFN3.5ft_bg5"] - q["TabPFN3.5ft"]).mean(),
                    "P": wilcoxon(q["TabPFN3.5ft_bg5"] - q["TabPFN3.5ft"]).pvalue})
t = pd.DataFrame(out).round(4)
t.to_csv("results/v60/bg5_summary.csv", index=False)
print(t.to_string(index=False))
