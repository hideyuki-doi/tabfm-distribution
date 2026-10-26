# Supplementary figure: change in mean AUC (vs TabPFN 3.5 without fine-tuning) by fine-tuning
# strategy and held-out region (Fast, 5 bags; leave-one-region-out).
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from valavi_config import HERE

F = os.path.join(HERE, "results", "v40", "finetune")
d = pd.concat([pd.read_csv(os.path.join(F, f)) for f in
               ["finetune_pa_results.csv", "finetune_vs_results.csv", "finetune_mix_results.csv",
                "finetune_mixtg_results.csv", "finetune_pa_nested_results.csv"]])
d["arm"] = d["arm"].str.replace(r"pa_nested_e\d+", "pa_nested", regex=True)
p = d.pivot_table(index=["holdout", "Species"], columns="arm", values="AUC")
arms = [("pb_finetuned", "PB (main)", "#eb6834"), ("vs_finetuned", "Virtual species", "#2a78d6"),
        ("mix_finetuned", "Mixed (VS + PB)", "#1baf7a"), ("mixtg_finetuned", "Mixed + survey bias", "#4a3aa7"),
        ("pa_finetuned", "PA, 10 epochs", "#9b9a95"), ("pa_nested", "PA, nested", "#c9c8c3")]
regions = ["AWT", "CAN", "NSW", "NZ", "SA", "SWI", "All"]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(10, 4.8))
w = 0.12
for k, (a, lab, c) in enumerate(arms):
    vals = [(p.loc[r, a] - p.loc[r, "default"]).mean() for r in regions[:-1]] + [(p[a] - p["default"]).mean()]
    x = np.arange(len(regions)) + (k - (len(arms) - 1) / 2) * w
    ax.bar(x, vals, width=w * 0.9, color=c, label=lab)
ax.axhline(0, color="#52514e", lw=1)
ax.axvline(len(regions) - 1.5, color="#c9c8c3", lw=1, ls="--")
ax.set_xticks(range(len(regions)), regions)
ax.set_ylabel("Change in mean AUC vs TabPFN 3.5\n(no fine-tuning)")
ax.grid(axis="y", color="#e4e3df")
ax.set_axisbelow(True)
ax.legend(frameon=False, ncol=6, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.1))
# title in the figure legend
fig.tight_layout()
os.makedirs(os.path.join(HERE, "figures", "v40"), exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(HERE, "figures", "v40", f"FigS2_finetuning_strategies.{ext}"), dpi=250, bbox_inches="tight")
print("saved")
