# Re-run only MaxNet for species where it failed, merging into existing results / predictions.
import os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd
import benchmark_v40_valavi as B
from valavi_config import HERE
OUT = os.path.join(HERE, "results", "v40", "results_r.csv")

def run(region, sp):
    d = [x for x in B.load_region(region) if x["species"] == sp][0]
    jd = os.path.join(HERE, "results", "v40", "rjobs_maxnet", f"{region}_{sp}")
    os.makedirs(jd, exist_ok=True)
    tr = pd.concat([d["pres"], d["bg"]], ignore_index=True)
    tr.insert(0, "occ", np.r_[np.ones(len(d["pres"]), int), np.zeros(len(d["bg"]), int)])
    tr.to_csv(f"{jd}/train.csv", index=False); d["test"].to_csv(f"{jd}/test.csv", index=False)
    r = subprocess.run(["Rscript", os.path.join(HERE, "R", "valavi_models.R"), jd, region, "MaxNet"],
                       capture_output=True, text=True, env={**os.environ, "OMP_NUM_THREADS": "1"})
    preds = pd.read_csv(f"{jd}/preds.csv") if os.path.exists(f"{jd}/preds.csv") else pd.DataFrame()
    t = pd.read_csv(f"{jd}/times.csv").set_index("model").time
    p = preds["MaxNet"].values if "MaxNet" in preds else None
    row = {"Model": "MaxNet", **B.calc_metrics(d["y_test"], p), "Time": float(t.get("MaxNet", np.nan)),
           "Error": None if p is not None else r.stderr.strip()[-300:], "Region": region, "Species": sp,
           "Group": d["group"], "n_pres": len(d["pres"]), "n_test": len(d["y_test"]), "prevalence": float(d["y_test"].mean())}
    if p is not None:
        f = os.path.join(B.PRED_DIR, f"{region}_{sp}_r.npz")
        z = dict(np.load(f)); z["MaxNet"] = p; np.savez_compressed(f, **z)
    print(region, sp, row["AUC"], (r.stderr.strip().splitlines() or [""])[-3:], flush=True)
    return row

todo = [tuple(a.split("_")) for a in sys.argv[1:]]
with ThreadPoolExecutor(len(todo)) as ex:
    rows = list(ex.map(lambda a: run(*a), todo))
pd.DataFrame(rows).to_csv(OUT, mode="a", header=False, index=False)
