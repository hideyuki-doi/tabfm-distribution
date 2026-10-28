# Re-save the R reference-model predictions of the given species at full precision (they were saved as
# float32, which tied some MaxNet and GAM predictions and shifted AP and AUC-PRG recomputed from the file).
# The R models are deterministic, so the refit must reproduce the reported metrics; this is checked and the
# old files are kept in results/v40/preds_float32/. results_r.csv is not changed.
# Usage: python regen_r_preds.py REGION_species ...
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

import benchmark_v40_valavi as B
import run_valavi_r as RR
from valavi_config import HERE

REF = pd.read_csv(os.path.join(HERE, "results", "v40", "results_r.csv")).drop_duplicates(
    ["Region", "Species", "Model"], keep="last").set_index(["Region", "Species", "Model"])
BACKUP = os.path.join(HERE, "results", "v40", "preds_float32")


def one(key):
    region, sp = key.split("_")
    d = next(x for x in B.load_region(region) if x["species"] == sp)
    src = os.path.join(B.PRED_DIR, f"{region}_{sp}_r.npz")
    os.makedirs(BACKUP, exist_ok=True)
    if not os.path.exists(os.path.join(BACKUP, os.path.basename(src))):
        shutil.copy2(src, BACKUP)
    rows, dt = RR.run_species(d, RR.R_MODELS)
    out = []
    for r in rows:
        ref = REF.loc[(region, sp, r["Model"])]
        out.append({"key": key, "Model": r["Model"], **{f"d{m}": abs(r[m] - ref[m]) for m in ["AUC", "PRG", "COR", "AP"]}})
    return out, dt


if __name__ == "__main__":
    keys = sys.argv[1:]
    with ThreadPoolExecutor(7) as ex:
        res = list(ex.map(one, keys))
    t = pd.DataFrame([r for rows, _ in res for r in rows])
    print(t.groupby("key")[["dAUC", "dPRG", "dCOR", "dAP"]].max().to_string())
    print("max over all:", t[["dAUC", "dPRG", "dCOR", "dAP"]].max().to_dict())
