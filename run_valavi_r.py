# Run Valavi et al. (2022) reference SDMs in R (R/valavi_models.R), species in parallel.
# Same data and evaluation as benchmark_v40_valavi.py; results -> results/v40/results_r.csv
# Usage: python run_valavi_r.py [--workers 12] [--regions AWT,...] [--max-species N]
import argparse
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

import benchmark_v40_valavi as B
from valavi_config import HERE, REGIONS

R_MODELS = ["GAM", "Lasso", "BRT", "RF", "RFdown", "MaxNet", "MaxEnt", "Ensemble"]
NAMES = {"RFdown": "RF down-sampled"}
JOB_DIR = os.path.join(HERE, "results", "v40", "rjobs")
OUT = os.path.join(HERE, "results", "v40", "results_r.csv")


def run_species(d, models):
    jd = os.path.join(JOB_DIR, f"{d['region']}_{d['species']}")
    os.makedirs(jd, exist_ok=True)
    train = pd.concat([d["pres"], d["bg"]], ignore_index=True)
    train.insert(0, "occ", np.r_[np.ones(len(d["pres"]), int), np.zeros(len(d["bg"]), int)])
    train.to_csv(os.path.join(jd, "train.csv"), index=False)
    d["test"].to_csv(os.path.join(jd, "test.csv"), index=False)
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    t0 = time.time()
    r = subprocess.run(["Rscript", os.path.join(HERE, "R", "valavi_models.R"), jd, d["region"], ",".join(models)],
                       capture_output=True, text=True, env=env)
    preds = pd.read_csv(os.path.join(jd, "preds.csv")) if os.path.exists(os.path.join(jd, "preds.csv")) else pd.DataFrame()
    times = pd.read_csv(os.path.join(jd, "times.csv")).set_index("model").time if os.path.exists(
        os.path.join(jd, "times.csv")) else pd.Series(dtype=float)
    rows, keep = [], {}
    for m in models:
        p = preds[m].values if m in preds.columns else None
        err = None if p is not None else (r.stderr.strip().splitlines() or ["missing"])[-1][:300]
        rows.append({"Model": NAMES.get(m, m), **B.calc_metrics(d["y_test"], p),
                     "Time": float(times.get(m, np.nan)), "Error": err})
        if p is not None:
            keep[NAMES.get(m, m)] = p  # full precision (see benchmark_v40_valavi.run_models)
    np.savez_compressed(os.path.join(B.PRED_DIR, f"{d['region']}_{d['species']}_r.npz"),
                        y_test=d["y_test"].astype(np.int8), **keep)
    for r_ in rows:
        r_.update({"Region": d["region"], "Species": d["species"], "Group": d["group"], "n_pres": len(d["pres"]),
                   "n_test": len(d["y_test"]), "prevalence": float(d["y_test"].mean())})
    if not r.returncode and len(keep) == len(models):
        shutil.rmtree(jd, ignore_errors=True)
    return rows, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--regions", default=",".join(REGIONS))
    ap.add_argument("--models", default=",".join(R_MODELS))
    ap.add_argument("--max-species", type=int, default=0)
    ap.add_argument("--keep-failed", action="store_true",
                    help="do not re-run species with a failed model (by default they are re-run; new rows supersede old ones)")
    args = ap.parse_args()
    os.makedirs(B.PRED_DIR, exist_ok=True)
    models = args.models.split(",")

    done = set()
    if os.path.exists(OUT):
        prev = pd.read_csv(OUT).drop_duplicates(["Region", "Species", "Model"], keep="last")
        done = set(zip(prev.Region, prev.Species))
        if not args.keep_failed:
            bad = prev[prev.AUC.isna()]
            done -= set(zip(bad.Region, bad.Species))
    jobs = []
    for region in args.regions.split(","):
        sp = B.load_region(region)
        sp = sp[: args.max_species] if args.max_species else sp
        jobs += [d for d in sp if (region, d["species"]) not in done]
    # largest jobs first (SWI / NZ species with many presences or test sites)
    jobs.sort(key=lambda d: -(len(d["pres"]) + len(d["y_test"]) / 10))
    print(f"R reference models: {len(jobs)} species, {args.workers} workers, models={models}", flush=True)

    t_start = time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(run_species, d, models): d for d in jobs}
        for k, f in enumerate(as_completed(futs), 1):
            d = futs[f]
            try:
                rows, dt = f.result()
            except Exception as e:
                print(f"[{d['region']} {d['species']}] FAILED {e!r}", flush=True)
                continue
            pd.DataFrame(rows).to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)
            summ = " ".join(f"{r['Model']}={r['AUC']:.3f}" for r in rows)
            print(f"[{k}/{len(jobs)} {d['region']} {d['species']} n_p={len(d['pres'])}] {dt:.0f}s | {summ}", flush=True)
    print(f"=== DONE in {(time.time() - t_start) / 60:.1f} min ===")


if __name__ == "__main__":
    main()
