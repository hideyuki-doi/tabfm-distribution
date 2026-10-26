# Fig. 11 of Valavi et al. (2022): predicted relative likelihood of occurrence for NZ species nz30.
# Models are fitted exactly as in the benchmark (v40) and predicted to a 1 km grid of New Zealand
# (100 m rasters resampled by nearest neighbour) and to the presence-absence test sites.
# Step 1 (this script, --fit): predictions -> results/v40/map_nz30/preds.npz
# Step 2 (--plot): figures/v40/Fig11_map_nz30.png
import argparse
import os
import subprocess

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling

import benchmark_v40_valavi as B
from valavi_config import HERE, VARS

REGION, SPECIES, AGG = "NZ", "nz30", 10  # 10 x 100 m = 1 km
OUT = os.path.join(HERE, "results", "v40", "map_nz30")
R_MODELS = ["GAM", "Lasso", "BRT", "RFdown", "MaxEnt", "Ensemble"]


def grid():
    layers, mask = {}, None
    for v in VARS[REGION]:
        with rasterio.open(os.path.join(HERE, "data", "Environment", REGION, f"{v}.tif")) as s:
            shape = (s.height // AGG, s.width // AGG)
            a = s.read(1, out_shape=shape, resampling=Resampling.nearest, masked=True)
            transform = s.transform * s.transform.scale(s.width / shape[1], s.height / shape[0])
        layers[v] = a.data.astype(float)
        m = ~np.ma.getmaskarray(a) & np.isfinite(a.data)
        mask = m if mask is None else mask & m
    rows, cols = np.nonzero(mask)
    df = pd.DataFrame({v: layers[v][rows, cols] for v in VARS[REGION]})
    return df, rows, cols, shape, transform


def fit():
    os.makedirs(OUT, exist_ok=True)
    d = [x for x in B.load_region(REGION) if x["species"] == SPECIES][0]
    g, rows, cols, shape, transform = grid()
    target = pd.concat([g, d["test"]], ignore_index=True)  # grid cells, then test sites
    n_grid = len(g)
    print(f"{SPECIES}: n_pres={len(d['pres'])}, grid cells={n_grid:,}, test sites={len(d['test']):,}", flush=True)

    # R reference models (same script as the benchmark)
    jd = os.path.join(OUT, "rjob")
    os.makedirs(jd, exist_ok=True)
    train = pd.concat([d["pres"], d["bg"]], ignore_index=True)
    train.insert(0, "occ", np.r_[np.ones(len(d["pres"]), int), np.zeros(len(d["bg"]), int)])
    train.to_csv(os.path.join(jd, "train.csv"), index=False)
    target.to_csv(os.path.join(jd, "test.csv"), index=False)
    rproc = subprocess.Popen(["Rscript", os.path.join(HERE, "R", "valavi_models.R"), jd, REGION, ",".join(R_MODELS)],
                             env={**os.environ, "OMP_NUM_THREADS": "1"})

    # TabPFN 3.5 Stable on the GPU, while R runs
    d_map = {**d, "test": target, "y_test": np.zeros(len(target), int)}
    B.RUN_TAG = "map"
    os.makedirs(B.PRED_DIR, exist_ok=True)
    rng = np.random.default_rng([B.SEED, B.REGIONS.index(REGION), 999])
    preds = {}
    for m in ["TabPFN3.5_Stable"]:
        B.run_models(d_map, [m], rng, REGION)  # metrics on dummy labels are ignored
        preds[m] = np.load(os.path.join(B.PRED_DIR, f"{REGION}_{SPECIES}_map.npz"))[m]
        print(f"{m} done", flush=True)
    rproc.wait()
    r = pd.read_csv(os.path.join(jd, "preds.csv"))
    for m in r.columns:
        preds["RF down-sampled" if m == "RFdown" else m] = r[m].values
    np.savez_compressed(os.path.join(OUT, "preds.npz"), rows=rows, cols=cols, shape=np.array(shape),
                        transform=np.array(transform)[:6], n_grid=n_grid, y_test=d["y_test"],
                        test_xy=pd.read_csv(os.path.join(B.DATA_DIR, "test_env", "NZtest_env.csv"))[["x", "y"]].values,
                        pres_xy=pd.read_csv(os.path.join(B.DATA_DIR, "train_po", "NZtrain_po.csv")).query(
                            "spid == @SPECIES")[["x", "y"]].values,
                        **{k.replace(" ", "_"): v for k, v in preds.items()})
    print("saved", os.path.join(OUT, "preds.npz"))


def fit_ft():
    """Add the fine-tuned TabPFN 3.5 (trained without NZ) to the existing map predictions."""
    d = [x for x in B.load_region(REGION) if x["species"] == SPECIES][0]
    g, rows, cols, shape, transform = grid()
    target = pd.concat([g, d["test"]], ignore_index=True)
    d_map = {**d, "test": target, "y_test": np.zeros(len(target), int)}
    B.RUN_TAG = "map"
    rng = np.random.default_rng([B.SEED, B.REGIONS.index(REGION), 998])
    B.run_models(d_map, ["TabPFN3.5ft_Stable"], rng, REGION)
    p = np.load(os.path.join(B.PRED_DIR, f"{REGION}_{SPECIES}_map.npz"))["TabPFN3.5ft_Stable"]
    f = os.path.join(OUT, "preds.npz")
    z = dict(np.load(f))
    z["TabPFN3.5ft_Stable"] = p
    np.savez_compressed(f, **z)
    print("saved fine-tuned predictions", flush=True)


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score

    z = np.load(os.path.join(OUT, "preds.npz"))
    rows, cols, shape, n = z["rows"], z["cols"], tuple(z["shape"]), int(z["n_grid"])
    a, b, c, d_, e, f = z["transform"]
    extent = [c, c + a * shape[1], f + e * shape[0], f]
    y_test = z["y_test"]
    panels = ["TabPFN3.5ft_Stable", "TabPFN3.5_Stable", "BRT", "RF_down-sampled", "Lasso", "GAM", "MaxEnt",
              "Ensemble"]
    panels = [m for m in panels if m in z]
    names = {"TabPFN3.5_Stable": "TabPFN 3.5 Stable", "TabPFN3.5ft_Stable": "TabPFN 3.5-FT Stable",
             "RF_down-sampled": "RF down-sampled"}
    cmap = plt.get_cmap("viridis_r")
    fig, axes = plt.subplots(3, 3, figsize=(13.5, 15))
    axes = axes.ravel()
    land = np.full(shape, np.nan)
    land[rows, cols] = 1
    ax = axes[0]
    ax.imshow(land, extent=extent, cmap="Greys", vmin=0, vmax=2.2, interpolation="nearest")
    ax.scatter(*z["pres_xy"].T, s=9, color="#2b1b3f", alpha=0.75, lw=0)
    ax.set_title("Occurrence", fontsize=13)
    for ax, m in zip(axes[1:], panels):
        p = z[m].astype(float)
        if m == "Ensemble" or p.max() > 1 or p.min() < 0:
            pass
        pg, pt = p[:n], p[n:]
        img = np.full(shape, np.nan)
        img[rows, cols] = pg
        im = ax.imshow(img, extent=extent, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
        auc = roc_auc_score(y_test, pt)
        ax.set_title(f"{names.get(m, m)}  (AUC {auc:.3f})", fontsize=12)
        # inset histogram: all grid cells (coloured) and test presences (grey)
        ins = ax.inset_axes([0.0, 0.62, 0.42, 0.3])
        bins = np.linspace(0, 1, 26)
        h, _ = np.histogram(pg, bins=bins, density=True)
        ins.bar(bins[:-1], h, width=np.diff(bins), align="edge", color=cmap((bins[:-1] + 0.02)), lw=0)
        ins.hist(pt[y_test == 1], bins=bins, density=True, color="#9b9a95", alpha=0.55)
        ins.set_yticks([])
        ins.set_xticks([0, 0.5, 1])
        ins.tick_params(labelsize=8)
        for sp in ["top", "right", "left"]:
            ins.spines[sp].set_visible(False)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    cax = fig.add_axes([0.93, 0.1, 0.012, 0.18])
    fig.colorbar(im, cax=cax, label="Likelihood")
    fig.suptitle("Predicted relative likelihood of occurrence for nz30 (NZ, 1 km grid). Insets: distribution of "
                 "predictions over all cells (coloured) and at test presences (grey)", fontsize=11, x=0.02, ha="left")
    out = os.path.join(HERE, "figures", "v40", "Fig11_map_nz30")
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=250, bbox_inches="tight")
    print("saved", out + ".png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--fit-ft", action="store_true")
    a = ap.parse_args()
    if a.fit:
        fit()
    if a.fit_ft:
        fit_ft()
    if a.plot:
        plot()
