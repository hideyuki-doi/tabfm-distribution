# Draw 50,000 random background cells per region from the NCEAS environment rasters,
# following Valavi et al. (2022): background sampled at random from the landscape,
# irrespective of species records. Output: data/background50k/{REGION}bg50k.csv
import os
import sys

import numpy as np
import pandas as pd
import rasterio

from valavi_config import VARS, DATA_DIR, HERE

N_BG = 50_000
SEED = 42
OUT = os.path.join(HERE, "data", "background50k")


def main(regions):
    os.makedirs(OUT, exist_ok=True)
    for region in regions:
        srcs = {v: rasterio.open(os.path.join(HERE, "data", "Environment", region, f"{v}.tif")) for v in VARS[region]}
        ref = next(iter(srcs.values()))
        for v, s in srcs.items():
            assert s.shape == ref.shape and s.transform == ref.transform, f"{region}/{v} grid differs"
        valid = np.ones(ref.shape, bool)
        layers = {}
        for v, s in srcs.items():
            a = s.read(1, masked=True)
            layers[v] = a.data
            valid &= ~np.ma.getmaskarray(a) & np.isfinite(a.data)
        idx = np.flatnonzero(valid)
        rng = np.random.default_rng(SEED)
        pick = np.sort(rng.choice(idx, N_BG, replace=False))
        rows, cols = np.unravel_index(pick, ref.shape)
        xs, ys = rasterio.transform.xy(ref.transform, rows, cols)
        df = pd.DataFrame({"x": xs, "y": ys, "occ": 0})
        for v in VARS[region]:
            df[v] = layers[v][rows, cols]
        df.to_csv(os.path.join(OUT, f"{region}bg50k.csv"), index=False)

        # sanity check: raster values at presence sites should equal the train_po values
        po = pd.read_csv(os.path.join(DATA_DIR, "train_po", f"{region}train_po.csv"))
        po = po.sample(min(500, len(po)), random_state=0)
        agree = {}
        for v, s in srcs.items():
            vals = np.array([x[0] for x in s.sample(list(zip(po.x, po.y)))], float)
            agree[v] = np.mean(np.isclose(vals, po[v].values, rtol=1e-3, atol=1e-3))
        print(f"{region}: valid cells={len(idx):,}  bg={len(df):,}  "
              f"raster==train_po at presences: min {min(agree.values()):.0%} ({min(agree, key=agree.get)})")


if __name__ == "__main__":
    main(sys.argv[1:] or list(VARS))
