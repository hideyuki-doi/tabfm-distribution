# Download the public input data into data/ (standard library only).
#   data/Records, data/Environment : NCEAS benchmark data of Elith et al. (2020, Biodiversity
#       Informatics 15:69-80), OSF project kwc4v (https://osf.io/kwc4v/), folder "data"
#   data/valavi2023/{blocks,tgb_samples} : spatial blocks and target-group background samples of
#       Valavi et al. (2023, Global Ecology and Biogeography 32:369-383),
#       github.com/rvalavi/SDM-Spatial-Validation (folder "data", pinned commit)
# Usage: python get_data.py [--no-rasters]
#   --no-rasters skips data/Environment (538 MB). The rasters are needed only to redraw the
#   background points (make_background.py) and for the New Zealand map (make_map_nz30.py).
import argparse
import json
import os
import time
import urllib.request

from valavi_config import HERE

OSF_DATA = "https://api.osf.io/v2/nodes/kwc4v/files/osfstorage/5d6e06147215100019a0d2d2/"
V23 = "rvalavi/SDM-Spatial-Validation"
V23_SHA = "3810039ce7cb32ecb3cc05d6cf5d22eb7db784a6"
DATA = os.path.join(HERE, "data")


def fetch(url, dest, tries=5):
    if os.path.exists(dest):
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(dest + ".part", "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            os.replace(dest + ".part", dest)
            return
        except OSError:
            if k == tries - 1:
                raise
            time.sleep(5 * (k + 1))


def get_json(url):
    with urllib.request.urlopen(url, timeout=120) as r:
        return json.load(r)


def osf_folder(url, dest, skip):
    """Download an OSF folder recursively (following pagination)."""
    while url:
        d = get_json(url)
        for x in d["data"]:
            a = x["attributes"]
            if a["kind"] == "folder":
                if a["name"] not in skip:
                    osf_folder(x["relationships"]["files"]["links"]["related"]["href"],
                               os.path.join(dest, a["name"]), ())
            else:
                fetch(x["links"]["download"], os.path.join(dest, a["name"]))
                print("  ", os.path.relpath(os.path.join(dest, a["name"]), HERE))
        url = d["links"].get("next")


def valavi2023():
    tree = get_json(f"https://api.github.com/repos/{V23}/git/trees/{V23_SHA}?recursive=1")["tree"]
    for t in tree:
        p = t["path"]
        if t["type"] == "blob" and (p.startswith("data/blocks/") or p.startswith("data/tgb_samples/")):
            fetch(f"https://raw.githubusercontent.com/{V23}/{V23_SHA}/{p}",
                  os.path.join(DATA, "valavi2023", p[len("data/"):]))
            print("  ", os.path.join("data", "valavi2023", p[len("data/"):]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-rasters", action="store_true")
    args = ap.parse_args()
    print("NCEAS data (OSF kwc4v) ->", DATA)
    osf_folder(OSF_DATA, DATA, skip=("Borders",) + (("Environment",) if args.no_rasters else ()))
    print("Valavi et al. (2023) blocks and target-group background ->", os.path.join(DATA, "valavi2023"))
    valavi2023()
