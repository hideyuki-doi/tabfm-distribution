# Tabular foundation models for predicting distributions from presence-only species records

Code and result tables for

> Doi H, Osawa T, Tsutsumida N. *Tabular foundation models for predicting distributions from presence-only species records.* (submitted)

The study evaluates the tabular foundation models TabPFN 3.5, TabPFN 2.5 and TabICL as presence-only distribution models. It uses the NCEAS benchmark (Elith et al. 2020): 225 species in six regions, evaluated on independent presence–absence surveys. The protocol follows Valavi et al. (2022, *Ecol. Monogr.* 92:e01486), so the results can be compared directly with their 21 modelling approaches. The study also covers:

- **Fine-tuning** of TabPFN 3.5. The main design is leave-one-region-out (LORO), so a region's model never sees data from that region. Alternative strategies and a species-split control are also included.
- **Spatially separated evaluation** with the blocks and target-group background of Valavi et al. (2023, *Glob. Ecol. Biogeogr.* 32:369–383).
- **Virtual species** with known truth: 120 species, each with 10, 30, 100 or 300 presence records, with and without environmental sampling bias. Half have additive and half have complex species–environment relationships.

## Contents

| Step | Script | Output | Paper |
|---|---|---|---|
| Shared settings | `valavi_config.py` | regions, covariates, categorical covariates | Methods |
| Metrics and statistics | `valavi_stats.py` | AUC-ROC, AUC-PRG, COR, Friedman aligned ranks with Shaffer post hoc | Methods |
| Data | `get_data.py`, `make_background.py` | `data/` (50,000 random background points per region) | Methods |
| Reference models (R) | `R/valavi_models.R`, `run_valavi_r.py`, `fix_maxnet.py` | `results/v40/results_r.csv` | GAM, Lasso, BRT, RF, RF down-sampled, MaxNet, MaxEnt, Ensemble |
| Python models | `benchmark_v40_valavi.py` | `results/v40/results_<tag>.csv` | XGBoost, TabPFN 3.5/2.5 (Fast, Stable), TabICL, TabPFN 3.5-FT |
| Ensembles with TabPFN | `make_ensembles.py` | `results/v40/results_derived.csv` | SI |
| Bagging configurations | `tune_tabpfn.py` | `results/v40/tuning/tune.csv` | SI (bagging table) |
| LORO fine-tuning | `finetune_sdm.py` | `results/v40/finetune/finetune_results.csv` and checkpoints | Main FT model |
| Other fine-tuning strategies | `finetune_pa.py`, `nested_pa.py`, `finetune_vs.py`, `ens_ft.py` | `results/v40/finetune/finetune_*_results.csv` | Fine-tuning figure, SI |
| Species-split control | `finetune_species_split.py`, `analyse_species_split.py` | `results/v40/finetune/finetune_species_split_results.csv`, `results/v40/review/species_split_*.csv` | Leakage test |
| Spatial evaluation | `spatial_cv.py`, `plot_v50.py` | `results/v50/results_{r,gpu}.csv` | Spatial-CV figure (SI) |
| Virtual species | `vs_benchmark.py`, `vs_bg5_summary.py`, `vs_figures.py` | `results/v60` (additive), `results/v61` (complex) | Virtual-species figure, SI |
| Virtual-species tests | `vs_tests.py` | `results/v6x/tests_species.csv` | Species-level tests quoted in the paper |
| Main figures | `make_main_figs.py` | `figures/main`, `figures/supp` | Figs 1, 2 and 4; calibration, species-level and spatial-CV SI figures |
| Benchmark figures | `plot_v40.py`, `plot_ft_strategies.py` | `figures/v40` | Replication check, ranks, post hoc P values and other SI figures |
| Map | `make_map_nz30.py` | `figures/v40` | New Zealand map (SI) |
| Robustness analyses | `robustness_1.py`, `robustness_2.py`, `robustness_3.py` | `results/v40/review/*.csv`, `results/v60/envelope.csv` | Bootstrap and mixed-model uncertainty, calibration slopes, categorical coding, sample size, envelope analysis |
| Published values | `reference/valavi2022_fig10_auc_by_region.csv` | | Regional AUC of Valavi et al. (2022, Fig. 10) |
| Compute time, strategies | `tables/` | | SI tables |
| Maintenance | `regen_r_preds.py` | `results/v40/preds/*_r.npz` | Re-saves R predictions at full precision (see Per-species predictions) |

`results/` holds the per-species result tables for every model and analysis in the paper. With these tables, `make_main_figs.py`, `plot_v40.py`, `plot_v50.py`, `plot_ft_strategies.py` and `analyse_species_split.py` run without a GPU or the input data. The calibration slopes, the robustness analyses and the virtual-species example curves also need the per-species predictions (see below).

## Setup

**Python 3.12** with an NVIDIA GPU for TabPFN, TabICL and fine-tuning. The analyses were run on an NVIDIA DGX Spark (GB10, CUDA 13).

```bash
conda create -n sdm-bench python=3.12
conda activate sdm-bench
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
```

**R 4.5.2** with mgcv, glmnet, gbm, randomForest and maxnet, plus **Java ≥ 8** for MaxEnt:

```bash
Rscript R/install_packages.R
```

**Third-party code.** The R reference models use `gbm.step()` from dismo 1.3-14 (GPL ≥ 3), `make_quadratic()` from [rvalavi/myspatial](https://github.com/rvalavi/myspatial) and `maxent.jar` (MaxEnt 3.4.4), exactly as in Valavi et al. (2022). These are not redistributed here. The following script downloads them and checks them against the copies used in the paper:

```bash
bash R/third_party/fetch.sh
```

**TabPFN weights.** TabPFN 3.5 and 2.5 weights are released by Prior Labs under a non-commercial licence. On first use, the `tabpfn` package asks you to accept the licence in a browser and caches a token in `~/.cache/tabpfn/auth_token`; alternatively, set the environment variable `TABPFN_TOKEN`. Never commit this token.

**Data.** All input data are public:

```bash
python get_data.py                   # NCEAS data (OSF kwc4v) and Valavi et al. (2023) blocks / TGB, about 0.6 GB
python make_background.py            # 50,000 random background points per region
```

- `data/Records`, `data/Environment`: Elith J. et al. (2020) Presence-only and presence-absence data for comparing species distribution modeling methods. *Biodiversity Informatics* 15:69–80. <https://osf.io/kwc4v/>
- `data/valavi2023`: spatial blocks and target-group background samples from <https://github.com/rvalavi/SDM-Spatial-Validation>

## Reproducing the analyses

```bash
# 1. Benchmark under the protocol of Valavi et al. (2022)
python run_valavi_r.py --workers 12                                   # R reference models
python benchmark_v40_valavi.py --models XGBoost --tag cpu_xgb
python benchmark_v40_valavi.py --models TabPFN3.5_Fast,TabPFN2.5_Fast,TabICL --tag gpu_fast
python benchmark_v40_valavi.py --models TabPFN3.5_Stable --tag gpu_stable35
python benchmark_v40_valavi.py --models TabPFN2.5_Stable --tag gpu_stable25

# 2. Leave-one-region-out fine-tuning, then the fine-tuned model in the benchmark
for R in AWT CAN NSW NZ SA SWI; do python finetune_sdm.py --holdout $R; done
python benchmark_v40_valavi.py --models TabPFN3.5ft_Fast --tag gpu_ft_fast
python benchmark_v40_valavi.py --models TabPFN3.5ft_Stable --tag gpu_ft_stable
python make_ensembles.py

# 3. Alternative fine-tuning strategies and the species-split control (optional)
for R in AWT CAN NSW NZ SA SWI; do
  python finetune_pa.py --holdout $R
  python finetune_vs.py --holdout $R
  python finetune_vs.py --holdout $R --mix
  python finetune_vs.py --holdout $R --mix --tg
done
python nested_pa.py phaseA && python nested_pa.py phaseB && python nested_pa.py phaseC
python ens_ft.py
python finetune_species_split.py && python analyse_species_split.py

# 4. Spatially separated evaluation (Valavi et al. 2023)
python spatial_cv.py tasks && python spatial_cv.py r && python spatial_cv.py gpu

# 5. Virtual species (additive: results/v60; complex: results/v61)
for F in additive complex; do
  for S in simulate r gpu analyse; do VS_FAMILY=$F python vs_benchmark.py $S; done
done
VS_FAMILY=additive python vs_benchmark.py gpu --bg-ratio 5    # 5x background per bag, n = 10 and 30
VS_FAMILY=additive python vs_benchmark.py analyse             # adds these runs to results/v60/results.csv
python vs_bg5_summary.py
python vs_tests.py                                            # species-level tests quoted in the paper

# 6. Figures, tables and robustness analyses
python make_main_figs.py
python plot_v40.py
python plot_v50.py
python plot_ft_strategies.py
python vs_figures.py
python make_map_nz30.py --fit && python make_map_nz30.py --plot
python robustness_1.py && python robustness_2.py && python robustness_3.py
```

Steps 3–5 use the leave-one-region-out checkpoints written by step 2 (`results/v40/finetune/tabpfn35_ft_<region>_e10_lr1e-05.ckpt`). TabPFN 3.5 "Fast" uses 5 balanced bags and "Stable" 100 bags with test-time augmentation; the spatially separated evaluation and the virtual species use 20 bags.

- **Random streams.** Fixed seeds are used throughout. In `benchmark_v40_valavi.py`, the TabPFN and TabICL models of one call draw their bags from one random stream per species, in the order given by `--models`. The reported values come from the calls listed above, and rerunning the same calls reproduces them exactly; running a model in a different call gives statistically equivalent but not identical values.
- **Resuming and failed fits.** Rerunning a script skips finished work. A fit that failed (no AUC, or a prediction file without every model) does not count as finished and is re-run; the new rows supersede the old ones (`run_valavi_r.py --keep-failed` disables this for the R models). A species that is only partly finished is re-run with the whole model list, so that every model receives the same random draws as in a complete run.
- **Statistics on virtual species.** The four sample sizes of a species are not independent, so tests pooled over sample sizes average the paired differences within each species and test them across species (`vs_tests.py`).
- **GPU fine-tuning** is not bitwise reproducible (non-deterministic CUDA kernels in backpropagation). Predictions from a given checkpoint are.

## Reproducibility check

The repository was checked by cloning it into a fresh environment built from `requirements.txt`, downloading the data with `get_data.py` and rerunning subsets of every analysis. The downloaded data, the background points, the simulated virtual species and the spatial folds were byte-identical to those used in the paper. Reruns of the R reference models, XGBoost, TabPFN 3.5 and 2.5 (5 and 100 bags) and TabICL reproduced the reported per-species metrics exactly, as did all summary tables and the analyses based on the released predictions. Fine-tuning reproduced the regional mean AUC to the fourth decimal (per-species differences up to 0.004; see GPU fine-tuning above).

## Per-species predictions

The per-species predictions at the survey sites (`results/v40/preds`) and the virtual-species predictions (`results/v6*/preds`) are too large for the repository. They are attached to the GitHub release as `predictions_benchmark_v40.tar.gz` and `predictions_virtual_species.tar.gz` (including the simulated landscapes). Extract them in the repository root, e.g. `tar -xzf predictions_benchmark_v40.tar.gz`. Metrics recomputed from the benchmark predictions agree with the reported values to within 10<sup>-6</sup>. In release v1.0 the R predictions were stored as float32, which tied some near-0 or near-1 predictions and shifted AP and AUC<sub>PRG</sub> recomputed from the files by up to 0.008 (reported values were computed at full precision); from v1.1 the affected files are re-saved at full precision (`regen_r_preds.py`), and all scripts save predictions at full precision. The virtual-species metrics are computed from the saved predictions themselves. The fine-tuned TabPFN checkpoints are not distributed, because the TabPFN weights are under a non-commercial licence. They can be recreated with step 2.

## Licence

| Component | Licence | In this repository? |
|---|---|---|
| Code (`*.py`, `R/*.R`, `R/third_party/fetch.sh`) | MIT (`LICENSE`) | yes |
| Result tables (`results/`, `tables/`) and the prediction archives of the release | CC BY 4.0 | yes |
| `reference/valavi2022_fig10_auc_by_region.csv` | values published by Valavi et al. (2022, Fig. 10); please cite them | yes |
| NCEAS data (Elith et al. 2020, OSF kwc4v) | CC BY 4.0 | no; downloaded by `get_data.py` |
| Spatial blocks and target-group background (Valavi et al. 2023) | no licence stated in the source repository | no; downloaded by `get_data.py` |
| dismo 1.3-14 (`gbm.step` and helpers) | GPL (≥ 3) | no; downloaded by `R/third_party/fetch.sh` |
| `make_quadratic()` (rvalavi/myspatial) | no licence stated in the source repository | no; downloaded by `R/third_party/fetch.sh` |
| MaxEnt 3.4.4 (`maxent.jar`) | MIT | no; downloaded by `R/third_party/fetch.sh` |
| `tabpfn` package | Apache 2.0 | no; installed with pip |
| TabPFN 2.5 and 3.5 weights | Prior Labs TabPFN-2.5 / TabPFN-3.5 licences (non-commercial use) | no; downloaded by `tabpfn` after accepting the licence |
| `tabicl` package and weights | BSD 3-Clause | no; installed with pip |

Fine-tuned TabPFN checkpoints are derived from the TabPFN 3.5 weights and are not redistributed; step 2 recreates them. Using this code for commercial purposes requires a separate licence for the TabPFN weights from Prior Labs.

## Citation

If you use this code or the result tables, please cite the paper. The citation will be updated on publication; GitHub's "Cite this repository" button uses `CITATION.cff`.

> Doi H, Osawa T, Tsutsumida N. Tabular foundation models for predicting distributions from presence-only species records. (submitted)

```bibtex
@unpublished{doi_tabfm_sdm,
  author = {Doi, Hideyuki and Osawa, Takeshi and Tsutsumida, Narumasa},
  title  = {Tabular foundation models for predicting distributions from presence-only species records},
  note   = {Submitted. Code: https://github.com/hideyuki-doi/tabfm-distribution},
  year   = {2026}
}
```

Please also cite the models, the data and the benchmark protocol that the analyses build on:

**Tabular foundation models**
- Hollmann N, et al. (2025) Accurate predictions on small data with a tabular foundation model. *Nature* 637:319–326. https://doi.org/10.1038/s41586-024-08328-6 (TabPFN)
- Müller S, Hollmann N, Pineda Arango S, Grabocka J, Hutter F (2022) Transformers can do Bayesian inference. *International Conference on Learning Representations*. arXiv:2112.10510 (prior-data fitted networks)
- Grinsztajn L, et al. (2025) TabPFN-2.5: advancing the state of the art in tabular foundation models. arXiv:2511.08667
- Jäger B, et al. (2026) TabPFN-3.5: technical report. arXiv:2609.17895
- Qu J, Holzmüller D, Varoquaux G, Le Morvan M (2025) TabICL: a tabular foundation model for in-context learning on large data. *Proceedings of the 42nd International Conference on Machine Learning*, PMLR 267:50817–50847

**Data and benchmark protocol**
- Elith J, et al. (2020) Presence-only and presence-absence data for comparing species distribution modeling methods. *Biodiversity Informatics* 15:69–80. https://doi.org/10.17161/bi.v15i2.13384 (NCEAS data)
- Elith J, et al. (2006) Novel methods improve prediction of species' distributions from occurrence data. *Ecography* 29:129–151. https://doi.org/10.1111/j.2006.0906-7590.04596.x
- Valavi R, Guillera-Arroita G, Lahoz-Monfort JJ, Elith J (2022) Predictive performance of presence-only species distribution models: a benchmark study with reproducible code. *Ecological Monographs* 92:e01486. https://doi.org/10.1002/ecm.1486
- Valavi R, et al. (2023) Flexible species distribution modelling methods perform well on spatially separated testing data. *Global Ecology and Biogeography* 32:369–383. https://doi.org/10.1111/geb.13639 (spatial blocks and target-group background)

**Reference models**
- Phillips SJ, Anderson RP, Schapire RE (2006) Maximum entropy modeling of species geographic distributions. *Ecological Modelling* 190:231–259. https://doi.org/10.1016/j.ecolmodel.2005.03.026 (MaxEnt)
- Phillips SJ, Anderson RP, Dudík M, Schapire RE, Blair ME (2017) Opening the black box: an open-source release of Maxent. *Ecography* 40:887–893. https://doi.org/10.1111/ecog.03049 (MaxNet)
- Elith J, Leathwick JR, Hastie T (2008) A working guide to boosted regression trees. *Journal of Animal Ecology* 77:802–813. https://doi.org/10.1111/j.1365-2656.2008.01390.x (BRT, `gbm.step`)
