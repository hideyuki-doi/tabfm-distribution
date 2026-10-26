#!/usr/bin/env bash
# Download the third-party code used by R/valavi_models.R. It is not redistributed in this repository.
#   - dismo 1.3-14 (GPL >= 3, CRAN archive): gbm.step.R, gbm.utils.R, calc.deviance.R
#   - make_quadratic() and predict.make_quadratic() from rvalavi/myspatial (R/prediction_helper.R)
#   - maxent.jar (MaxEnt 3.4.4) as distributed with Valavi et al. (2023), rvalavi/SDM-Spatial-Validation
# The files are checked against the SHA-256 sums of the copies used in the paper (SHA256SUMS).
# Usage: bash R/third_party/fetch.sh
set -euo pipefail
cd "$(dirname "$0")"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

curl -fsSL https://cran.r-project.org/src/contrib/Archive/dismo/dismo_1.3-14.tar.gz | tar -xz -C "$tmp"
cp "$tmp"/dismo/R/gbm.step.R "$tmp"/dismo/R/gbm.utils.R "$tmp"/dismo/R/calc.deviance.R .

curl -fsSL -o "$tmp/prediction_helper.R" \
  https://raw.githubusercontent.com/rvalavi/myspatial/d44713ca8d6574b8ffd088ecf8e2db963b4da3e3/R/prediction_helper.R
# (extracted as for the paper; predict.make_quadratic ends up defined twice, identically)
sed -n '/make_quadratic <- function/,/^}/p' "$tmp/prediction_helper.R" > make_quadratic.R
sed -n '/predict.make_quadratic <- function/,/^}/p' "$tmp/prediction_helper.R" |
  awk 'BEGIN{n=0} /predict.make_quadratic <- function/{n++} n==1' >> make_quadratic.R

curl -fsSL -o maxent.jar \
  https://github.com/rvalavi/SDM-Spatial-Validation/raw/3810039ce7cb32ecb3cc05d6cf5d22eb7db784a6/maxent.jar

sha256sum -c SHA256SUMS
