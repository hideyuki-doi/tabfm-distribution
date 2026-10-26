# R packages for R/valavi_models.R. Versions used for the paper (R 4.5.2):
# mgcv 1.9.1, glmnet 5.1, gbm 2.3.1, randomForest 4.7.1.2, maxnet 0.1.4.
# MaxEnt (maxent.jar) needs Java >= 8 on the PATH; it is called with system2(), so rJava is not needed.
pkgs <- c(mgcv = "1.9.1", glmnet = "5.1", gbm = "2.3.1", randomForest = "4.7.1.2", maxnet = "0.1.4")
miss <- setdiff(names(pkgs), rownames(installed.packages()))
if (length(miss)) install.packages(miss, repos = "https://cloud.r-project.org")
for (p in names(pkgs)) {
  v <- as.character(packageVersion(p))
  if (v != pkgs[[p]]) message(sprintf("note: %s %s installed; the paper used %s", p, v, pkgs[[p]]))
}
