# Reference SDMs fitted exactly as in Valavi et al. (2022) Ecol. Monogr. 92:e01486
# (Table 2 and Appendix S2: github.com/rvalavi/Presence-Only-SDM/index.Rmd).
#
# Usage: Rscript valavi_models.R <job_dir> <region> <models>
#   <job_dir>/train.csv : occ + covariates (presences then 50k background)
#   <job_dir>/test.csv  : covariates at the presence-absence sites
#   writes <job_dir>/preds.csv (one column per model) and <job_dir>/times.csv
# models: comma list of GAM, Lasso, BRT, RF, RFdown, MaxNet, MaxEnt, Ensemble

suppressPackageStartupMessages({
  library(mgcv); library(glmnet); library(gbm); library(randomForest); library(maxnet); library(Matrix)
})

args <- commandArgs(trailingOnly = TRUE)
jd <- args[1]; region <- args[2]; models <- strsplit(args[3], ",")[[1]]
here <- dirname(normalizePath(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))))
tp <- file.path(here, "third_party")
source(file.path(tp, "gbm.utils.R"))     # dismo internals (.roc, .calibration) used by gbm.step
source(file.path(tp, "calc.deviance.R")) # dismo::calc.deviance
source(file.path(tp, "gbm.step.R"))      # dismo::gbm.step (dismo 1.3-14)
source(file.path(tp, "make_quadratic.R"))  # myspatial::make_quadratic (orthogonal quadratic terms)
maxent_jar <- file.path(tp, "maxent.jar")   # MaxEnt 3.4.4 (as in the paper)

cats <- list(CAN = "ontveg", NSW = "vegsys", NZ = c("age", "toxicats"), SWI = "calc")[[region]]
training <- read.csv(file.path(jd, "train.csv"))
testing_env <- read.csv(file.path(jd, "test.csv"))
covars <- setdiff(names(training), "occ")
cats <- intersect(cats, covars)  # e.g. NSW vegsys is not used in the spatial-CV design

# categorical covariates as factors (unseen test levels -> most common training level)
for (v in cats) {
  lv <- sort(unique(training[[v]]))
  tst <- testing_env[[v]]
  tst[!tst %in% lv] <- as.numeric(names(which.max(table(training[[v]]))))
  training[[v]] <- factor(training[[v]], levels = lv)
  testing_env[[v]] <- factor(tst, levels = lv)
}
cont <- setdiff(covars, cats)

# normalized copies (continuous covariates scaled with training mean / sd) for GAM, BRT, RF
training_n <- training; testing_n <- testing_env
for (v in cont) {
  m <- mean(training[[v]]); s <- sd(training[[v]]); if (s == 0) s <- 1
  training_n[[v]] <- (training[[v]] - m) / s
  testing_n[[v]] <- (testing_env[[v]] - m) / s
}

prNum <- sum(training$occ == 1)
bgNum <- sum(training$occ == 0)
wt <- ifelse(training$occ == 1, 1, prNum / bgNum)   # down-weighting (DW)

preds <- list(); times <- list()
MODEL_TIME_LIMIT <- as.numeric(Sys.getenv("MODEL_TIME_LIMIT", 3 * 3600))       # s; no model took > 1.3 h except GAM fits that never converge
GAM_FACTOR_TIME_LIMIT <- as.numeric(Sys.getenv("GAM_FACTOR_TIME_LIMIT", 1800))      # s; GAM with factor terms, before falling back to no factors
run <- function(name, expr) {
  t0 <- Sys.time()
  setTimeLimit(elapsed = MODEL_TIME_LIMIT, transient = TRUE)
  on.exit(setTimeLimit(elapsed = Inf))
  p <- tryCatch(expr, error = function(e) { message(name, " error: ", conditionMessage(e)); NULL })
  setTimeLimit(elapsed = Inf)
  times[[name]] <<- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  if (!is.null(p)) preds[[name]] <<- as.numeric(p)
}

# ---- GAM: mgcv, REML, smooths for continuous covariates, DW --------------------------
fit_gam <- function(use_factors = TRUE) {
  vs <- if (use_factors) covars else setdiff(covars, cats)
  terms <- sapply(vs, function(v) {
    if (v %in% cats) return(v)
    if (length(unique(training_n[[v]])) < 15) return(v)   # few-valued ordinal covariates as linear terms
    paste0("s(", v, ")")
  })
  form <- as.formula(paste("occ ~", paste(terms, collapse = " + ")))
  set.seed(32639)
  gm <- mgcv::gam(formula = form, data = training_n, family = binomial(link = "logit"),
                  weights = wt, method = "REML")
  predict(gm, testing_n, type = "response")
}
# factor terms can make the REML fit diverge (e.g. NZ age / toxicats); Valavi's own NZ and CAN
# GAM formulas omit the factors, so fall back to that
fit_gam_safe <- function() {
  if (!length(cats)) return(fit_gam(TRUE))
  p <- tryCatch({
    setTimeLimit(elapsed = GAM_FACTOR_TIME_LIMIT, transient = TRUE)
    fit_gam(TRUE)
  }, error = function(e) {
    message("GAM with factors failed (", conditionMessage(e), "); refitting without factor terms")
    NULL
  })
  setTimeLimit(elapsed = MODEL_TIME_LIMIT, transient = TRUE)
  if (is.null(p)) p <- fit_gam(FALSE)
  p
}

# ---- Lasso: glmnet on orthogonal quadratic terms, DW, 10-fold CV, lambda.min ---------
fit_lasso <- function() {
  quad_obj <- make_quadratic(training, cols = covars, verbose = FALSE)
  training_quad <- predict.make_quadratic(quad_obj, newdata = training)
  testing_quad <- predict.make_quadratic(quad_obj, newdata = testing_env)
  new_vars <- names(training_quad)[names(training_quad) != "occ"]
  training_sparse <- sparse.model.matrix(~. - 1, training_quad[, new_vars])
  testing_sparse <- sparse.model.matrix(~. - 1, testing_quad[, new_vars])
  set.seed(32639)
  lasso_cv <- tryCatch(
    cv.glmnet(x = training_sparse, y = training_quad$occ, family = "binomial",
              alpha = 1, weights = wt, nfolds = 10),
    error = function(e) {   # rare non-convergence at small lambda: shorter path, more iterations
      set.seed(32639)
      cv.glmnet(x = training_sparse, y = training_quad$occ, family = "binomial",
                alpha = 1, weights = wt, nfolds = 10, maxit = 1e6, lambda.min.ratio = 0.01)
    })
  predict(lasso_cv, testing_sparse, type = "response", s = "lambda.min")
}

# ---- BRT: dismo::gbm.step, tc = 1 (<50 presences) or 5, lr 0.001, bag 0.75, DW --------
fit_brt <- function() {
  # fallbacks if gbm.step finds the CV deviance rising from the first step:
  # smaller learning rates, then a smaller step size (10 trees)
  for (cfg in list(c(0.001, 50), c(0.0005, 50), c(0.0001, 50), c(0.0001, 10))) {
    set.seed(32639)
    brt <- gbm.step(data = training_n, gbm.x = which(names(training_n) %in% covars),
                    gbm.y = which(names(training_n) == "occ"), family = "bernoulli",
                    tree.complexity = ifelse(prNum < 50, 1, 5), learning.rate = cfg[1],
                    bag.fraction = 0.75, max.trees = 10000, n.trees = cfg[2], n.folds = 5,
                    site.weights = wt, silent = TRUE, plot.main = FALSE)
    if (!is.null(brt)) {
      return(predict(brt, testing_n, n.trees = brt$gbm.call$best.trees, type = "response"))
    }
  }
  # CV never improves on the null model: the optimal BRT has no trees (constant prediction)
  message("BRT: gbm.step found no improvement over the intercept; using the null model")
  rep(prNum / (prNum + bgNum), nrow(testing_n))
}

# ---- RF (default) and RF down-sampled ------------------------------------------------
fit_rf <- function() {
  d <- training_n; d$occ <- as.factor(d$occ)
  set.seed(32639)
  rf <- randomForest(formula = occ ~ ., data = d, ntree = 500)
  predict(rf, testing_n, type = "prob")[, "1"]
}
fit_rfdown <- function() {
  d <- training_n; d$occ <- as.factor(d$occ)
  set.seed(32639)
  rf <- randomForest(formula = occ ~ ., data = d, ntree = 1000,
                     sampsize = c("0" = prNum, "1" = prNum), replace = TRUE)
  predict(rf, testing_n, type = "prob")[, "1"]
}

# maxnet::maxnet (0.1.4) with two changes: (1) if glmnet stops before the end of the 200-lambda
# path (non-convergence at the smallest lambdas), use the last lambda it reached instead of failing;
# (2) features with identical names are merged by adding their coefficients. For a covariate with few
# distinct values, the forward hinge at the minimum and the reverse hinge at the maximum are the same
# feature with the same name, and maxnet's predict() then fails with 'non-conformable arguments'.
# Merging leaves the linear predictor, and so the model, unchanged.
maxnet_partial <- function(p, data, f, regmult = 1, regfun = maxnet::maxnet.default.regularization) {
  pdata <- data[p == 1, ]; ndata <- data[p == 0, ]
  toadd <- apply(pdata, 1, function(rr) !any(apply(ndata, 1, function(r) identical(r, rr))))
  p <- c(p, rep(0, sum(toadd))); data <- rbind(data, pdata[toadd, ])
  mm <- model.matrix(f, data)
  reg <- regfun(p, mm) * regmult
  weights <- p + (1 - p) * 100
  glmnet::glmnet.control(pmin = 1e-08, fdev = 0)
  model <- suppressWarnings(glmnet::glmnet(x = mm, y = as.factor(p), family = "binomial", standardize = F,
    penalty.factor = reg, lambda = 10^(seq(4, 0, length.out = 200)) * sum(reg) / length(reg) * sum(p) / sum(weights),
    weights = weights))
  class(model) <- c("maxnet", class(model))
  k <- length(model$lambda)
  message("maxnet_partial: path stopped at lambda ", k, "/200")
  bb <- model$beta[, k]
  bb <- bb[bb != 0]
  model$betas <- vapply(split(bb, factor(names(bb), levels = unique(names(bb)))), sum, numeric(1))
  if (anyDuplicated(names(bb))) message("maxnet_partial: merged duplicated features ",
                                       paste(unique(names(bb)[duplicated(names(bb))]), collapse = ", "))
  model$alpha <- 0
  rr <- maxnet:::predict.maxnet(model, data[p == 0, , drop = FALSE], type = "exponent", clamp = F)
  raw <- rr / sum(rr)
  model$entropy <- -sum(raw * log(raw))
  model$alpha <- -log(sum(rr))
  model$penalty.factor <- reg
  model$featuremins <- apply(mm, 2, min)
  model$featuremaxs <- apply(mm, 2, max)
  vv <- (sapply(data, class) != "factor")
  model$varmin <- apply(data[, vv, drop = FALSE], 2, min)
  model$varmax <- apply(data[, vv, drop = FALSE], 2, max)
  means <- apply(data[p == 1, vv, drop = FALSE], 2, mean)
  majorities <- sapply(names(data)[!vv], function(n) which.max(table(data[p == 1, n, drop = FALSE])))
  names(majorities) <- names(data)[!vv]
  model$samplemeans <- unlist(c(means, majorities))
  model$levels <- lapply(data, levels)
  model
}

# ---- MaxNet: default features, regmult 1 --------------------------------------------
fit_maxnet <- function() {
  covariates <- training[, covars, drop = FALSE]
  set.seed(32639)
  # glmnet occasionally fails to converge near the end of maxnet's 200-lambda path
  # (default thresh = 1e-7); retry the identical model with a looser convergence threshold
  mx <- tryCatch(
    maxnet::maxnet(p = training$occ, data = covariates, regmult = 1,
                   maxnet.formula(training$occ, covariates, classes = "default")),
    error = function(e) {
      message("MaxNet: ", conditionMessage(e), " -> retry with glmnet thresh = 1e-5")
      m <- tryCatch(maxnet::maxnet(p = training$occ, data = covariates, regmult = 1,
                                   maxnet.formula(training$occ, covariates, classes = "default"),
                                   thresh = 1e-5), error = function(e2) NULL)
      if (!is.null(m)) return(m)
      message("MaxNet: refitting with maxnet_partial (partial path / merged duplicate features)")
      maxnet_partial(p = training$occ, data = covariates, regmult = 1,
                     f = maxnet.formula(training$occ, covariates, classes = "default"))
    })
  predict(mx, testing_env[, covars, drop = FALSE], type = "cloglog")
}

# ---- MaxEnt 3.4.4 (as dismo::maxent with args = "nothreshold") ---------------------
fit_maxent <- function() {
  d <- file.path(jd, "maxent"); dir.create(d, showWarnings = FALSE)
  num <- function(df) { for (v in cats) df[[v]] <- as.numeric(as.character(df[[v]])); df }
  pv <- num(training[training$occ == 1, covars, drop = FALSE])
  av <- num(training[training$occ == 0, covars, drop = FALSE])
  te <- num(testing_env[, covars, drop = FALSE])
  write.csv(cbind(species = "species", x = seq_len(nrow(pv)), y = seq_len(nrow(pv)), pv),
            file.path(d, "presence"), row.names = FALSE, quote = FALSE)
  write.csv(cbind(species = "background", x = seq_len(nrow(av)), y = seq_len(nrow(av)), av),
            file.path(d, "absence"), row.names = FALSE, quote = FALSE)
  write.csv(cbind(species = "test", x = seq_len(nrow(te)), y = seq_len(nrow(te)), te),
            file.path(d, "test.csv"), row.names = FALSE, quote = FALSE)
  toggles <- if (length(cats)) paste0("togglelayertype=", cats) else character(0)
  st <- system2("java", c("-mx8g", "-jar", maxent_jar, "-z", "autorun",
                          "-e", file.path(d, "absence"), "-o", d, "-s", file.path(d, "presence"),
                          "nothreshold", toggles, "warnings=false", "askoverwrite=false",
                          "responsecurves=false", "pictures=false", "plots=false",
                          "writebackgroundpredictions=false", "threads=1"),
                stdout = FALSE, stderr = FALSE)
  if (st != 0) stop("maxent.jar fit failed")
  system2("java", c("-mx8g", "-cp", maxent_jar, "density.Project", file.path(d, "species.lambdas"),
                    file.path(d, "test.csv"), file.path(d, "proj.csv")), stdout = FALSE, stderr = FALSE)
  pr <- read.csv(file.path(d, "proj.csv"))
  pr[order(pr$x), 3]   # cloglog
}

fitters <- list(GAM = fit_gam_safe, Lasso = fit_lasso, BRT = fit_brt, RF = fit_rf, RFdown = fit_rfdown,
                MaxNet = fit_maxnet, MaxEnt = fit_maxent)
for (m in setdiff(models, "Ensemble")) run(m, fitters[[m]]())

# ---- Ensemble: mean of 0-1 rescaled Lasso, GAM, MaxEnt, BRT, RF down-sampled ----------
if ("Ensemble" %in% models) {
  parts <- c("Lasso", "GAM", "MaxEnt", "BRT", "RFdown")
  if (all(parts %in% names(preds))) {
    rescale <- function(x) if (max(x) > min(x)) (x - min(x)) / (max(x) - min(x)) else rep(0, length(x))  # constant member adds no ranking
    run("Ensemble", rowMeans(sapply(preds[parts], rescale)))
  } else {
    times[["Ensemble"]] <- 0
  }
}

out <- as.data.frame(preds)
write.csv(out, file.path(jd, "preds.csv"), row.names = FALSE)
write.csv(data.frame(model = names(times), time = unlist(times)), file.path(jd, "times.csv"), row.names = FALSE)
