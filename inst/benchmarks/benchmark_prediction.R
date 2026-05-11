#!/usr/bin/env Rscript
# benchmark_prediction.R — Prediction benchmark: maxentcpp vs dismo
#
# Compares prediction time on the bundled 10x10 mock fixture after
# training models with equivalent settings.
# Requires: maxentcpp, microbenchmark
# Optional: dismo, rJava, raster (for Java comparison)
#
# Usage:
#   Rscript inst/benchmarks/benchmark_prediction.R

cat("=== maxentcpp Prediction Benchmark ===\n\n")

# ---- Check dependencies ----------------------------------------------------

if (!requireNamespace("maxentcpp", quietly = TRUE))
    stop("maxentcpp is required for benchmarks.")
if (!requireNamespace("microbenchmark", quietly = TRUE))
    stop("microbenchmark is required. Install with install.packages('microbenchmark').")

has_dismo <- requireNamespace("dismo", quietly = TRUE) &&
    requireNamespace("rJava", quietly = TRUE) &&
    requireNamespace("raster", quietly = TRUE)

if (has_dismo) {
    jar_path <- file.path(system.file(package = "dismo"), "java", "maxent.jar")
    has_dismo <- file.exists(jar_path)
    if (!has_dismo)
        message("maxent.jar not found in dismo; skipping Java benchmarks.")
}

# ---- Prepare data and train models (outside timing loop) --------------------

paths <- maxentcppCompTest::mock_raster_paths()
occ   <- maxentcppCompTest::mock_occurrences()

# --- maxentcpp model ---
g1 <- maxentcpp::maxent_read_asc(paths["bio1"])
g2 <- maxentcpp::maxent_read_asc(paths["bio2"])
m1 <- maxentcpp::maxent_grid_to_matrix(g1)
m2 <- maxentcpp::maxent_grid_to_matrix(g2)
info <- maxentcpp::maxent_grid_info(g1)

bio1_vec <- as.numeric(t(m1))
bio2_vec <- as.numeric(t(m2))
n <- length(bio1_vec)

occ_indices <- vapply(seq_len(nrow(occ)), function(i) {
    row_i <- floor((info$yll + info$nrows * info$cellsize - occ$lat[i]) /
                       info$cellsize)
    col_i <- floor((occ$lon[i] - info$xll) / info$cellsize)
    row_i <- max(0L, min(as.integer(row_i), info$nrows - 1L))
    col_i <- max(0L, min(as.integer(col_i), info$ncols - 1L))
    as.integer(row_i * info$ncols + col_i)
}, integer(1L))

f1 <- maxentcpp::maxent_linear_feature(bio1_vec, "bio1")
f2 <- maxentcpp::maxent_linear_feature(bio2_vec, "bio2")
fs <- maxentcpp::maxent_featured_space(n, occ_indices, list(f1, f2))
maxentcpp::maxent_fit(fs, max_iter = 500L, convergence = 1e-5,
                      beta_multiplier = 1.0)

# Build feature matrix for prediction
f1p <- maxentcpp::maxent_linear_feature(bio1_vec, "bio1")
f2p <- maxentcpp::maxent_linear_feature(bio2_vec, "bio2")
feat_mat <- cbind(
    vapply(seq_len(n), function(i) maxentcpp::maxent_feature_eval(f1p, i),
           numeric(1)),
    vapply(seq_len(n), function(i) maxentcpp::maxent_feature_eval(f2p, i),
           numeric(1))
)

# --- dismo model ---
if (has_dismo) {
    env_raster <- raster::stack(paths["bio1"], paths["bio2"])
    occ_coords <- occ[, c("lon", "lat")]

    cat("Training dismo model...\n")
    dismo_model <- dismo::maxent(
        x      = env_raster,
        p      = occ_coords,
        silent = TRUE,
        args   = c("nowarnings", "notooltips",
                    "linear=true", "quadratic=false",
                    "hinge=false", "threshold=false",
                    "product=false",
                    "maximumiterations=500",
                    "convergencethreshold=1e-5",
                    "betamultiplier=1.0")
    )
}

# ---- Benchmark functions ----------------------------------------------------

bench_maxentcpp_predict <- function() {
    maxentcpp::maxent_predict_model(fs, feat_mat)
}

if (has_dismo) {
    bench_dismo_predict <- function() {
        dismo::predict(dismo_model, env_raster)
    }
}

# ---- Run benchmarks ---------------------------------------------------------

cat("\nDataset: mock 10x10 (100 cells, 2 variables)\n")
cat("Settings: linear features, pre-trained models\n\n")

n_reps <- 20L

if (has_dismo) {
    cat(sprintf("Running %d replications (maxentcpp + dismo predict)...\n",
                n_reps))
    results <- microbenchmark::microbenchmark(
        maxentcpp = bench_maxentcpp_predict(),
        dismo     = bench_dismo_predict(),
        times     = n_reps,
        setup     = gc(verbose = FALSE)
    )
} else {
    cat("dismo/rJava not available; benchmarking maxentcpp only.\n")
    cat(sprintf("Running %d replications...\n", n_reps))
    results <- microbenchmark::microbenchmark(
        maxentcpp = bench_maxentcpp_predict(),
        times     = n_reps,
        setup     = gc(verbose = FALSE)
    )
}

# ---- Report -----------------------------------------------------------------

cat("\n--- Prediction Benchmark Results ---\n\n")
print(results)

cat("\n--- Summary (milliseconds) ---\n\n")
summary_df <- summary(results)
summary_df$median_ms <- summary_df$median / 1e6
summary_df$mean_ms   <- summary_df$mean / 1e6
print(summary_df[, c("expr", "median_ms", "mean_ms", "neval")])

if (has_dismo && nrow(summary_df) == 2) {
    cpp_median <- summary_df$median_ms[summary_df$expr == "maxentcpp"]
    java_median <- summary_df$median_ms[summary_df$expr == "dismo"]
    ratio <- cpp_median / java_median
    cat(sprintf("\nmaxentcpp/dismo ratio: %.3f (< 1.0 = maxentcpp faster)\n",
                ratio))
}

# ---- Numerical verification -------------------------------------------------

cat("\n--- Numerical Verification ---\n\n")
cpp_preds <- maxentcpp::maxent_predict_model(fs, feat_mat)
cat(sprintf("maxentcpp predictions: min=%.6f, max=%.6f, mean=%.6f\n",
            min(cpp_preds), max(cpp_preds), mean(cpp_preds)))

if (has_dismo) {
    dismo_preds <- as.numeric(raster::values(dismo::predict(dismo_model,
                                                             env_raster)))
    cat(sprintf("dismo predictions:     min=%.6f, max=%.6f, mean=%.6f\n",
                min(dismo_preds, na.rm = TRUE),
                max(dismo_preds, na.rm = TRUE),
                mean(dismo_preds, na.rm = TRUE)))

    cmp <- maxentcppCompTest::compare_maxent_predictions(cpp_preds,
                                                          dismo_preds)
    cat(sprintf("\nSpearman correlation: %.6f\n", cmp$spearman_cor))
    cat(sprintf("Pearson correlation:  %.6f\n", cmp$pearson_cor))
    cat(sprintf("Max rank difference:  %.6f\n", cmp$max_abs_diff_rank))
    cat(sprintf("Agreement (rho >= 0.95): %s\n",
                ifelse(cmp$agreement, "YES", "NO")))
}

cat("\nPrediction benchmark complete.\n")
