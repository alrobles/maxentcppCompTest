#!/usr/bin/env Rscript
# benchmark_training.R — Training benchmark: maxentcpp vs dismo (Java Maxent)
#
# Compares training time on the bundled 10x10 mock fixture.
# Requires: maxentcpp, microbenchmark
# Optional: dismo, rJava, raster (for Java comparison)
#
# Usage:
#   Rscript inst/benchmarks/benchmark_training.R
#   # Or from R:
#   source(system.file("benchmarks", "benchmark_training.R",
#                       package = "maxentcppCompTest"))

cat("=== maxentcpp Training Benchmark ===\n\n")

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

# ---- Prepare mock data (outside timing loop) --------------------------------

paths <- maxentcppCompTest::mock_raster_paths()
occ   <- maxentcppCompTest::mock_occurrences()

# maxentcpp data prep
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

# ---- Benchmark functions ----------------------------------------------------

bench_maxentcpp_train <- function() {
    f1 <- maxentcpp::maxent_linear_feature(bio1_vec, "bio1")
    f2 <- maxentcpp::maxent_linear_feature(bio2_vec, "bio2")
    fs <- maxentcpp::maxent_featured_space(n, occ_indices, list(f1, f2))
    maxentcpp::maxent_fit(fs, max_iter = 500L, convergence = 1e-5,
                          beta_multiplier = 1.0)
}

if (has_dismo) {
    env_raster <- raster::stack(paths["bio1"], paths["bio2"])
    occ_coords <- occ[, c("lon", "lat")]

    bench_dismo_train <- function() {
        dismo::maxent(
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
}

# ---- Run benchmarks ---------------------------------------------------------

cat("Dataset: mock 10x10 (100 cells, 2 variables, 10 samples)\n")
cat("Settings: linear features, 500 max iterations, convergence 1e-5\n\n")

n_reps <- 20L

if (has_dismo) {
    # Warm up JVM
    cat("Warming up JVM...\n")
    suppressMessages(try(bench_dismo_train(), silent = TRUE))

    cat(sprintf("Running %d replications (maxentcpp + dismo)...\n", n_reps))
    results <- microbenchmark::microbenchmark(
        maxentcpp = bench_maxentcpp_train(),
        dismo     = bench_dismo_train(),
        times     = n_reps,
        setup     = gc(verbose = FALSE)
    )
} else {
    cat("dismo/rJava not available; benchmarking maxentcpp only.\n")
    cat(sprintf("Running %d replications...\n", n_reps))
    results <- microbenchmark::microbenchmark(
        maxentcpp = bench_maxentcpp_train(),
        times     = n_reps,
        setup     = gc(verbose = FALSE)
    )
}

# ---- Report -----------------------------------------------------------------

cat("\n--- Training Benchmark Results ---\n\n")
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
    cat(sprintf("maxentcpp median: %.2f ms\n", cpp_median))
    cat(sprintf("dismo median:     %.2f ms\n", java_median))
}

# ---- Asymmetric fixture benchmark -------------------------------------------

cat("\n=== Asymmetric Fixture (non-trivial lambdas) ===\n\n")

paths_asym <- maxentcppCompTest::mock_raster_paths_asym()
occ_asym   <- maxentcppCompTest::mock_occurrences_asym()

g1a <- maxentcpp::maxent_read_asc(paths_asym["bio1"])
g2a <- maxentcpp::maxent_read_asc(paths_asym["bio2"])
m1a <- maxentcpp::maxent_grid_to_matrix(g1a)
m2a <- maxentcpp::maxent_grid_to_matrix(g2a)

bio1_asym <- as.numeric(t(m1a))
bio2_asym <- as.numeric(t(m2a))

occ_indices_asym <- vapply(seq_len(nrow(occ_asym)), function(i) {
    row_i <- floor((info$yll + info$nrows * info$cellsize - occ_asym$lat[i]) /
                       info$cellsize)
    col_i <- floor((occ_asym$lon[i] - info$xll) / info$cellsize)
    row_i <- max(0L, min(as.integer(row_i), info$nrows - 1L))
    col_i <- max(0L, min(as.integer(col_i), info$ncols - 1L))
    as.integer(row_i * info$ncols + col_i)
}, integer(1L))

bench_maxentcpp_train_asym <- function() {
    f1 <- maxentcpp::maxent_linear_feature(bio1_asym, "bio1")
    f2 <- maxentcpp::maxent_linear_feature(bio2_asym, "bio2")
    fs <- maxentcpp::maxent_featured_space(n, occ_indices_asym, list(f1, f2))
    maxentcpp::maxent_fit(fs, max_iter = 500L, convergence = 1e-5,
                          beta_multiplier = 1.0)
}

cat(sprintf("Running %d replications (asymmetric fixture)...\n", n_reps))
results_asym <- microbenchmark::microbenchmark(
    maxentcpp_asym = bench_maxentcpp_train_asym(),
    times = n_reps,
    setup = gc(verbose = FALSE)
)

cat("\n--- Asymmetric Training Results ---\n\n")
print(results_asym)

cat("\nBenchmark complete.\n")
