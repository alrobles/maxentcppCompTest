# Benchmark Design — maxentcpp vs dismo Performance Comparison

**Date:** 2026-05-11
**Purpose:** Design a reproducible benchmark suite that measures maxentcpp speed
against dismo (Java Maxent) and identifies optimization opportunities.

---

## 1. What We Are Measuring

MaxEnt modeling has two computationally distinct phases:

| Phase | What Happens | dismo Path | maxentcpp Path |
|-------|-------------|------------|----------------|
| **Training** | Iterative optimizer fits lambdas to maximize regularized entropy | JVM → `density.Sequential.run()` | C++17 → `FeaturedSpace::train()` via Rcpp |
| **Prediction** | Apply trained model to new raster cells | JVM → `mebridge.predict()` → disk I/O | C++ → `maxent_predict_model()` or streaming `maxent_predict_raster()` |

Both phases should be benchmarked independently because:
- Training is CPU-bound (optimizer iterations)
- Prediction is I/O-bound for large rasters (disk read + memory allocation)

---

## 2. Benchmark Datasets

### 2.1 Mock Fixture (Baseline)
- **Source:** `inst/extdata/{bio1,bio2}.asc` + `occurrences.csv`
- **Size:** 10×10 = 100 cells, 2 variables, 10 samples
- **Purpose:** Sanity check; too small for meaningful timing but ensures both
  implementations produce equivalent output before scaling up.

### 2.2 Bioclim-19 Medium (Recommended)
- **Source:** WorldClim 2.1 bioclimatic variables at 10-arc-minute resolution
  for a single continent (e.g., South America)
- **Size:** ~50,000–200,000 non-NA cells, 19 variables, ~500 occurrence records
- **Purpose:** Representative of a real SDM study. Large enough to expose
  startup overhead and memory differences.
- **Preparation:** Download via `geodata::worldclim_global()` or bundle a
  pre-cropped subset in `inst/extdata/benchmark/`.

### 2.3 High-Resolution Stress Test (Optional)
- **Source:** WorldClim at 2.5-arc-minute resolution, global extent
- **Size:** ~4 million non-NA cells, 19 variables
- **Purpose:** Expose memory limitations. dismo loads the full raster into
  Java heap; maxentcpp uses streaming evaluation.
- **Note:** This test may require increasing JVM heap (`-Xmx4g`) for dismo.

---

## 3. Benchmark Protocol

### 3.1 Training Benchmark

```r
# Pseudocode — to be implemented in inst/benchmarks/benchmark_training.R

library(microbenchmark)

# Prepare data once (outside timing loop)
env_stack <- terra::rast(bioclim_paths)
occ <- read.csv(occurrence_path)

# --- dismo path ---
bench_dismo_train <- function() {
    dismo::maxent(raster::stack(env_stack), p = occ[, c("lon", "lat")],
                  args = c("linear=true", "quadratic=false",
                           "hinge=false", "threshold=false",
                           "product=false",
                           "maximumiterations=500",
                           "convergencethreshold=1e-5"))
}

# --- maxentcpp path ---
bench_maxentcpp_train <- function() {
    maxentcpp::maxent_run(env_stack, occ,
                          features = "linear",
                          max_iter = 500L,
                          convergence = 1e-5)
}

results <- microbenchmark(
    dismo    = bench_dismo_train(),
    maxentcpp = bench_maxentcpp_train(),
    times    = 20L,
    setup    = gc()
)
```

**Key controls:**
- Same feature types (linear only, then linear+quadratic, etc.)
- Same iteration count and convergence threshold
- Same regularization multiplier
- `gc()` between runs to minimize GC interference
- 20 replications for stable timing

### 3.2 Prediction Benchmark

```r
# Pseudocode — to be implemented in inst/benchmarks/benchmark_prediction.R

# After training both models on the same data:

bench_dismo_predict <- function() {
    dismo::predict(dismo_model, raster::stack(env_stack))
}

bench_maxentcpp_predict <- function() {
    maxentcpp::maxent_predict_raster(maxentcpp_model, env_stack)
}

results <- microbenchmark(
    dismo    = bench_dismo_predict(),
    maxentcpp = bench_maxentcpp_predict(),
    times    = 20L,
    setup    = gc()
)
```

### 3.3 Profiling (Identify Hot Spots)

```r
# R-level profiling
Rprof("maxentcpp_train.out", interval = 0.01)
maxentcpp::maxent_run(env_stack, occ, ...)
Rprof(NULL)
summaryRprof("maxentcpp_train.out")

# C++ level: compile maxentcpp with -pg or use perf
# perf record -g Rscript inst/benchmarks/benchmark_training.R
# perf report
```

---

## 4. Expected Performance Characteristics

### 4.1 Where maxentcpp Should Be Faster

| Component | Why |
|-----------|-----|
| JVM startup | dismo pays ~1–3 seconds for JVM initialization on first call; maxentcpp has zero startup overhead (shared library already loaded) |
| Data transfer | dismo writes CSV files to disk, Java reads them back; maxentcpp passes pointers via Rcpp (zero-copy) |
| Training loop | C++ Sequential optimizer is compiled ahead of time; Java relies on JIT compilation (warm-up period) |
| Prediction (streaming) | maxentcpp evaluates raster blocks without loading the full grid; dismo loads the entire raster into Java heap |
| Memory | maxentcpp uses Eigen dense matrices with RAII; Java's garbage collector adds latency spikes |

### 4.2 Where dismo Might Be Competitive

| Component | Why |
|-----------|-----|
| Hot-loop optimizer (after JVM warm-up) | HotSpot JIT can produce very efficient native code for tight loops; C++ advantage may be modest after warm-up |
| Feature generation | Java Maxent generates quadratic/hinge/threshold features internally; maxentcpp currently only has linear features |
| Batch mode | `kuenm`-style batch calibration (`system2("java -jar maxent.jar")`) avoids rJava overhead; one JVM process handles many models |

### 4.3 Performance Targets

| Metric | Target | Rationale |
|--------|--------|-----------|
| Training (linear, 100 cells) | ≤ dismo | Baseline sanity |
| Training (linear, 50k cells) | ≤ 0.5× dismo | C++ should beat Java on medium grids |
| Prediction (50k cells) | ≤ 0.3× dismo | Streaming eliminates Java heap pressure |
| JVM startup avoidance | 1–3 seconds saved | Measurable on short runs |
| Memory (high-res) | ≤ 0.5× dismo peak RSS | Streaming vs. full-grid allocation |

---

## 5. Optimization Opportunities in maxentcpp

Based on code review of `maxentcpp/src/cpp/include/maxent/`:

### 5.1 Already Optimized
- Eigen for linear algebra (BLAS-backed)
- Row-major storage matching raster layout
- RAII memory management (no leaks)
- Block-wise streaming prediction

### 5.2 Potential Improvements

| Opportunity | Impact | Effort | Notes |
|-------------|--------|--------|-------|
| OpenMP parallelize `goodAlpha` loop over features | Medium | Low | Each feature's alpha is independent within an iteration |
| SIMD-friendly density normalization | Medium | Medium | `exp()` over large vectors; could use `Eigen::Array::exp()` |
| Cache-aware feature evaluation order | Low | Medium | Ensure features stored contiguously in memory |
| Lazy feature generation | Medium | Medium | Generate quadratic features on-demand rather than materializing the full matrix |
| Pre-allocated trajectory buffers | Low | Low | Avoid reallocation during trajectory capture |

### 5.3 What NOT to Optimize

- **The optimizer's mathematical logic.** Changing `goodAlpha` or `newtonStep`
  to be "faster" would break numerical fidelity to Java.
- **Convergence criteria.** Must match Java exactly.
- **Feature normalization.** Must produce identical values to Java.

---

## 6. Reporting

The benchmark results should be reported as:

1. **Table:** Median time (ms) with IQR for each implementation × dataset × phase
2. **Ratio:** maxentcpp/dismo (values < 1.0 mean maxentcpp is faster)
3. **Plot:** Violin plot of timing distributions (shows JVM warm-up variability)
4. **Memory:** Peak RSS for prediction on the high-res dataset
5. **Numerical verification:** Confirm that both produce equivalent output
   (Spearman ≥ 0.99 on predictions) for every benchmark run

Template for the results table:

| Dataset | Phase | dismo (ms) | maxentcpp (ms) | Ratio | Spearman |
|---------|-------|-----------|---------------|-------|----------|
| Mock (100 cells) | Train | — | — | — | — |
| Mock (100 cells) | Predict | — | — | — | — |
| Bioclim-19 (50k) | Train | — | — | — | — |
| Bioclim-19 (50k) | Predict | — | — | — | — |

---

## 7. Implementation Plan

1. **Create `inst/benchmarks/` directory** in maxentcppCompTest
2. **`benchmark_training.R`** — Training comparison on mock + bioclim datasets
3. **`benchmark_prediction.R`** — Prediction comparison
4. **`benchmark_report.Rmd`** — R Markdown that runs benchmarks and generates
   tables + plots
5. **Add to CI** — Run on a fixed-hardware GitHub Actions runner for reproducibility
   (or document hardware specs for local runs)

The benchmark suite should be runnable with:
```r
source("inst/benchmarks/benchmark_training.R")
source("inst/benchmarks/benchmark_prediction.R")
# Or render the full report:
rmarkdown::render("inst/benchmarks/benchmark_report.Rmd")
```
