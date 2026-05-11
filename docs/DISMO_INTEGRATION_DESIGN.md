# dismo Integration Design — Side-by-Side Testing Strategy

**Date:** 2026-05-11
**Purpose:** Design how `alrobles/dismo` (fork of `rspatial/dismo`) will integrate
with `maxentcpp` and `maxentcppCompTest` for validated side-by-side comparison.

---

## 1. Current State

### alrobles/dismo (fork)
- Forked from `rspatial/dismo` on 2026-04-17
- Unmodified — identical to upstream
- Contains the standard `maxent()` S4 method that bridges to Java via rJava

### rspatial/dismo (upstream)
- Low maintenance activity (last meaningful update: 2023)
- `dismo::maxent()` is the de facto standard way to call Java MaxEnt from R
- Depends on: rJava, raster (legacy, not terra)
- Users must manually download `maxent.jar` and place it in the package directory

---

## 2. Integration Strategy

### Option A: Dual-Backend maxent() (Recommended)

Modify `alrobles/dismo`'s `maxent()` method to support an `engine` argument:

```r
maxent(x, p, ..., engine = c("java", "maxentcpp"))
```

When `engine = "java"`:
- Existing behavior unchanged (rJava bridge to maxent.jar)

When `engine = "maxentcpp"`:
- Calls `maxentcpp::maxent_run()` internally
- Returns a `MaxEnt` S4 object with the same structure as the Java version
- `predict()` dispatches to `maxentcpp::maxent_predict_raster()`

**Advantages:**
- Zero API change for existing users (default remains `"java"`)
- Drop-in comparison: `model_java <- maxent(x, p, engine="java")`
  vs. `model_cpp <- maxent(x, p, engine="maxentcpp")`
- Downstream packages (ENMTools, kuenm, etc.) work without modification
- Migration path: once validated, change default to `"maxentcpp"`

**Implementation effort:** ~200 lines of R code (new method dispatch + result
format conversion)

### Option B: Parallel Package (Alternative)

Create a separate `dismo.cpp` or `maxentdismo` package that provides the same S4
classes and methods but backed by maxentcpp.

**Disadvantages:** Fragmented ecosystem, users must change `library()` calls.
Not recommended.

---

## 3. S4 Class Compatibility

The dismo `MaxEnt` class has these slots:

```r
setClass('MaxEnt',
    contains = 'DistModel',
    representation(
        lambdas  = 'vector',     # Lambda string (feature weights)
        results  = 'matrix',     # Training results (gain, AUC, etc.)
        path     = 'character',  # Output directory path
        html     = 'character'   # HTML report path
    )
)
```

For the `engine = "maxentcpp"` path, we need to populate these slots from
maxentcpp outputs:

| Slot | Source from maxentcpp | Notes |
|------|---------------------|-------|
| `lambdas` | `maxent_model_lambdas()` → format as dismo lambda strings | Must match dismo format: `"feature_name, lambda, min, max"` |
| `results` | Construct from loss, entropy, AUC, etc. | Matrix with rownames matching dismo convention |
| `path` | Temporary directory (or NA) | maxentcpp doesn't write to disk by default |
| `html` | NA or generate minimal HTML | maxentcpp has `maxent_report()` |

The `predict()` method for `MaxEnt` objects uses `maxent.predict.R`. For the
C++ backend, we need to override or extend this to call maxentcpp's prediction
functions instead of Java.

---

## 4. Side-by-Side Test Harness

### 4.1 Test Structure

Add to `maxentcppCompTest/tests/testthat/`:

```
test-dismo-comparison.R          # Full dismo workflow comparison
test-dismo-lambda-compat.R       # Lambda string format compatibility
test-dismo-predict-compat.R      # predict() output format compatibility
```

### 4.2 Test: Lambda String Compatibility

```r
test_that("maxentcpp lambdas match dismo lambda format", {
    skip_if_not_installed("maxentcpp")
    skip_if_not_installed("dismo")

    # Train both
    cpp_model  <- maxentcpp::maxent_run(env, occ, ...)
    java_model <- dismo::maxent(raster::stack(env), p = occ, ...)

    # Parse dismo lambda strings
    java_lambdas <- parse_dismo_lambdas(java_model@lambdas)
    cpp_lambdas  <- maxentcpp_to_dismo_lambdas(cpp_model)

    # Compare feature weights
    expect_equal(cpp_lambdas$lambda, java_lambdas$lambda,
                 tolerance = 1e-4)  # Relaxed for default settings differences
})
```

### 4.3 Test: Prediction Map Equivalence

```r
test_that("maxentcpp and dismo produce equivalent prediction rasters", {
    skip_if_not_installed("maxentcpp")
    skip_if_not_installed("dismo")

    # Both predict on the same raster stack
    cpp_pred  <- maxentcpp::maxent_predict_raster(cpp_model, env)
    java_pred <- dismo::predict(java_model, raster::stack(env))

    # Convert to comparable vectors
    cpp_vals  <- terra::values(cpp_pred)[!is.na(terra::values(cpp_pred))]
    java_vals <- raster::values(java_pred)[!is.na(raster::values(java_pred))]

    # Rank correlation
    expect_true(cor(cpp_vals, java_vals, method = "spearman") >= 0.99)

    # Absolute difference
    expect_true(max(abs(cpp_vals - java_vals)) < 0.05)
})
```

### 4.4 Test: Full Ecological Workflow

```r
test_that("maxentcpp and dismo produce same ecological conclusions", {
    # Train both on same data
    # Predict both on same raster
    # Threshold both at same quantile
    # Compare binary maps (Jaccard similarity >= 0.95)
    # Compare AUC on same test set (difference < 0.02)
    # Compare variable importance ranking (Kendall tau >= 0.90)
})
```

---

## 5. Conversion Utilities

The dismo fork needs utility functions to bridge between maxentcpp and dismo
data formats:

### 5.1 Lambda Format Converter

```r
#' Convert maxentcpp lambdas to dismo lambda string format
#'
#' @param model A trained maxentcpp model object
#' @return Character vector in dismo lambda format
maxentcpp_to_dismo_lambdas <- function(model) {
    # dismo format: "feature_name, lambda, min, max\n"
    # One line per feature, followed by metadata lines
    lambdas   <- model$lambdas
    feat_info <- model$feature_info
    paste0(feat_info$name, ", ",
           format(lambdas, scientific = TRUE), ", ",
           format(feat_info$min, scientific = TRUE), ", ",
           format(feat_info$max, scientific = TRUE))
}
```

### 5.2 Raster Prediction Adapter

```r
#' Predict using maxentcpp but return a terra/raster object
#'
#' @param model MaxEnt S4 object with maxentcpp backend
#' @param x SpatRaster or RasterStack
#' @return SpatRaster with cloglog predictions
predict_maxentcpp_raster <- function(model, x) {
    if (inherits(x, "RasterStack")) {
        x <- terra::rast(x)  # Convert legacy raster to terra
    }
    maxentcpp::maxent_predict_raster(model$cpp_model, x,
                                      output_type = "cloglog")
}
```

---

## 6. Implementation Roadmap

### Phase 1: Test Infrastructure (in maxentcppCompTest)
1. Add `test-dismo-comparison.R` with side-by-side tests
2. Add lambda format parsing utilities to `R/compare.R`
3. Ensure CI can install both maxentcpp and dismo with Java

### Phase 2: dismo Fork Modifications (in alrobles/dismo)
1. Add `engine` argument to `maxent()` method
2. Implement maxentcpp backend dispatch
3. Add lambda format conversion
4. Ensure `predict()` works with both backends
5. Update `DESCRIPTION` with conditional dependency on maxentcpp

### Phase 3: Validation
1. Run full test suite with both engines
2. Compare predictions on real-world datasets (Bradypus, etc.)
3. Document any divergences and their causes
4. Benchmark performance comparison

### Phase 4: Downstream Integration
1. Test with ENMTools (should work with no changes if dismo API preserved)
2. Test with wallace/ENMeval (may need algorithm string registration)
3. Document migration guide for users

---

## 7. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| dismo uses `raster` (legacy), maxentcpp uses `terra` | Format mismatch in predict() | Convert via `terra::rast()` / `raster::raster()` as needed |
| Default feature types differ | Different models | Explicitly match feature arguments in comparison tests |
| dismo's `maxent()` has many hidden defaults | Unfair comparison | Parse dismo's actual arguments from the MaxEnt object |
| maxent.jar version differences | Different results | Pin to a specific maxent.jar version in CI |
| rJava installation failures on CI | Tests can't run | Use `skip_if` guards; ensure Java is in CI environment |
