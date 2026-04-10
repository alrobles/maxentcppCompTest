# ROADMAP — maxentcppCompTest

## Overview

`maxentcppCompTest` is an R testing package whose sole purpose is to validate
that the C++ reimplementation of MaxEnt (`maxentcpp`, living in
[alrobles/Maxent/R-package](https://github.com/alrobles/Maxent/tree/main/R-package))
produces predictions that are equivalent to those of the original Java-based
MaxEnt software (used via the `dismo` R package).

---

## Repository Layout

```
maxentcppCompTest/
├── DESCRIPTION                        # R package metadata
├── NAMESPACE                          # Exported symbols
├── LICENSE                            # MIT licence
├── ROADMAP.md                         # This file
├── R/
│   ├── mock_data.R                    # mock_raster_paths(), mock_occurrences()
│   ├── run_maxent.R                   # run_maxentcpp(), run_maxent_java()
│   └── compare.R                      # compare_maxent_predictions()
├── inst/
│   ├── extdata/
│   │   ├── bio1.asc                   # 10×10 temperature raster (°C-like)
│   │   ├── bio2.asc                   # 10×10 precipitation raster (mm-like)
│   │   └── occurrences.csv            # 10 mock occurrence records
│   └── java/
│       └── com/macfaq/io/
│           ├── LittleEndianInputStream.java   # Migrated from alrobles/Maxent
│           └── LittleEndianOutputStream.java  # Migrated from alrobles/Maxent
├── tests/
│   ├── testthat.R
│   └── testthat/
│       ├── test-mock-data.R           # Validate bundled data files
│       ├── test-maxentcpp.R           # Test C++ implementation
│       ├── test-maxent-java.R         # Test Java implementation (dismo)
│       └── test-comparison.R          # Cross-implementation comparison
└── .github/
    └── workflows/
        └── R-CMD-check.yml            # CI on push / PR (Ubuntu, macOS, Windows)
```

---

## Mock Data Specification

### Environmental Rasters (`inst/extdata/`)

Both rasters share the same spatial header:

| Parameter      | Value  |
|----------------|--------|
| `ncols`        | 10     |
| `nrows`        | 10     |
| `xllcorner`    | 0.0    |
| `yllcorner`    | 0.0    |
| `cellsize`     | 1.0    |
| `NODATA_value` | -9999  |

Extent: longitude 0–10, latitude 0–10.

* **bio1.asc** – temperature proxy.  
  Cell value at ASC row *r*, column *c* (0-based):  
  `10 + 2*r + c`  (range 10–37).

* **bio2.asc** – precipitation proxy.  
  Cell value: `100 + 10*r + 50*c`  (range 100–640).

### Occurrence Records (`inst/extdata/occurrences.csv`)

Ten diagonal occurrence points, one per raster cell on the main diagonal:

| species | lon | lat |
|---------|-----|-----|
| sp1 | 0.5 | 9.5 |
| sp1 | 1.5 | 8.5 |
| … | … | … |
| sp1 | 9.5 | 0.5 |

---

## Implementation Phases

### Phase 1 — Package Skeleton ✅

- [x] `DESCRIPTION`, `NAMESPACE`, `LICENSE`
- [x] `.github/workflows/R-CMD-check.yml` (multi-OS CI)
- [x] `ROADMAP.md`

### Phase 2 — Mock Data ✅

- [x] `inst/extdata/bio1.asc` (10×10 temperature raster)
- [x] `inst/extdata/bio2.asc` (10×10 precipitation raster)
- [x] `inst/extdata/occurrences.csv` (10 occurrence records)

### Phase 3 — Java Code Migration ✅

- [x] `inst/java/com/macfaq/io/LittleEndianInputStream.java`
  (migrated from [alrobles/Maxent](https://github.com/alrobles/Maxent))
- [x] `inst/java/com/macfaq/io/LittleEndianOutputStream.java`
  (migrated from [alrobles/Maxent](https://github.com/alrobles/Maxent))

### Phase 4 — R Helper Functions ✅

- [x] `R/mock_data.R` — `mock_raster_paths()`, `mock_occurrences()`
- [x] `R/run_maxent.R` — `run_maxentcpp()`, `run_maxent_java()`
- [x] `R/compare.R` — `compare_maxent_predictions()`

### Phase 5 — Tests ✅

- [x] `tests/testthat/test-mock-data.R` — data file existence and content
- [x] `tests/testthat/test-maxentcpp.R` — C++ implementation unit tests
- [x] `tests/testthat/test-maxent-java.R` — Java implementation unit tests
- [x] `tests/testthat/test-comparison.R` — cross-implementation agreement

### Phase 6 — CI/CD ✅

- [x] GitHub Actions workflow running R CMD CHECK on Ubuntu, macOS, Windows
- [x] Automatic install of `maxentcpp` from `alrobles/Maxent/R-package`

---

## How to Use

### Prerequisites

```r
# Core package
install.packages(c("terra", "testthat"))

# C++ MaxEnt (requires a C++17 compiler)
remotes::install_github("alrobles/Maxent", subdir = "R-package")

# Java MaxEnt via dismo (optional — needed for comparison tests)
install.packages(c("dismo", "rJava", "raster"))
# Then download maxent.jar from
# https://biodiversityinformatics.amnh.org/open_source/maxent/
# and place it in file.path(system.file(package = "dismo"), "java")
```

### Install this package

```r
remotes::install_github("alrobles/maxentcppCompTest")
```

### Run tests

```r
# From the package directory:
devtools::test()

# Or via R CMD CHECK:
rcmdcheck::rcmdcheck()
```

### Ad-hoc comparison

```r
library(maxentcppCompTest)

# C++ predictions
cpp_result  <- run_maxentcpp()

# Java predictions (requires dismo + maxent.jar)
java_result <- run_maxent_java()

# Compare
cmp <- compare_maxent_predictions(
    cpp_result$predictions,
    java_result$predictions
)

cat("Spearman correlation:", round(cmp$spearman_cor, 4), "\n")
cat("Agreement (≥ 0.95):", cmp$agreement, "\n")
```

---

## Comparison Methodology

The two implementations can differ in absolute scale because they use
different output transformations (raw Gibbs distribution vs. cloglog/logistic).
Ordinal agreement is therefore assessed via **Spearman rank correlation** on
the 100 background-cell predictions.

The acceptance threshold is **ρ ≥ 0.95** — if both implementations rank the
same cells as more or less suitable, the comparison passes.

---

## References

* Phillips, S. J., Dudík, M. & Schapire, R. E. (2004).
  A maximum entropy approach to species distribution modeling.
  *ICML 2004*, 655–662.
* Hijmans, R. J. et al. `dismo` R package.
  <https://CRAN.R-project.org/package=dismo>
* Harold, E. R. (1999). `com.macfaq.io` Java I/O utilities.
  Bundled in the original Maxent source (see `inst/java/`).
