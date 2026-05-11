# MaxEntCppCompTest: Cross-Language Validation Suite for maxentcpp

Companion testing package for
[maxentcpp](https://github.com/alrobles/maxentcpp), a C++17
reimplementation of the MaxEnt species distribution modeling algorithm.
This package validates that `maxentcpp` produces numerically equivalent
results to the original Java MaxEnt (Phillips et al., 2006) through
systematic cross-language comparison.

## Background: Legacy Software in Computational Ecology

Computational ecology relies on a small number of foundational software
tools — many of which were written over a decade ago and have not been
modernized. MaxEnt (Phillips et al., 2006) is the most widely used
species distribution modeling algorithm, with over 10,000 citations. Yet
its sole production implementation remains a Java binary (`maxent.jar`)
that is accessed from R through fragile bridges: `dismo::maxent()` via
rJava, or command-line invocation via `system2("java -jar maxent.jar")`.

Reimplementing legacy scientific software in modern languages is rare in
ecology but has important precedents in adjacent fields:

- **Circuitscape** (Python → Julia, Hall et al. 2021): Landscape
  connectivity modeling tool rewritten for order-of-magnitude speedups.
- **LANDIS-II** (C → C#, Scheller et al. 2009): Forest landscape model
  re-engineered with modular architecture and automated testing.
- **WaterGAP** (Fortran → Python, Nyenah et al. 2025): Global
  hydrological model reprogrammed with modern software engineering
  practices and quantitative validation against the original.
- **maxnet** (Java → R/glmnet, Phillips 2017): The original MaxEnt author's
  own reimplementation using `glmnet` coordinate descent — same
  statistical model but a different optimization algorithm.

`maxentcpp` goes further than previous reimplementation efforts by
**porting the original Sequential optimizer algorithm to C++ and
validating per-iteration numerical equivalence** against the Java
reference. This combination — faithful algorithm port, per-iteration
trajectory comparison, and quantitative fidelity testing — sets a new
standard for software quality in computational ecology.

To our knowledge, `maxentcpp` is the first project to systematically
reimplement a widely-used SDM algorithm in a compiled language with
per-iteration numerical validation against the original implementation.

## Objective

Ensure that `maxentcpp` produces results equivalent to the original Java
MaxEnt at every level of the algorithm:

| Level | Tolerance | What Is Compared |
|-------|-----------|------------------|
| Linear features | 1e-14 | Feature evaluation: manual R vs Java vs C++ |
| Golden files | 1e-12 | Scalars, lambdas, density against committed CSVs |
| Optimizer trajectory | 1e-9 | Per-iteration lambda gap (asymmetric fixture, 500 iterations) |
| Sequential port | 1e-6 | `maxent_sequential_fit()` vs Java oracle at 10 checkpoints |
| End-to-end workflow | Spearman ≥ 0.95 | maxentcpp vs dismo rank-normalized predictions |

All tolerances are far below any ecologically meaningful threshold.
Environmental measurement error is O(0.1)–O(1.0) for temperature and
O(1)–O(10) for precipitation; model uncertainty from occurrence data
quality dwarfs 1e-6 differences by many orders of magnitude.

## Test Structure

```
tests/testthat/
├── test-linear-features.R       # Feature normalization: 3-way agreement
├── test-sample-stats.R          # Sample expectation, deviation, beta
├── test-optimizer.R             # Loss, entropy, lambdas, goodAlpha
├── test-density.R               # Normalized density, raw predictions
├── test-cloglog.R               # Cloglog transform, monotonicity
├── test-golden-java.R           # Phase A: Oracle determinism (1e-12)
├── test-optimizer-trajectory.R  # Phase B: Per-iteration gap quantification
├── test-sequential-stepwise.R   # Phase C: Sequential port (1e-6 gate)
├── test-comparison.R            # Rank-normalized Spearman (≥ 0.95)
├── test-mock-data.R             # Data fixture validation
├── test-maxentcpp.R             # run_maxentcpp() structure tests
├── test-maxent-java.R           # run_maxent_java() structure tests
└── helper-java-bridge.R         # Java oracle interface utilities
```

**59 test expectations** across **13 files** (1,621 lines of test code).

## Fixtures

| Fixture | Grid | Variables | Samples | Signal | Purpose |
|---------|------|-----------|---------|--------|---------|
| Symmetric | 10×10 | 2 | 10 (diagonal) | None | Infrastructure validation (λ → 0) |
| Asymmetric | 10×10 | 2 | 10 (clustered) | Strong | Optimizer divergence; validates Sequential port |

Golden baselines:
```
inst/extdata/golden/
├── scalars.csv
├── lambdas.csv
├── density.csv
├── trajectory_loss.csv
├── trajectory_entropy.csv
├── trajectory_lambdas.csv
└── asym/
    └── (same structure for asymmetric fixture)
```

## How to Run

```r
# Install dependencies
install.packages(c("testthat", "rJava"))
remotes::install_github("alrobles/maxentcpp")

# Run the full test suite
devtools::test()

# Run a specific test battery
testthat::test_file("tests/testthat/test-sequential-stepwise.R")
```

**Requirements:** R ≥ 4.0, Java ≥ 8 (for the Java oracle), maxentcpp,
rJava, dismo (for full-workflow comparison).

## Documentation

Detailed analysis documents are in `docs/`:

- [**TEST_COVERAGE_REPORT.md**](docs/TEST_COVERAGE_REPORT.md) — Full
  inventory of test coverage, tolerance hierarchy, gap analysis
- [**ECOSYSTEM_COMPARISON.md**](docs/ECOSYSTEM_COMPARISON.md) — How
  kuenm, kuenm2, wallace, ENMTools, and biomod2 use MaxEnt; maxentcpp's
  unique position in the ecosystem
- [**BENCHMARK_DESIGN.md**](docs/BENCHMARK_DESIGN.md) — Design for
  maxentcpp vs dismo performance profiling
- [**DISMO_INTEGRATION_DESIGN.md**](docs/DISMO_INTEGRATION_DESIGN.md) —
  Strategy for dual-backend dismo fork integration
- [**FIDELITY_BASELINE_REPORT.md**](docs/FIDELITY_BASELINE_REPORT.md) —
  Phase B quantitative results (C++ vs Java vs MaxentMini)

## Related Repositories

| Repository | Role |
|------------|------|
| [maxentcpp](https://github.com/alrobles/maxentcpp) | C++17 MaxEnt implementation (R package) |
| [Maxent](https://github.com/alrobles/Maxent) | Java MaxEnt fork (oracle reference) |
| [maxentcpp-devel](https://github.com/alrobles/maxentcpp-devel) | Development history (Java → C++ translation) |
| [maxentcpp-paper](https://github.com/alrobles/maxentcpp-paper) | JOSS paper manuscript |
| [dismo](https://github.com/alrobles/dismo) | dismo fork for side-by-side testing |

## References

- Phillips, S.J., Anderson, R.P., & Schapire, R.E. (2006). Maximum
  entropy modeling of species geographic distributions. *Ecological
  Modelling*, 190(3-4), 231-259.
- Hall, K.R. et al. (2021). Circuitscape in Julia: Empowering Dynamic
  Approaches to Connectivity Assessment. *Land*, 10(3), 301.
- Scheller, R.M. et al. (2009). Increasing the reliability of
  ecological models using modern software engineering techniques.
  *Frontiers in Ecology and the Environment*, 8(5), 253-260.
- Nyenah, E. et al. (2025). The process and value of reprogramming a
  legacy global hydrological model. *Geoscientific Model Development*,
  18, 5635-5653.

---

For issues and queries, please open an issue on this repository.
