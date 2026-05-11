# Test Coverage Report — maxentcppCompTest

**Date:** 2026-05-11
**Scope:** Assess whether `maxentcppCompTest` provides sufficient coverage for a
side-by-side comparison of `maxentcpp` against the original Java Maxent, down to
numerically acceptable tolerances for ecological applications.

---

## 1. Inventory of Test Files

| File | Tests | Lines | What It Covers |
|------|------:|------:|----------------|
| `test-linear-features.R` | 4 | 95 | Feature normalization: Java vs C++ vs manual R on bio1/bio2, constant-feature edge case, three-way 1e-14 agreement |
| `test-sample-stats.R` | 5 | 109 | Sample expectation, sample deviation (pop. std dev), beta (regularization) — Java vs R manual |
| `test-optimizer.R` | 7 | 223 | Initial loss = log(n), loss after 1/10/convergence iterations, lambda agreement, goodAlpha formula, reduceAlpha damping schedule |
| `test-density.R` | 3 | 89 | Density normalization (sum-to-1), raw predictions, Shannon entropy — C++ vs Java |
| `test-cloglog.R` | 5 | 126 | Cloglog formula correctness, output range [0,1], monotonicity, three-way agreement, end-to-end cloglog |
| `test-golden-java.R` | 2 | 126 | Golden-file reproducibility at 1e-12: scalars, lambdas, sample stats, density/raw/cloglog; determinism across repeated calls |
| `test-optimizer-trajectory.R` | 3 | 181 | Symmetric/asymmetric trajectory reproducibility at 1e-14; Phase B baseline gap quantification (C++ vs Java vs MaxentMini) |
| `test-sequential-stepwise.R` | 2 | 128 | `maxent_sequential_fit` matches Java oracle ≤ 1e-6 at 10 checkpoints (asymmetric + symmetric fixtures) |
| `test-comparison.R` | 6 | 87 | `compare_maxent_predictions()` unit tests + full workflow: maxentcpp vs dismo Java (Spearman ≥ 0.95) |
| `test-mock-data.R` | 10 | 118 | Data file existence, grid dimensions, cell counts, occurrence records, Java source files |
| `test-maxentcpp.R` | 8 | 81 | `run_maxentcpp()` structure, predictions (100 cells, finite, non-negative, varying), fit components |
| `test-maxent-java.R` | 4 | 61 | `run_maxent_java()` structure, predictions (100 cells, [0,1]), 10 samples |
| `helper-java-bridge.R` | — | 197 | Helper functions: Java JVM initialization, feature eval, training, density, cloglog, trajectory, golden-file fixtures |

**Totals:** 59 test expectations across 13 files, 1,621 lines of test code.

---

## 2. Tolerance Hierarchy

The test suite uses a well-structured hierarchy of numerical tolerances:

| Level | Tolerance | Where Used | Interpretation |
|-------|-----------|------------|----------------|
| Bit-identical | 0 (exact) | Determinism tests, symmetric fixture lambdas | Java produces identical output on repeated calls |
| Machine precision | 1e-14 | Linear feature eval, goodAlpha formula, golden trajectory replay | Algorithm-level correctness within IEEE 754 limits |
| Golden-file | 1e-12 | Scalars, lambdas, density, raw, cloglog against committed CSVs | Oracle reproducibility across builds |
| Sequential port | 1e-6 | `test-sequential-stepwise.R` at 10 checkpoints | **Primary fidelity gate:** C++ Sequential matches Java Sequential |
| End-to-end cloglog | 1e-6 | `test-cloglog.R` Test 22 | User-facing prediction agreement |
| Optimizer internals | 1e-8 | Loss/entropy/density after training (goodAlpha oracle) | Cross-implementation agreement on MaxentMini-level optimizer |
| Ecological workflow | 0.95 (Spearman) | `test-comparison.R` full workflow via dismo | Rank-order agreement through the full dismo→raster→predict pipeline |

**Assessment:** The tolerance hierarchy is sound. The 1e-6 gate on the Sequential
optimizer is the most meaningful: it proves that `maxentcpp` and real Java Maxent
converge to the same model parameters. The 0.95 Spearman threshold for the
dismo-level comparison is conservative — it accounts for differences in how dismo
wraps Java Maxent (default arguments, feature selection, etc.).

---

## 3. Fixture Coverage

### 3.1 Existing Fixtures

| Fixture | Grid | Vars | Samples | Signal | Purpose |
|---------|------|------|---------|--------|---------|
| Symmetric (Phase A) | 10×10 | 2 (bio1, bio2) | 10 (diagonal) | None (sample mean = background mean) | Baseline: all optimizers produce λ = (0,0) |
| Asymmetric (Phase B) | 10×10 | 2 (bio1, bio2) | 10 (clustered bottom-right) | Strong | Exercises optimizer divergence; validates Sequential port to 1e-6 |

### 3.2 Feature Type Coverage

| Feature Type | Tested? | Notes |
|--------------|---------|-------|
| Linear | **Yes** | Both fixtures, full trajectory validation |
| Quadratic | No | Listed as deferred in FIDELITY_BASELINE_REPORT §6 |
| Product | No | — |
| Hinge | No | C++ port in progress; deferred to Phase C |
| Threshold | No | — |
| Categorical | No | — |

### 3.3 Coverage Gaps

1. **Only linear features are tested.** Real-world Maxent runs use linear + quadratic +
   hinge + threshold by default. The current tests prove equivalence on the simplest
   feature class. Once `maxentcpp` supports quadratic/hinge/threshold features, the
   test suite must expand to cover them.

2. **Only 2 environmental variables.** Real analyses typically use 5–19 bioclimatic
   variables. A fixture with ≥3 variables would exercise the feature-selection path
   (`deltaLossBound` / `getBestFeature`) more thoroughly.

3. **No NODATA / missing-value fixture.** The mock rasters have no NODATA cells.
   Real-world rasters always have NODATA (ocean, clouds). A fixture with masked cells
   would test the NODATA-handling path.

4. **No clamping test.** Clamping restricts predictions to training ranges when
   projecting to novel environments. This is important for ecological transfer studies.

5. **No permutation-importance or variable-contribution test.** These are key outputs
   that ecologists use to interpret models.

6. **Small grid only (10×10 = 100 cells).** Performance and memory behavior on larger
   grids (e.g., 1000×1000) is not tested.

---

## 4. dismo Output Comparison Assessment

The current `test-comparison.R` "full workflow" test (Test line 61–86) compares
maxentcpp and dismo outputs through `run_maxentcpp()` and `run_maxent_java()`.

### What It Tests

- Prediction count equality (n_points, n_samples)
- Spearman rank correlation ≥ 0.95
- Agreement flag from `compare_maxent_predictions()`

### What It Does NOT Test

For a user-base building strategy, we need to verify that **user-facing outputs**
are comparable — not just internal model predictions. The following outputs matter
to ecologists and are not yet compared:

| Output | dismo Produces | maxentcpp Produces | Tested? |
|--------|----------------|-------------------|---------|
| Raw prediction map | Yes (raster) | Yes (vector, streamable) | Partial (vector only) |
| Cloglog prediction map | Yes | Yes | Yes (test-cloglog.R) |
| Logistic prediction map | Yes | ? | No |
| Lambda file (.lambdas) | Yes | Yes | Yes (golden files) |
| Response curves | Yes (HTML) | No built-in | No |
| Variable importance | Yes (permutation) | Yes | No |
| AUC / evaluation metrics | Yes | ? | No |
| MESS (novelty detection) | Yes (via dismo) | Yes (built-in) | No |
| HTML report | Yes | Not planned | N/A |

### Recommended Additions for Side-by-Side Comparison

1. **Lambda-file comparison:** Parse dismo's `.lambdas` text file and compare
   feature weights against maxentcpp's lambda vector. This is the most direct
   numerical comparison.

2. **Cloglog map comparison on raster:** Run both implementations on the same
   raster stack, produce cloglog output, and compare cell-by-cell.

3. **Variable importance comparison:** Both produce permutation importance;
   compare ranking and values.

4. **AUC comparison:** Both should produce similar AUC on the same train/test split.

---

## 5. Numerical Acceptability for Ecology

### What tolerance is "ecologically acceptable"?

In species distribution modeling, the primary use of MaxEnt output is:
1. **Ranking habitat suitability** (ordinal, not cardinal)
2. **Thresholding for presence/absence maps** (binary)
3. **Comparing models via AUC, TSS, etc.** (aggregate statistics)

For these uses:
- **Lambda agreement to 1e-6** (achieved) means the models learn the same
  coefficients → same predictions → same rankings → same ecological conclusions.
- **Cloglog agreement to 1e-6** (achieved) means probability maps are
  indistinguishable at the precision of the underlying environmental data (which
  is typically reported to 1–2 decimal places).
- **The current 1e-6 tolerance is well below any ecologically meaningful
  threshold.** Environmental measurement error is typically O(0.1)–O(1.0) for
  temperature and O(1)–O(10) for precipitation. Model uncertainty from occurrence
  data quality dwarfs 1e-6 numerical differences by many orders of magnitude.

**Conclusion:** The test suite already validates numerical parity to a level far
exceeding ecological requirements. The 1e-6 gate on Sequential optimizer
parameters is sufficient. For user-facing comparisons (dismo workflow), the 0.95
Spearman threshold is a reasonable floor but could be tightened to 0.99 once
default feature types and regularization settings match exactly.

---

## 6. Summary and Recommendations

### Strengths
- Rigorous multi-oracle comparison (Java Sequential, MaxentMini, C++)
- Golden-file regression prevents silent degradation
- Tolerance hierarchy is well-documented and justified
- Per-iteration trajectory comparison (not just final-state)
- Phase B baseline report is exemplary scientific documentation

### Gaps (Prioritized)
1. **[High]** No quadratic/hinge/threshold feature tests (blocked on C++ port)
2. **[High]** No ≥3-variable fixture (exercises feature selection)
3. **[Medium]** No raster-level output comparison against dismo
4. **[Medium]** No variable importance / AUC comparison
5. **[Low]** No NODATA / missing-value fixture
6. **[Low]** No clamping or transfer-projection test
7. **[Low]** No large-grid performance test

### Verdict

**The test suite provides sufficient coverage for validating the core MaxEnt
algorithm on linear features with 2 variables.** The 1e-6 tolerance on the
Sequential optimizer port is both achieved and ecologically more than adequate.
The asymmetric fixture was a critical addition that exposed the MaxentMini vs.
real-Java gap.

**For a production-ready side-by-side comparison with dismo** (the user's stated
goal), the suite needs expansion in two directions: (a) feature type coverage
(quadratic, hinge, threshold — dependent on C++ port progress) and (b) user-facing
output comparison (lambda files, raster maps, importance metrics).
