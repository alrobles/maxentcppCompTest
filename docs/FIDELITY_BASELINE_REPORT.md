# Fidelity Baseline Report — maxentcpp vs real Java Maxent

**Phase B of the audit plan for** `alrobles/maxentcpp#36` / `alrobles/maxentcpp#37`.
This report quantifies the per-iteration gap between three optimizers
on two mock fixtures so that Phase C (Sequential port) and Phase D
(1e-6 milestone) have a documented starting point.

All trajectory CSVs referenced here are committed under
`inst/extdata/golden/{,asym/}trajectory_{java,mini,cpp}.csv`.  They can
be regenerated with the commands at the end of this document.

---

## 1. Oracles under comparison

| Tag    | Implementation                                  | Optimizer semantics                                                                                                                                               |
|--------|-------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| java   | `density.Sequential` via `density.MaxentRefRunner` from the unmodified `alrobles/Maxent/density/*.java` source | Feature selection by `deltaLossBound`; one feature/iter; `newtonStep + searchAlpha` for continuous features; `doParallelUpdate` every 10 iters; loss-violation undo. |
| mini   | `density.MaxentMini` (the goodAlpha-only oracle already used by the existing `test-optimizer.R`, `test-density.R`, `test-cloglog.R` batteries in this repo) | Every iteration runs `goodAlpha` on each feature; no feature selection; no Newton step. |
| cpp    | `maxent::FeaturedSpace::train()` from `alrobles/maxentcpp/src/cpp/include/maxent/featured_space.hpp` | Currently also goodAlpha-only across all features (intentionally matches `MaxentMini`, not real `Sequential`). |

Trajectories for all three are captured at the same checkpoints —
`{1, 2, 3, 5, 10, 20, 50, 100, 200, 500}` — with convergence-based early
termination disabled (`convergence_threshold = 0`) so each oracle always
runs for exactly the requested iterations.

Capture surfaces:

- **java** — new `TrajectorySequential` inner class in
  `inst/java/MaxentRefRunner.java` that overrides `terminationTest` to
  snapshot `(loss, entropy, lambdas)` at each checkpoint while keeping
  the rest of `Sequential.run()` bit-identical.
- **mini** — `MaxentMini.trainNIterations(...)` called once per
  checkpoint from the new `--mini` entry point in `MaxentRefRunner.main`.
- **cpp**  — new standalone CLI `inst/cpp/trajectory_cpp.cpp`
  (links against `featured_space.cpp` directly, no Rcpp), which
  re-constructs a fresh `FeaturedSpace` for each checkpoint and
  trains for exactly *k* iterations.

## 2. Fixtures

### 2.1 Symmetric fixture (Phase A)

Files: `inst/extdata/{bio1,bio2}.asc`, `inst/extdata/occurrences.csv`.
10 occurrences on the grid diagonal.  After the Java
`(x - min) / (max - min)` scaling, sample means exactly equal
background means, so `goodAlpha` returns `-λ = 0` on both features.

### 2.2 Asymmetric fixture (Phase B, new)

Files: `inst/extdata/asym/{bio1,bio2}.asc`, `inst/extdata/asym/occurrences.csv`.
bio1 is a shifted gradient (values 10..37 row-major); bio2 is
the same as the symmetric fixture.  All 10 occurrences cluster in the
bottom-right quadrant (rows 0..4, columns 7..9).  Sample means differ
from background means after scaling, so all three optimizers produce
non-trivial lambdas.

## 3. Results

### 3.1 Symmetric fixture — all three oracles match to machine precision

| checkpoint | ‖λ_cpp−λ_java‖∞ | \|loss_cpp−loss_java\| | ‖λ_mini−λ_java‖∞ | \|loss_mini−loss_java\| |
|---:|---:|---:|---:|---:|
|   1 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
|   2 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
|   3 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
|   5 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
|  10 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
|  20 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
|  50 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 100 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 200 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 500 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 |

Final state at iter 500 (all three identical):
`λ = (0, 0)`, `loss = 4.60517018598809`, `entropy = 4.60517018598808`.

The symmetric fixture alone cannot distinguish the three optimizers —
confirming the Phase A finding that flagged the need for an asymmetric
signal.

### 3.2 Asymmetric fixture — C++ tracks real Java to 1e-9; MaxentMini diverges

| checkpoint | ‖λ_cpp−λ_java‖∞ | \|loss_cpp−loss_java\| | ‖λ_mini−λ_java‖∞ | \|loss_mini−loss_java\| |
|---:|---:|---:|---:|---:|
|   1 | 1.093e-01 | 2.678e-02 | 1.090e-01 | 3.851e-02 |
|   2 | 2.155e-01 | 5.132e-02 | 2.147e-01 | 7.447e-02 |
|   3 | 3.186e-01 | 7.373e-02 | 3.174e-01 | 1.080e-01 |
|   5 | 3.838e-01 | 1.131e-01 | 3.812e-01 | 1.688e-01 |
|  10 | 4.655e-01 | 1.925e-01 | 5.046e-01 | 2.973e-01 |
|  20 | 2.458e+00 | 4.942e-01 | 3.221e+00 | 8.176e-01 |
|  50 | 6.858e-01 | 3.285e-03 | 9.281e-01 | 1.183e-02 |
| 100 | 1.183e-02 | 1.072e-06 | 4.572e-02 | 1.298e-02 |
| 200 | 5.649e-06 | 2.496e-13 | 4.620e-02 | 1.298e-02 |
| **500** | **6.467e-09** | **2.220e-15** | **4.620e-02** | **1.298e-02** |

Final state at iter 500:

```
java: loss=3.170938928003121   entropy=3.170938927496662   λ=[5.879531356699762, 4.013518796542913]
cpp : loss=3.170938928003124   entropy=3.170938928003122   λ=[5.879531350232670, 4.013518800237437]
mini: loss=3.157961401537453   entropy=3.157961401537445   λ=[5.925727930235640, 4.043261746615105]
```

Summary:

- `‖λ_cpp  − λ_java‖∞ ≈ 6.5e-9`  (well inside the 1e-6 target).
- `‖λ_mini − λ_java‖∞ ≈ 4.6e-2`.
- `‖λ_cpp  − λ_mini‖∞ ≈ 4.6e-2`.

The mid-trajectory gap is much larger than the final gap (peaking at
`2.46` at iter 20 for `cpp` and `3.22` for `mini`).  That is expected:
real Java Sequential takes a single larger step per feature via
`newtonStep`, while MaxentMini and the current C++ port take many
smaller `goodAlpha` steps across all features; they converge later but
to essentially the same fixed point (for `cpp`) or to a slightly
different fixed point (for `mini`).

## 4. Interpretation

1. **C++ ≈ real Java at convergence, even with a simplified optimizer.**
   The C++ port's goodAlpha-only loop happens to reach (numerically) the
   same fixed point as real `Sequential` on this 2-feature fixture; the
   largest residual is ~6e-9 on lambdas and ~2e-15 on the regularized
   loss.  This is a stronger-than-expected result.  The gap originally
   predicted (O(1e-2) – O(1e-1) in the audit plan) was a worst-case
   scenario assuming the optimizers could converge to different stable
   solutions; on this 2-feature linear problem they do not.

2. **MaxentMini is NOT numerically equivalent to real Java.**
   The documented "reference" oracle used by the existing
   `test-optimizer.R`, `test-density.R`, `test-cloglog.R` in this repo
   diverges from real Java by ~4.6e-2 on lambdas — four orders of
   magnitude above the 1e-6 milestone target.  The pre-existing 1e-6
   parity tests in `maxentcpp` and `maxentcppCompTest` are therefore
   evidence of `maxentcpp ≈ MaxentMini`, **not** of `maxentcpp ≈ real
   Java`.  This was the core finding that motivated Phase A; the
   asymmetric fixture makes it quantitative.

3. **Loss accounting difference.**  `FeaturedSpace.getLoss()` in both
   Java and C++ returns the *unregularized* negative log-likelihood,
   while `Sequential.getLoss()` in Java returns `X.getLoss() + L1reg`.
   The C++ `train()` return value (`TrainResult.loss`) currently echoes
   the unregularized quantity — the trajectory binary adds
   `fs.get_l1_reg()` back to keep the comparison apples-to-apples.
   Whether `TrainResult.loss` should change is a separate API decision
   that can be revisited during Phase C.

4. **Stale `FeaturedSpace.entropy` cache in Java.**
   `density.FeaturedSpace.getEntropy()` caches the first computed
   value; `Sequential.run()` never invalidates the cache, so the
   per-iteration entropy reported by a naive `X.getEntropy()` call is
   stale.  `TrajectorySequential` explicitly resets `X.entropy = -1.0`
   before each snapshot; without this, the Java trajectory would show a
   constant `4.604...` entropy throughout.  This is a Java-side
   observation only and does not affect the final scalars.

## 5. Phase C / Phase D implications

- **1e-6 milestone (Phase D) on this fixture is already within reach for
  the linear-feature + cloglog scenario.**  The C++ port tracks real
  Java to ~6e-9 on lambdas when both optimizers are run to the same
  (large) iteration count with convergence disabled.

- **Phase C's port of `Sequential` is still required.**  The
  asymmetric-fixture mid-trajectory shows that the current C++
  optimizer takes a very different path through parameter space (its
  iter-20 state is ~2.5 units away from Java in L∞ terms).  Real-world
  runs with default convergence thresholds will stop somewhere along
  that path, not at the shared fixed point, so a user running
  `maxentcpp` with default settings on an asymmetric dataset will NOT
  see 1e-6 agreement with real Java.  Phase C must:
    1. Port `deltaLossBound`-based feature selection
       (`getBestFeature` / `doSequentialUpdate`).
    2. Port `newtonStep + searchAlpha` for continuous features.
    3. Port `doParallelUpdate` every 10 iterations.
    4. Port the loss-violation undo path.
    5. Preserve `FeaturedSpace::train()`'s existing goodAlpha path as a
       fallback for non-continuous features (hinge, threshold) once
       those are added.

- **The Phase D 1e-6 test will need the asymmetric fixture.**  On the
  symmetric fixture every oracle returns 0 and the 1e-6 gate is
  trivially satisfied, which is exactly the masking that Phase A
  flagged.

## 6. Additional oracles & fixtures (deferred)

Still worth adding, but outside Phase B's scope:

- Multi-feature (≥3) and non-linear (quadratic / hinge) fixtures once
  Phase C ports the corresponding feature families.
- A fixture with a feature whose sample deviation is at the
  `minDeviation = 0.001` floor — this exercises the clamped-beta path
  in both optimizers.
- A fixture where `doParallelUpdate` actually fires (requires enough
  active features for the every-10-iterations update to be
  non-degenerate).

## 7. Reproducing these tables

```sh
# 1. Build the reference oracle JAR (includes MaxentRefRunner + MaxentMini).
bash inst/java/build_ref.sh

# 2. Regenerate Java + MaxentMini trajectories on both fixtures.
java -cp inst/java/maxent_ref.jar density.MaxentRefRunner \
     inst/extdata/bio1.asc inst/extdata/bio2.asc \
     inst/extdata/occurrences.csv \
     inst/extdata/golden
java -cp inst/java/maxent_ref.jar density.MaxentRefRunner \
     inst/extdata/asym/bio1.asc inst/extdata/asym/bio2.asc \
     inst/extdata/asym/occurrences.csv \
     inst/extdata/golden/asym
java -cp inst/java/maxent_ref.jar density.MaxentRefRunner --mini \
     inst/extdata/bio1.asc inst/extdata/bio2.asc \
     inst/extdata/occurrences.csv \
     inst/extdata/golden
java -cp inst/java/maxent_ref.jar density.MaxentRefRunner --mini \
     inst/extdata/asym/bio1.asc inst/extdata/asym/bio2.asc \
     inst/extdata/asym/occurrences.csv \
     inst/extdata/golden/asym

# 3. Build and run the C++ trajectory binary against maxentcpp directly.
g++ -std=c++17 -O2 \
    -I../maxentcpp/src/cpp/include \
    -o /tmp/trajectory_cpp \
    inst/cpp/trajectory_cpp.cpp \
    ../maxentcpp/src/cpp/src/featured_space.cpp
/tmp/trajectory_cpp \
    inst/extdata/bio1.asc inst/extdata/bio2.asc \
    inst/extdata/occurrences.csv \
    inst/extdata/golden/trajectory_cpp.csv
/tmp/trajectory_cpp \
    inst/extdata/asym/bio1.asc inst/extdata/asym/bio2.asc \
    inst/extdata/asym/occurrences.csv \
    inst/extdata/golden/asym/trajectory_cpp.csv

# 4. Print the summary tables.
python3 tools/compare_trajectories.py \
    inst/extdata/golden inst/extdata/golden/asym
```

The R-level regression lives in
`tests/testthat/test-optimizer-trajectory.R`; it verifies that
`MaxentRefRunner` still reproduces the committed goldens bit-for-bit
and that the published gaps stay within the documented bands.
