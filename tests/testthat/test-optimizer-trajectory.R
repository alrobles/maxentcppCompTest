# tests/testthat/test-optimizer-trajectory.R
#
# Phase B of maxentcpp #36 / #37 -- quantify the per-iteration gap between
# the real Java `density.Sequential` optimizer (via MaxentRefRunner),
# the simplified goodAlpha-only `MaxentMini` oracle, and the C++ port
# `maxent::FeaturedSpace::train()`.
#
# This file is the R-level companion to
# `docs/FIDELITY_BASELINE_REPORT.md` and the standalone analysis tool
# `tools/compare_trajectories.py`.  The committed golden CSVs in
# `inst/extdata/golden/` and `inst/extdata/golden/asym/` are the primary
# Phase B deliverable; the tests below verify that the committed goldens
# are still produced bit-for-bit by the current MaxentRefRunner and
# match the high-level expectations stated in the baseline report.
#
# Expectations (see FIDELITY_BASELINE_REPORT.md for the full derivation):
#
#   * Symmetric fixture  : all three optimizers reach lambda = (0, 0)
#                          -> Linf(lambda) = 0 across the full trajectory.
#   * Asymmetric fixture : at iter 500 (convergence disabled),
#                          Linf(lambda_cpp - lambda_java)  <= 1e-6
#                          Linf(lambda_mini - lambda_java) is O(1e-2)
#                            (expected: MaxentMini lacks the real
#                             Sequential optimizer's feature selection).
#
# The 1e-6 milestone from issue #36 is the final Phase D gate; Phase B
# only publishes the current baseline number, so the asymmetric asserts
# are diagnostic (informational) rather than strict.
# ---------------------------------------------------------------------------

.golden_dir_sym <- function() {
    d <- system.file("extdata", "golden", package = "maxentcppCompTest")
    if (!nzchar(d)) {
        pkg <- system.file(package = "maxentcppCompTest")
        d   <- file.path(pkg, "inst", "extdata", "golden")
    }
    d
}

.golden_dir_asym <- function() {
    file.path(.golden_dir_sym(), "asym")
}

.read_trajectory <- function(path) {
    df <- utils::read.csv(path, stringsAsFactors = FALSE)
    # Only keep the five canonical columns so assertions line up across
    # implementations (the C++ trajectory also carries loss_unreg, l1_reg).
    df[, c("iteration", "loss", "entropy", "lambda_0", "lambda_1")]
}

.linf_lambda <- function(a, b) {
    stopifnot(all(a$iteration == b$iteration))
    pmax(abs(a$lambda_0 - b$lambda_0),
         abs(a$lambda_1 - b$lambda_1))
}

test_that("MaxentRefRunner reproduces the committed symmetric trajectory", {
    skip_if_no_java_ref()

    gold <- .golden_dir_sym()
    skip_if_not(file.exists(file.path(gold, "trajectory_java.csv")),
                message = "golden trajectory_java.csv not present")

    fx <- make_mock_fixture()
    tj <- maxentcppCompTest::java_ref_trajectory(
        fx$bio1_vec, fx$bio2_vec, fx$sample_indices,
        beta_multiplier = fx$beta_multiplier
    )

    ref <- .read_trajectory(file.path(gold, "trajectory_java.csv"))

    # Bit-identical determinism; deterministic optimizer + deterministic
    # inputs -> 0 tolerance is appropriate.
    expect_equal(tj$iteration, ref$iteration)
    expect_equal(tj$loss,      ref$loss,      tolerance = 1e-14)
    expect_equal(tj$entropy,   ref$entropy,   tolerance = 1e-14)
    expect_equal(tj$lambda_0,  ref$lambda_0,  tolerance = 1e-14)
    expect_equal(tj$lambda_1,  ref$lambda_1,  tolerance = 1e-14)

    # Symmetric-fixture invariant: no signal -> all zero lambdas.
    expect_true(all(tj$lambda_0 == 0))
    expect_true(all(tj$lambda_1 == 0))
})

test_that("MaxentRefRunner reproduces the committed asymmetric trajectory", {
    skip_if_no_java_ref()

    # The committed asymmetric goldens were generated on Linux x86_64
    # with OpenJDK 17 (then rebuilt with --release 11 for CI).  The real
    # density.Sequential optimizer accumulates enough floating-point
    # operations over 500 iterations that java.lang.Math.exp / log
    # produce trajectory values that differ by ~5e-10 on macOS ARM64
    # and ~1e-8 on Windows at the same iteration; this is ordinary
    # cross-platform FP non-determinism, not an oracle regression.
    # The 1e-14 reproducibility claim in Phase B is scoped to the
    # golden-generation platform (Linux); on other OSes we only verify
    # the goldens are consumed without error via the full pipeline.
    skip_on_os(c("mac", "windows"))

    gold <- .golden_dir_asym()
    skip_if_not(file.exists(file.path(gold, "trajectory_java.csv")),
                message = "asymmetric trajectory_java.csv not present")

    fx <- make_mock_fixture_asym()
    tj <- maxentcppCompTest::java_ref_trajectory(
        fx$bio1_vec, fx$bio2_vec, fx$sample_indices,
        beta_multiplier = fx$beta_multiplier
    )

    ref <- .read_trajectory(file.path(gold, "trajectory_java.csv"))
    expect_equal(tj$loss,     ref$loss,     tolerance = 1e-14)
    expect_equal(tj$entropy,  ref$entropy,  tolerance = 1e-14)
    expect_equal(tj$lambda_0, ref$lambda_0, tolerance = 1e-14)
    expect_equal(tj$lambda_1, ref$lambda_1, tolerance = 1e-14)

    # Asymmetric-fixture invariant: the optimizer MUST move lambdas off
    # zero at least once; otherwise the fixture is not exercising the
    # Sequential vs MaxentMini gap.
    expect_true(max(abs(tj$lambda_0)) > 0.1,
                info = "asymmetric fixture produced trivial lambda_0")
    expect_true(max(abs(tj$lambda_1)) > 0.1,
                info = "asymmetric fixture produced trivial lambda_1")
})

test_that("Phase B baseline report: asymmetric-fixture gap at iter 500", {
    # Informational assertions on the committed golden CSVs -- this is
    # the Phase B "quantified baseline" record.  The tolerances below are
    # the published baseline numbers; crossing them either direction is
    # a signal worth reviewing before merging.

    gold <- .golden_dir_asym()
    skip_if_not(file.exists(file.path(gold, "trajectory_java.csv")),
                message = "asymmetric trajectory_java.csv not present")
    skip_if_not(file.exists(file.path(gold, "trajectory_cpp.csv")),
                message = "asymmetric trajectory_cpp.csv not present")
    skip_if_not(file.exists(file.path(gold, "trajectory_mini.csv")),
                message = "asymmetric trajectory_mini.csv not present")

    tj <- .read_trajectory(file.path(gold, "trajectory_java.csv"))
    tc <- .read_trajectory(file.path(gold, "trajectory_cpp.csv"))
    tm <- .read_trajectory(file.path(gold, "trajectory_mini.csv"))

    expect_equal(tj$iteration, tc$iteration)
    expect_equal(tj$iteration, tm$iteration)

    # Last checkpoint (iter 500) is the convergence-disabled "fully
    # trained" state of each optimizer.
    last <- nrow(tj)

    # Final C++ vs Java lambda gap is in the 1e-9 neighborhood.  If it
    # ever regresses above 1e-6, the 1e-6 Phase D milestone is no longer
    # safely achievable without optimizer changes.
    linf_cpp <- max(abs(tc$lambda_0[last] - tj$lambda_0[last]),
                    abs(tc$lambda_1[last] - tj$lambda_1[last]))
    expect_true(linf_cpp < 1e-6,
                info = sprintf("Linf(lambda_cpp - lambda_java) = %.3e",
                               linf_cpp))

    # Final MaxentMini vs Java gap is O(1e-2) because MaxentMini is
    # goodAlpha-only with no feature selection.  Phase C closes this
    # gap by porting Sequential's deltaLossBound + newtonStep path into
    # maxentcpp.  We assert an upper bound so a regression that
    # accidentally tightens this gap (meaning the three oracles have
    # silently converged) also trips the test.
    linf_mini <- max(abs(tm$lambda_0[last] - tj$lambda_0[last]),
                     abs(tm$lambda_1[last] - tj$lambda_1[last]))
    expect_true(linf_mini > 1e-3,
                info = sprintf("Linf(lambda_mini - lambda_java) = %.3e",
                               linf_mini))
    expect_true(linf_mini < 1e-1,
                info = sprintf("Linf(lambda_mini - lambda_java) = %.3e",
                               linf_mini))

    # Losses/entropies at iter 500 should agree on the regularized scale.
    expect_true(abs(tc$loss[last] - tj$loss[last]) < 1e-6,
                info = sprintf("|loss_cpp - loss_java| = %.3e",
                               abs(tc$loss[last] - tj$loss[last])))
    expect_true(abs(tc$entropy[last] - tj$entropy[last]) < 1e-6,
                info = sprintf("|entropy_cpp - entropy_java| = %.3e",
                               abs(tc$entropy[last] - tj$entropy[last])))
})
