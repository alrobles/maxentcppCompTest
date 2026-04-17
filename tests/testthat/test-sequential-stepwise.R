# tests/testthat/test-sequential-stepwise.R
#
# Phase C (maxentcpp #36) R-level companion:
# Drives the Rcpp binding `maxentcpp::maxent_sequential_fit()` over the
# same asymmetric and symmetric fixtures used in Phase B and asserts
# ||Delta lambda||_inf <= 1e-6 against the committed Java oracle
# trajectory (`inst/extdata/golden/{,asym/}trajectory_java.csv`) at
# every checkpoint in {1, 2, 3, 5, 10, 20, 50, 100, 200, 500}.
#
# The C++ side of the 1e-6 claim is already proven in maxentcpp's own
# test suite (SequentialTest, merged in alrobles/maxentcpp#45); this
# file is the user-facing corollary: a Phase C regression that any
# future Rcpp-binding change would have to keep passing.
#
# Skipped unless both
#   * `maxentcpp` is installed (private package — skip on CI by default);
#   * the committed `trajectory_java.csv` golden is bundled in extdata
# are true, so CI on machines without the private `maxentcpp` dep stays
# green.  See `docs/FIDELITY_BASELINE_REPORT.md` for the Phase B baseline
# numbers and `docs/ARCHITECTURE_xtensor_openmp.md` (maxentcpp#45) for
# the optimizer port.
# ---------------------------------------------------------------------------

.seq_checkpoints <- c(1L, 2L, 3L, 5L, 10L, 20L, 50L, 100L, 200L, 500L)

.golden_trajectory_java <- function(dir) {
    path <- file.path(dir, "trajectory_java.csv")
    if (!file.exists(path)) return(NULL)
    df <- utils::read.csv(path, stringsAsFactors = FALSE)
    df[, c("iteration", "loss", "entropy", "lambda_0", "lambda_1")]
}

.run_sequential_trajectory <- function(fx, checkpoints = .seq_checkpoints) {
    f1 <- maxentcpp::maxent_linear_feature(fx$bio1_vec, "bio1")
    f2 <- maxentcpp::maxent_linear_feature(fx$bio2_vec, "bio2")
    fs <- maxentcpp::maxent_featured_space(fx$n, fx$sample_indices,
                                           list(f1, f2))
    maxentcpp::maxent_sequential_fit(
        fs,
        max_iter                  = fx$max_iter,
        convergence               = 0,
        beta_multiplier           = fx$beta_multiplier,
        min_deviation             = fx$min_deviation,
        parallel_update_frequency = 10L,
        disable_convergence_test  = TRUE,
        trajectory_iterations     = as.integer(checkpoints)
    )
}

# ---------------------------------------------------------------------------
# Asymmetric fixture -- the Phase B gap case.  Real Sequential moves lambdas
# far off zero; C++ Sequential (via our new Rcpp binding) must match Java to
# within 1e-6 at every checkpoint.
# ---------------------------------------------------------------------------

test_that("maxent_sequential_fit matches Java oracle <= 1e-6 (asymmetric)", {
    skip_if_not_installed("maxentcpp")

    gold_dir <- file.path(
        system.file("extdata", "golden", package = "maxentcppCompTest"),
        "asym"
    )
    gold <- .golden_trajectory_java(gold_dir)
    skip_if(is.null(gold),
            message = "asymmetric trajectory_java.csv golden not bundled")

    fx  <- make_mock_fixture_asym()
    res <- .run_sequential_trajectory(fx)

    traj <- res$trajectory
    expect_s3_class(traj, "data.frame")
    expect_true(all(traj$iteration == gold$iteration),
                info = "trajectory checkpoint iteration indices must match")

    # Vectorized ||Delta lambda||_inf per checkpoint.
    dlam <- pmax(abs(traj$lambda_0 - gold$lambda_0),
                 abs(traj$lambda_1 - gold$lambda_1))
    worst <- max(dlam)
    expect_lt(worst, 1e-6,
              label = sprintf(
                  "max ||dlam||_inf across checkpoints = %.3e (at iter %d)",
                  worst, traj$iteration[which.max(dlam)]))

    # Final-state corroboration: loss / entropy / lambdas must all agree
    # on the regularized scale.  The 1e-6 above is the tight claim; these
    # are the standard sanity follow-ups.
    last <- nrow(traj)
    expect_lt(abs(traj$loss[last]    - gold$loss[last]),    1e-6)
    expect_lt(abs(traj$entropy[last] - gold$entropy[last]), 1e-6)

    # Asymmetric invariant: lambdas must move off zero, otherwise we've
    # silently regressed to the degenerate MaxentMini-on-symmetric case.
    expect_true(max(abs(res$lambdas)) > 0.1,
                info = "asymmetric fixture produced trivial lambdas")
})

# ---------------------------------------------------------------------------
# Symmetric fixture -- sample means equal background means after scaling,
# so both the real Java optimizer and ours converge to lambda = (0, 0).
# This check guards against a regression in which the Rcpp binding
# accidentally passes uninitialized state and starts producing non-zero
# lambdas from a trivially zero-signal fixture.
# ---------------------------------------------------------------------------

test_that("maxent_sequential_fit matches Java oracle <= 1e-6 (symmetric)", {
    skip_if_not_installed("maxentcpp")

    gold_dir <- system.file("extdata", "golden",
                            package = "maxentcppCompTest")
    gold <- .golden_trajectory_java(gold_dir)
    skip_if(is.null(gold),
            message = "symmetric trajectory_java.csv golden not bundled")

    fx  <- make_mock_fixture()
    res <- .run_sequential_trajectory(fx)

    traj <- res$trajectory
    expect_true(all(traj$iteration == gold$iteration))

    dlam <- pmax(abs(traj$lambda_0 - gold$lambda_0),
                 abs(traj$lambda_1 - gold$lambda_1))
    expect_lt(max(dlam), 1e-6,
              label = "max ||dlam||_inf (symmetric)")

    # Symmetric-fixture invariant: trivial lambdas throughout.
    expect_true(all(abs(traj$lambda_0) < 1e-10))
    expect_true(all(abs(traj$lambda_1) < 1e-10))
})
