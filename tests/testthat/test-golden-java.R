# tests/testthat/test-golden-java.R
#
# Phase A sanity test (maxentcpp #36 / #37).
#
# Verifies that re-running `density.MaxentRefRunner` against the checked-in
# mock fixture (inst/extdata/{bio1.asc,bio2.asc,occurrences.csv}) reproduces
# the checked-in golden CSVs (inst/extdata/golden/) at numerical tolerance
# well below 1e-6.  This does NOT yet test maxentcpp itself -- it only
# validates that the real-Java reference oracle is deterministic and that
# the golden baseline faithfully captures its output.
#
# If this test ever fails, the offending commit is the one that broke the
# oracle; re-generating the goldens should be a deliberate, reviewed act.
# ---------------------------------------------------------------------------

.golden_dir <- function() {
    d <- system.file("extdata", "golden", package = "maxentcppCompTest")
    if (!nzchar(d)) {
        pkg <- system.file(package = "maxentcppCompTest")
        d   <- file.path(pkg, "inst", "extdata", "golden")
    }
    d
}

.read_scalars <- function(path) {
    df <- utils::read.csv(path, stringsAsFactors = FALSE)
    setNames(df$value, df$quantity)
}

test_that("MaxentRefRunner reproduces the committed goldens at 1e-12", {
    skip_if_no_java_ref()

    gold_dir <- .golden_dir()
    skip_if_not(dir.exists(gold_dir),
                message = "inst/extdata/golden/ not present")
    skip_if_not(file.exists(file.path(gold_dir, "scalars.csv")),
                message = "golden scalars.csv not present")

    fx  <- make_mock_fixture()
    run <- maxentcppCompTest::java_ref_run(
        fx$bio1_vec, fx$bio2_vec, fx$sample_indices,
        beta_multiplier = fx$beta_multiplier,
        max_iter        = fx$max_iter,
        convergence     = fx$convergence
    )

    # ---- scalars --------------------------------------------------------
    g_scalars <- .read_scalars(file.path(gold_dir, "scalars.csv"))
    diag      <- maxentcppCompTest::java_ref_scalars(run$handle)

    expect_equal(diag$loss,                        g_scalars[["loss"]],
                 tolerance = 1e-12)
    expect_equal(diag$unregularized_loss,          g_scalars[["unregularized_loss"]],
                 tolerance = 1e-12)
    expect_equal(diag$l1_reg,                      g_scalars[["l1_reg"]],
                 tolerance = 1e-12)
    expect_equal(diag$gain,                        g_scalars[["gain"]],
                 tolerance = 1e-12)
    expect_equal(diag$entropy,                     g_scalars[["entropy"]],
                 tolerance = 1e-12)
    expect_equal(diag$density_normalizer,          g_scalars[["density_normalizer"]],
                 tolerance = 1e-12)
    expect_equal(diag$linear_predictor_normalizer, g_scalars[["linear_predictor_normalizer"]],
                 tolerance = 1e-12)
    expect_equal(diag$beta_multiplier,             g_scalars[["beta_multiplier"]],
                 tolerance = 1e-12)
    expect_equal(diag$auto_beta_lqp,               g_scalars[["auto_beta_lqp"]],
                 tolerance = 1e-12)
    expect_equal(as.integer(diag$iterations),
                 as.integer(g_scalars[["iterations"]]))
    expect_equal(as.integer(diag$num_features),
                 as.integer(g_scalars[["num_features"]]))
    expect_equal(as.integer(diag$num_points),
                 as.integer(g_scalars[["num_points"]]))

    # ---- lambdas / sample stats ----------------------------------------
    g_lambdas <- utils::read.csv(file.path(gold_dir, "lambdas.csv"),
                                 stringsAsFactors = FALSE)
    stats     <- maxentcppCompTest::java_ref_feature_stats(run$handle)

    expect_equal(stats$feature_names, g_lambdas$feature)
    expect_equal(as.numeric(maxentcppCompTest::java_ref_lambdas(run$handle)),
                 g_lambdas$lambda,             tolerance = 1e-12)
    expect_equal(stats$sample_expectation,
                 g_lambdas$sample_expectation, tolerance = 1e-12)
    expect_equal(stats$sample_deviation,
                 g_lambdas$sample_deviation,   tolerance = 1e-12)
    expect_equal(stats$expectation,
                 g_lambdas$expectation,        tolerance = 1e-12)
    expect_equal(stats$beta,
                 g_lambdas$beta,               tolerance = 1e-12)

    # ---- per-point density / raw / cloglog -----------------------------
    g_dens <- utils::read.csv(file.path(gold_dir, "density.csv"),
                              stringsAsFactors = FALSE)

    expect_equal(as.numeric(maxentcppCompTest::java_ref_density(run$handle)),
                 g_dens$density,      tolerance = 1e-12)
    expect_equal(as.numeric(maxentcppCompTest::java_ref_raw(run$handle)),
                 g_dens$raw,          tolerance = 1e-12)
    expect_equal(as.numeric(maxentcppCompTest::java_ref_cloglog(run$handle)),
                 g_dens$cloglog_java, tolerance = 1e-12)
})

test_that("MaxentRefRunner is deterministic across repeated calls", {
    skip_if_no_java_ref()
    fx <- make_mock_fixture()
    run1 <- maxentcppCompTest::java_ref_run(
        fx$bio1_vec, fx$bio2_vec, fx$sample_indices,
        beta_multiplier = fx$beta_multiplier,
        max_iter = fx$max_iter, convergence = fx$convergence
    )
    run2 <- maxentcppCompTest::java_ref_run(
        fx$bio1_vec, fx$bio2_vec, fx$sample_indices,
        beta_multiplier = fx$beta_multiplier,
        max_iter = fx$max_iter, convergence = fx$convergence
    )
    expect_identical(run1$iterations, run2$iterations)
    expect_equal(run1$loss,    run2$loss,    tolerance = 0)
    expect_equal(run1$entropy, run2$entropy, tolerance = 0)
    expect_equal(
        as.numeric(maxentcppCompTest::java_ref_density(run1$handle)),
        as.numeric(maxentcppCompTest::java_ref_density(run2$handle)),
        tolerance = 0
    )
})
