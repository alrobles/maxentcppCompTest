# tests/testthat/test-sample-stats.R
#
# Test battery: Sample Expectations & Deviations (Tests 5–7)
#
# Validates that sample expectations (empirical mean of the normalized feature
# over occurrence points), sample deviations (population std dev), and the
# resulting regularization betas are computed identically in Java and R.
# ---------------------------------------------------------------------------

# Helper: normalize raw values using the same formula as MaxentMini ----------
.normalize <- function(values) {
    min_v <- min(values)
    max_v <- max(values)
    scale <- if (max_v == min_v) 1.0 else (max_v - min_v)
    (values - min_v) / scale
}

# Helper: population std dev (as used by MaxentMini) ------------------------
.pop_sd <- function(x) sqrt(mean((x - mean(x))^2))

# Test 5: Sample expectation for bio1 ----------------------------------------
test_that("Test 5: Java and R sample expectation agree for bio1", {
    skip_if_no_java()

    fx   <- make_mock_fixture()
    v    <- fx$bio1_vec
    v_n  <- .normalize(v)

    r_sample_exp <- mean(v_n[fx$sample_indices + 1L])   # 0-based → 1-based
    java_exp <- java_sample_expectation(v, fx$sample_indices)

    expect_equal(java_exp, r_sample_exp, tolerance = 1e-10)
    expect_true(java_exp >= 0.0 && java_exp <= 1.0,
                label = "Sample expectation is in [0, 1]")
})

# Test 5b: Sample expectation also agrees for bio2 ---------------------------
test_that("Test 5b: Java and R sample expectation agree for bio2", {
    skip_if_no_java()

    fx   <- make_mock_fixture()
    v    <- fx$bio2_vec
    v_n  <- .normalize(v)

    r_sample_exp <- mean(v_n[fx$sample_indices + 1L])
    java_exp <- java_sample_expectation(v, fx$sample_indices)

    expect_equal(java_exp, r_sample_exp, tolerance = 1e-10)
})

# Test 6: Sample deviation for bio1 ------------------------------------------
test_that("Test 6: Java and R sample deviation (pop std dev) agree for bio1", {
    skip_if_no_java()

    fx  <- make_mock_fixture()
    v   <- fx$bio1_vec
    v_n <- .normalize(v)

    r_sample_dev <- .pop_sd(v_n[fx$sample_indices + 1L])
    java_dev <- java_sample_deviation(v, fx$sample_indices)

    expect_equal(java_dev, r_sample_dev, tolerance = 1e-10)
})

# Test 6b: Sample deviation also agrees for bio2 -----------------------------
test_that("Test 6b: Java and R sample deviation agree for bio2", {
    skip_if_no_java()

    fx  <- make_mock_fixture()
    v   <- fx$bio2_vec
    v_n <- .normalize(v)

    r_sample_dev <- .pop_sd(v_n[fx$sample_indices + 1L])
    java_dev <- java_sample_deviation(v, fx$sample_indices)

    expect_equal(java_dev, r_sample_dev, tolerance = 1e-10)
})

# Test 7: Beta (regularization) calculation ----------------------------------
test_that("Test 7: Java and R beta agree for bio1 and bio2", {
    skip_if_no_java()

    fx        <- make_mock_fixture()
    beta_mult <- fx$beta_multiplier
    min_dev   <- fx$min_deviation
    n_samp    <- fx$n_samples

    compute_r_beta <- function(values) {
        v_n     <- .normalize(values)
        dev_raw <- .pop_sd(v_n[fx$sample_indices + 1L])
        dev     <- max(dev_raw, min_dev)
        beta_mult * dev / sqrt(n_samp)
    }

    r_beta1 <- compute_r_beta(fx$bio1_vec)
    r_beta2 <- compute_r_beta(fx$bio2_vec)

    # Java computes betas internally; recover via deviation + formula
    java_dev1 <- java_sample_deviation(fx$bio1_vec, fx$sample_indices)
    java_dev2 <- java_sample_deviation(fx$bio2_vec, fx$sample_indices)

    java_beta1 <- beta_mult * max(java_dev1, min_dev) / sqrt(n_samp)
    java_beta2 <- beta_mult * max(java_dev2, min_dev) / sqrt(n_samp)

    expect_equal(java_beta1, r_beta1, tolerance = 1e-10,
                 label = "bio1 beta: Java vs R")
    expect_equal(java_beta2, r_beta2, tolerance = 1e-10,
                 label = "bio2 beta: Java vs R")
})
