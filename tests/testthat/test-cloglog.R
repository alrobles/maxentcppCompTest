# tests/testthat/test-cloglog.R
#
# Test battery: Cloglog Transformation (Tests 18–22)
#
# Validates that the complementary log-log output transformation
#   cloglog(x) = 1 - exp(-x * exp(H))
# where x = normalized density weight and H = Shannon entropy, is computed
# correctly and equivalently by both implementations.
# ---------------------------------------------------------------------------

# Test 18: Cloglog formula correctness ---------------------------------------
test_that("Test 18: Java cloglog matches formula 1 - exp(-raw * exp(entropy))", {
    skip_if_no_java()

    raw_vals <- c(0.0, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0)
    entropy  <- 3.5

    r_cloglog   <- 1 - exp(-raw_vals * exp(entropy))
    java_cloglog <- java_cloglog_transform(raw_vals, entropy)

    expect_equal(java_cloglog, r_cloglog, tolerance = 1e-14,
                 label = "Cloglog formula: Java vs R manual")
})

# Test 19: Cloglog output range [0, 1] ----------------------------------------
test_that("Test 19: Cloglog values are in [0, 1] for both implementations", {
    skip_if_no_java()
    skip_if_not_installed("maxentcpp")

    fx <- make_mock_fixture()

    # Java
    java_res     <- java_featured_space_train(
        fx$bio1_vec, fx$bio2_vec, fx$sample_indices,
        fx$beta_multiplier, fx$min_deviation,
        fx$max_iter, fx$convergence
    )
    java_weights <- java_get_density(
        fx$bio1_vec, fx$bio2_vec, fx$sample_indices,
        fx$beta_multiplier, fx$min_deviation,
        fx$max_iter, fx$convergence
    )
    java_cloglog <- java_cloglog_transform(java_weights, java_res$entropy)

    expect_true(all(java_cloglog >= 0.0), label = "Java cloglog >= 0")
    expect_true(all(java_cloglog <= 1.0), label = "Java cloglog <= 1")

    # C++
    fs          <- .make_cpp_fs(fx)
    cpp_weights <- maxentcpp::maxent_model_weights(fs)
    cpp_entropy <- maxentcpp::maxent_model_entropy(fs)
    cpp_cloglog <- 1 - exp(-cpp_weights * exp(cpp_entropy))

    expect_true(all(cpp_cloglog >= 0.0), label = "C++ cloglog >= 0")
    expect_true(all(cpp_cloglog <= 1.0), label = "C++ cloglog <= 1")
})

# Test 20: Cloglog monotonicity preservation ---------------------------------
test_that("Test 20: Cloglog preserves monotonic ordering of raw values", {
    skip_if_no_java()

    set.seed(42)
    raw_vals  <- sort(runif(50L, 0, 0.02))
    entropy   <- 4.0
    cloglog_v <- java_cloglog_transform(raw_vals, entropy)

    # Sorted input must produce sorted (non-decreasing) output
    expect_true(all(diff(cloglog_v) >= 0),
                label = "Cloglog is monotone non-decreasing")
})

# Test 21: Manual R, Java, and C++ cloglog all agree to 1e-10 ---------------
test_that("Test 21: Manual R, Java, and C++ cloglog agree to 1e-10", {
    skip_if_no_java()
    skip_if_not_installed("maxentcpp")

    fx <- make_mock_fixture()

    # Train C++ model to get reference weights and entropy
    fs          <- .make_cpp_fs(fx)
    cpp_weights <- maxentcpp::maxent_model_weights(fs)
    cpp_entropy <- maxentcpp::maxent_model_entropy(fs)

    # Manual R formula (ground truth)
    r_cloglog <- 1 - exp(-cpp_weights * exp(cpp_entropy))

    # Java cloglog using same weights / entropy
    java_cloglog <- java_cloglog_transform(cpp_weights, cpp_entropy)

    expect_equal(r_cloglog, java_cloglog, tolerance = 1e-10,
                 label = "Manual R vs Java cloglog (using C++ weights)")
    expect_true(all(is.finite(r_cloglog)), label = "All cloglog values are finite")
})

# Test 22: End-to-end cloglog prediction comparison --------------------------
test_that("Test 22: End-to-end C++ vs Java cloglog predictions agree to 1e-6", {
    skip_if_no_java()
    skip_if_not_installed("maxentcpp")

    fx <- make_mock_fixture()

    # Java end-to-end
    java_res     <- java_featured_space_train(
        fx$bio1_vec, fx$bio2_vec, fx$sample_indices,
        fx$beta_multiplier, fx$min_deviation,
        fx$max_iter, fx$convergence
    )
    java_weights <- java_get_density(
        fx$bio1_vec, fx$bio2_vec, fx$sample_indices,
        fx$beta_multiplier, fx$min_deviation,
        fx$max_iter, fx$convergence
    )
    java_cloglog <- java_cloglog_transform(java_weights, java_res$entropy)

    # C++ end-to-end
    fs          <- .make_cpp_fs(fx)
    cpp_weights <- maxentcpp::maxent_model_weights(fs)
    cpp_entropy <- maxentcpp::maxent_model_entropy(fs)
    cpp_cloglog <- 1 - exp(-cpp_weights * exp(cpp_entropy))

    expect_length(java_cloglog, fx$n)
    expect_length(cpp_cloglog,  fx$n)

    expect_equal(cpp_cloglog, java_cloglog, tolerance = 1e-6,
                 label = "End-to-end cloglog: all 100 cells, C++ vs Java")
})
