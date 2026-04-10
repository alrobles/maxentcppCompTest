# tests/testthat/test-linear-features.R
#
# Test battery: Linear Feature Evaluation (Tests 1–4)
#
# Validates that the linear feature normalization formula
#   f(i) = (values[i] - min) / (max - min)
# produces numerically identical results in the C++ (maxentcpp) and Java
# (MaxentMini) implementations, and agrees with a manual R calculation.
# ---------------------------------------------------------------------------

# Test 1: Java vs C++ on bio1 — all 100 background points -------------------
test_that("Test 1: Java and C++ linear feature eval agree on bio1", {
    skip_if_no_java()
    skip_if_not_installed("maxentcpp")

    fx <- make_mock_fixture()

    f_cpp    <- maxentcpp::maxent_linear_feature(fx$bio1_vec, "bio1")
    cpp_vals <- vapply(seq_len(fx$n),
                       function(i) maxentcpp::maxent_feature_eval(f_cpp, i),
                       numeric(1L))

    java_vals <- java_linear_feature_eval(fx$bio1_vec, "bio1")

    expect_length(java_vals, fx$n)
    expect_equal(cpp_vals, java_vals, tolerance = 1e-12)
})

# Test 2: Java vs C++ on bio2 — all 100 background points -------------------
test_that("Test 2: Java and C++ linear feature eval agree on bio2", {
    skip_if_no_java()
    skip_if_not_installed("maxentcpp")

    fx <- make_mock_fixture()

    f_cpp    <- maxentcpp::maxent_linear_feature(fx$bio2_vec, "bio2")
    cpp_vals <- vapply(seq_len(fx$n),
                       function(i) maxentcpp::maxent_feature_eval(f_cpp, i),
                       numeric(1L))

    java_vals <- java_linear_feature_eval(fx$bio2_vec, "bio2")

    expect_equal(cpp_vals, java_vals, tolerance = 1e-12)
})

# Test 3: Edge cases — constant feature (min == max) -------------------------
test_that("Test 3: Constant feature (min == max) returns all-zero vector", {
    skip_if_no_java()
    skip_if_not_installed("maxentcpp")

    const_vals <- rep(5.0, 20L)

    f_cpp      <- maxentcpp::maxent_linear_feature(const_vals, "const")
    cpp_result <- vapply(seq_len(20L),
                         function(i) maxentcpp::maxent_feature_eval(f_cpp, i),
                         numeric(1L))

    java_result <- java_linear_feature_eval(const_vals)

    # Both implementations must return 0 for a constant feature
    expect_true(all(is.finite(cpp_result)),
                label = "C++ constant feature: all finite")
    expect_true(all(is.finite(java_result)),
                label = "Java constant feature: all finite")
    expect_equal(cpp_result,  rep(0.0, 20L), tolerance = 1e-12)
    expect_equal(java_result, rep(0.0, 20L), tolerance = 1e-12)
})

# Test 4: Manual R formula vs C++ and Java — all three agree to 1e-14 -------
test_that("Test 4: Manual R, C++, and Java linear feature agree to 1e-14", {
    skip_if_no_java()
    skip_if_not_installed("maxentcpp")

    fx    <- make_mock_fixture()
    v     <- fx$bio1_vec
    min_v <- min(v)
    max_v <- max(v)
    scale <- max_v - min_v

    r_vals   <- (v - min_v) / scale

    f_cpp    <- maxentcpp::maxent_linear_feature(v, "bio1")
    cpp_vals <- vapply(seq_len(fx$n),
                       function(i) maxentcpp::maxent_feature_eval(f_cpp, i),
                       numeric(1L))

    java_vals <- java_linear_feature_eval(v)

    expect_equal(r_vals,   cpp_vals,  tolerance = 1e-14,
                 label = "R manual vs C++")
    expect_equal(r_vals,   java_vals, tolerance = 1e-14,
                 label = "R manual vs Java")
    expect_equal(cpp_vals, java_vals, tolerance = 1e-14,
                 label = "C++ vs Java")
})
