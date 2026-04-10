# tests/testthat/test-density.R
#
# Test battery: Density & Prediction (Tests 15–17)
#
# Validates that the normalized density weights, raw prediction values,
# and Shannon entropy are numerically equivalent between the C++ (maxentcpp)
# and Java (MaxentMini) implementations after full training to convergence.
# ---------------------------------------------------------------------------

# Test 15: Density normalization after training ------------------------------
test_that("Test 15: C++ and Java density weights agree after training", {
    skip_if_no_java()
    skip_if_not_installed("maxentcpp")

    fx <- make_mock_fixture()

    java_weights <- java_get_density(
        fx$bio1_vec, fx$bio2_vec, fx$sample_indices,
        fx$beta_multiplier, fx$min_deviation,
        fx$max_iter, fx$convergence
    )

    fs <- .make_cpp_fs(fx)
    cpp_weights <- maxentcpp::maxent_model_weights(fs)

    expect_length(java_weights, fx$n)
    expect_length(cpp_weights,  fx$n)

    expect_equal(sum(java_weights), 1.0, tolerance = 1e-10,
                 label = "Java weights sum to 1")
    expect_equal(sum(cpp_weights),  1.0, tolerance = 1e-10,
                 label = "C++ weights sum to 1")

    expect_equal(cpp_weights, java_weights, tolerance = 1e-8,
                 label = "All 100 density weights: C++ vs Java")
})

# Test 16: Raw prediction values (normalized density) ------------------------
test_that("Test 16: Raw predictions (weights) agree between C++ and Java", {
    skip_if_no_java()
    skip_if_not_installed("maxentcpp")

    fx <- make_mock_fixture()

    java_weights <- java_get_density(
        fx$bio1_vec, fx$bio2_vec, fx$sample_indices,
        fx$beta_multiplier, fx$min_deviation,
        fx$max_iter, fx$convergence
    )

    fs <- .make_cpp_fs(fx)
    cpp_weights <- maxentcpp::maxent_model_weights(fs)

    # All non-negative
    expect_true(all(java_weights >= 0), label = "Java weights >= 0")
    expect_true(all(cpp_weights  >= 0), label = "C++ weights >= 0")

    # Spearman rank correlation >= 0.99
    spearman_cor <- stats::cor(cpp_weights, java_weights, method = "spearman")
    expect_true(spearman_cor >= 0.99,
                label = sprintf("Spearman %.4f >= 0.99", spearman_cor))

    # Element-wise agreement
    expect_equal(cpp_weights, java_weights, tolerance = 1e-8,
                 label = "Element-wise: C++ vs Java raw predictions")
})

# Test 17: Entropy after training --------------------------------------------
test_that("Test 17: C++ and Java entropy agree after training", {
    skip_if_no_java()
    skip_if_not_installed("maxentcpp")

    fx <- make_mock_fixture()

    java_res <- java_featured_space_train(
        fx$bio1_vec, fx$bio2_vec, fx$sample_indices,
        fx$beta_multiplier, fx$min_deviation,
        fx$max_iter, fx$convergence
    )

    fs          <- .make_cpp_fs(fx)
    cpp_entropy <- maxentcpp::maxent_model_entropy(fs)

    expect_true(java_res$entropy >= 0, label = "Java entropy >= 0")
    expect_true(cpp_entropy       >= 0, label = "C++ entropy >= 0")

    expect_equal(cpp_entropy, java_res$entropy, tolerance = 1e-8,
                 label = "Shannon entropy: C++ vs Java")
})
