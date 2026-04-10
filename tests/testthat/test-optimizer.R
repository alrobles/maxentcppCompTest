# tests/testthat/test-optimizer.R
#
# Test battery: Optimizer / Loss Function (Tests 8–14)
#
# Validates that the goodAlpha-based sequential coordinate-ascent optimizer
# produces numerically equivalent loss trajectories, lambda values, and
# helper-function outputs between the C++ (maxentcpp) and Java (MaxentMini)
# implementations.
# ---------------------------------------------------------------------------

# Test 8: Initial loss = log(n) before any training --------------------------
test_that("Test 8: C++ and Java agree on initial loss = log(n)", {
    skip_if_no_java()
    skip_if_not_installed("maxentcpp")

    fx <- make_mock_fixture()

    java_init_loss <- java_get_initial_loss(
        fx$bio1_vec, fx$bio2_vec, fx$sample_indices,
        fx$beta_multiplier, fx$min_deviation
    )
    expected_loss <- log(fx$n)

    expect_equal(java_init_loss, expected_loss, tolerance = 1e-10,
                 label = "Java initial loss == log(n)")

    # C++ initial loss (untrained FeaturedSpace)
    f1 <- maxentcpp::maxent_linear_feature(fx$bio1_vec, "bio1")
    f2 <- maxentcpp::maxent_linear_feature(fx$bio2_vec, "bio2")
    fs <- maxentcpp::maxent_featured_space(fx$n, fx$sample_indices, list(f1, f2))
    cpp_init_loss <- maxentcpp::maxent_model_loss(fs)

    expect_equal(cpp_init_loss, java_init_loss, tolerance = 1e-10,
                 label = "C++ vs Java initial loss")
})

# Test 9: Loss after 1 iteration ---------------------------------------------
test_that("Test 9: C++ and Java agree on loss after 1 iteration", {
    skip_if_no_java()
    skip_if_not_installed("maxentcpp")

    fx <- make_mock_fixture()

    java_res <- java_train_n_iterations(
        fx$bio1_vec, fx$bio2_vec, fx$sample_indices,
        n_iterations    = 1L,
        beta_multiplier = fx$beta_multiplier,
        min_deviation   = fx$min_deviation
    )

    f1      <- maxentcpp::maxent_linear_feature(fx$bio1_vec, "bio1")
    f2      <- maxentcpp::maxent_linear_feature(fx$bio2_vec, "bio2")
    fs      <- maxentcpp::maxent_featured_space(fx$n, fx$sample_indices,
                                                list(f1, f2))
    cpp_fit <- maxentcpp::maxent_fit(fs,
                                     max_iter        = 1L,
                                     convergence     = fx$convergence,
                                     beta_multiplier = fx$beta_multiplier,
                                     min_deviation   = fx$min_deviation)

    expect_equal(cpp_fit$loss, java_res$loss, tolerance = 1e-8,
                 label = "Loss after 1 iteration: C++ vs Java")
})

# Test 10: Loss after 10 iterations ------------------------------------------
test_that("Test 10: C++ and Java agree on loss after 10 iterations", {
    skip_if_no_java()
    skip_if_not_installed("maxentcpp")

    fx <- make_mock_fixture()

    java_res <- java_train_n_iterations(
        fx$bio1_vec, fx$bio2_vec, fx$sample_indices,
        n_iterations    = 10L,
        beta_multiplier = fx$beta_multiplier,
        min_deviation   = fx$min_deviation
    )

    f1      <- maxentcpp::maxent_linear_feature(fx$bio1_vec, "bio1")
    f2      <- maxentcpp::maxent_linear_feature(fx$bio2_vec, "bio2")
    fs      <- maxentcpp::maxent_featured_space(fx$n, fx$sample_indices,
                                                list(f1, f2))
    cpp_fit <- maxentcpp::maxent_fit(fs,
                                     max_iter        = 10L,
                                     convergence     = fx$convergence,
                                     beta_multiplier = fx$beta_multiplier,
                                     min_deviation   = fx$min_deviation)

    expect_equal(cpp_fit$loss, java_res$loss, tolerance = 1e-8,
                 label = "Loss after 10 iterations: C++ vs Java")
})

# Test 11: Final loss after convergence --------------------------------------
test_that("Test 11: C++ and Java agree on final loss after convergence", {
    skip_if_no_java()
    skip_if_not_installed("maxentcpp")

    fx <- make_mock_fixture()

    java_res <- java_featured_space_train(
        fx$bio1_vec, fx$bio2_vec, fx$sample_indices,
        fx$beta_multiplier, fx$min_deviation,
        fx$max_iter, fx$convergence
    )

    f1      <- maxentcpp::maxent_linear_feature(fx$bio1_vec, "bio1")
    f2      <- maxentcpp::maxent_linear_feature(fx$bio2_vec, "bio2")
    fs      <- maxentcpp::maxent_featured_space(fx$n, fx$sample_indices,
                                                list(f1, f2))
    cpp_fit <- maxentcpp::maxent_fit(fs,
                                     max_iter        = fx$max_iter,
                                     convergence     = fx$convergence,
                                     beta_multiplier = fx$beta_multiplier,
                                     min_deviation   = fx$min_deviation)

    expect_equal(cpp_fit$loss,    java_res$loss,    tolerance = 1e-6,
                 label = "Final loss: C++ vs Java")
    expect_equal(cpp_fit$entropy, java_res$entropy, tolerance = 1e-6,
                 label = "Final entropy: C++ vs Java")
})

# Test 12: Lambda values after convergence -----------------------------------
test_that("Test 12: C++ and Java lambda values agree after convergence", {
    skip_if_no_java()
    skip_if_not_installed("maxentcpp")

    fx <- make_mock_fixture()

    java_res <- java_featured_space_train(
        fx$bio1_vec, fx$bio2_vec, fx$sample_indices,
        fx$beta_multiplier, fx$min_deviation,
        fx$max_iter, fx$convergence
    )

    f1      <- maxentcpp::maxent_linear_feature(fx$bio1_vec, "bio1")
    f2      <- maxentcpp::maxent_linear_feature(fx$bio2_vec, "bio2")
    fs      <- maxentcpp::maxent_featured_space(fx$n, fx$sample_indices,
                                                list(f1, f2))
    cpp_fit <- maxentcpp::maxent_fit(fs,
                                     max_iter        = fx$max_iter,
                                     convergence     = fx$convergence,
                                     beta_multiplier = fx$beta_multiplier,
                                     min_deviation   = fx$min_deviation)

    expect_equal(cpp_fit$lambdas[1], java_res$lambdas[1], tolerance = 1e-6,
                 label = "lambda_bio1: C++ vs Java")
    expect_equal(cpp_fit$lambdas[2], java_res$lambdas[2], tolerance = 1e-6,
                 label = "lambda_bio2: C++ vs Java")
})

# Test 13: goodAlpha step size formula ----------------------------------------
test_that("Test 13: Java goodAlpha matches R manual calculation", {
    skip_if_no_java()

    # Test the goodAlpha formula directly with representative parameter values
    # (no need for maxentcpp here — testing the pure formula)
    test_cases <- list(
        list(N1 = 0.6,  W1 = 0.4,  beta = 0.05, lambda = 0.0),
        list(N1 = 0.3,  W1 = 0.5,  beta = 0.02, lambda = 0.1),
        list(N1 = 0.8,  W1 = 0.6,  beta = 0.10, lambda = -0.5),
        list(N1 = 0.5,  W1 = 0.5,  beta = 0.01, lambda = 0.0)
    )

    r_good_alpha <- function(N1, W1, beta, lambda) {
        W0 <- 1 - W1; N0 <- 1 - N1
        if (W0 < 1e-6 || W1 < 1e-6) return(0)
        alpha_try <- function(N1_, N0_, beta_sign) {
            if (N1_ - beta > 1e-6) {
                a <- log((N1_ - beta) * W0 / ((N0_ + beta) * W1))
                if (a + lambda > 0) return(a)
            }
            NULL
        }
        a1 <- alpha_try(N1, N0, 1)
        if (!is.null(a1)) return(a1)
        if (N0 - beta > 1e-6) {
            a2 <- log((N1 + beta) * W0 / ((N0 - beta) * W1))
            if (a2 + lambda < 0) return(a2)
        }
        -lambda
    }

    for (tc in test_cases) {
        r_alpha   <- r_good_alpha(tc$N1, tc$W1, tc$beta, tc$lambda)
        java_alpha <- java_good_alpha(tc$N1, tc$W1, tc$beta, tc$lambda)

        expect_equal(java_alpha, r_alpha, tolerance = 1e-14,
                     label = sprintf("goodAlpha(N1=%.2f, W1=%.2f, beta=%.2f, lam=%.2f)",
                                     tc$N1, tc$W1, tc$beta, tc$lambda))
        expect_true(is.finite(java_alpha))
    }
})

# Test 14: reduceAlpha damping schedule --------------------------------------
test_that("Test 14: Java reduceAlpha matches expected damping schedule", {
    skip_if_no_java()

    alpha_raw <- 2.4

    # iteration < 10: divide by 50
    expect_equal(java_reduce_alpha(alpha_raw, 0L),  alpha_raw / 50, tolerance = 1e-14,
                 label = "iteration 0: /50")
    expect_equal(java_reduce_alpha(alpha_raw, 9L),  alpha_raw / 50, tolerance = 1e-14,
                 label = "iteration 9: /50")

    # 10 <= iteration < 20: divide by 10
    expect_equal(java_reduce_alpha(alpha_raw, 10L), alpha_raw / 10, tolerance = 1e-14,
                 label = "iteration 10: /10")
    expect_equal(java_reduce_alpha(alpha_raw, 19L), alpha_raw / 10, tolerance = 1e-14,
                 label = "iteration 19: /10")

    # 20 <= iteration < 50: divide by 3
    expect_equal(java_reduce_alpha(alpha_raw, 20L), alpha_raw / 3,  tolerance = 1e-14,
                 label = "iteration 20: /3")
    expect_equal(java_reduce_alpha(alpha_raw, 49L), alpha_raw / 3,  tolerance = 1e-14,
                 label = "iteration 49: /3")

    # iteration >= 50: no change
    expect_equal(java_reduce_alpha(alpha_raw, 50L),  alpha_raw, tolerance = 1e-14,
                 label = "iteration 50: x1")
    expect_equal(java_reduce_alpha(alpha_raw, 200L), alpha_raw, tolerance = 1e-14,
                 label = "iteration 200: x1")
})
