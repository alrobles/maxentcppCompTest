test_that("run_maxentcpp stops when maxentcpp is not installed", {
    skip_if(requireNamespace("maxentcpp", quietly = TRUE),
            "maxentcpp is installed; skipping absence check")

    expect_error(run_maxentcpp(), "maxentcpp")
})

test_that("run_maxentcpp returns correct structure", {
    skip_if_not_installed("maxentcpp")

    result <- run_maxentcpp(max_iter = 200L)

    expect_type(result, "list")
    expect_named(result, c("featured_space", "predictions", "fit",
                           "n_points", "n_samples"))
})

test_that("run_maxentcpp predictions cover 100 background cells", {
    skip_if_not_installed("maxentcpp")

    result <- run_maxentcpp(max_iter = 200L)

    expect_equal(result$n_points, 100L)
    expect_equal(length(result$predictions), 100L)
})

test_that("run_maxentcpp predictions are finite and non-negative", {
    skip_if_not_installed("maxentcpp")

    result <- run_maxentcpp(max_iter = 200L)

    expect_true(all(is.finite(result$predictions)))
    expect_true(all(result$predictions >= 0))
})

test_that("run_maxentcpp fit result has expected components", {
    skip_if_not_installed("maxentcpp")

    result <- run_maxentcpp(max_iter = 200L)
    fit    <- result$fit

    expect_type(fit, "list")
    expect_true("loss"       %in% names(fit))
    expect_true("entropy"    %in% names(fit))
    expect_true("iterations" %in% names(fit))
    expect_true("converged"  %in% names(fit))
    expect_true("lambdas"    %in% names(fit))
})

test_that("run_maxentcpp loss is finite", {
    skip_if_not_installed("maxentcpp")

    result <- run_maxentcpp(max_iter = 200L)

    expect_true(is.finite(result$fit$loss))
})

test_that("run_maxentcpp entropy is non-negative", {
    skip_if_not_installed("maxentcpp")

    result <- run_maxentcpp(max_iter = 200L)

    expect_true(result$fit$entropy >= 0)
})

test_that("run_maxentcpp records 10 occurrence samples", {
    skip_if_not_installed("maxentcpp")

    result <- run_maxentcpp(max_iter = 200L)

    expect_equal(result$n_samples, 10L)
})

test_that("run_maxentcpp predictions vary across background cells", {
    skip_if_not_installed("maxentcpp")

    result <- run_maxentcpp(max_iter = 200L)

    # Predictions should not all be identical
    expect_true(var(result$predictions) > 0)
})
