test_that("compare_maxent_predictions returns correct structure", {
    set.seed(1)
    cpp_preds  <- runif(100)
    java_preds <- cpp_preds + rnorm(100, 0, 0.01)

    cmp <- compare_maxent_predictions(cpp_preds, java_preds)

    expect_type(cmp, "list")
    expect_named(cmp, c("pearson_cor", "spearman_cor",
                        "max_abs_diff_rank", "n", "agreement"))
})

test_that("compare_maxent_predictions length mismatch raises error", {
    expect_error(
        compare_maxent_predictions(runif(100), runif(50)),
        "same length"
    )
})

test_that("compare_maxent_predictions with identical vectors gives correlation 1", {
    preds <- runif(100)
    cmp   <- compare_maxent_predictions(preds, preds)

    expect_equal(cmp$pearson_cor,  1.0, tolerance = 1e-10)
    expect_equal(cmp$spearman_cor, 1.0, tolerance = 1e-10)
    expect_equal(cmp$max_abs_diff_rank, 0.0, tolerance = 1e-10)
    expect_true(cmp$agreement)
})

test_that("compare_maxent_predictions detects reversed order disagreement", {
    preds <- 1:100
    cmp   <- compare_maxent_predictions(preds, rev(preds))

    expect_equal(cmp$spearman_cor, -1.0, tolerance = 1e-10)
    expect_false(cmp$agreement)
})

test_that("compare_maxent_predictions n equals input length", {
    n     <- 100L
    cpp_p <- runif(n)
    java_p <- runif(n)
    cmp   <- compare_maxent_predictions(cpp_p, java_p)

    expect_equal(cmp$n, n)
})

test_that("compare_maxent_predictions agreement flag respects threshold", {
    set.seed(42)
    cpp_p  <- runif(100)
    java_p <- cpp_p + rnorm(100, 0, 0.001)  # almost identical

    cmp_strict  <- compare_maxent_predictions(cpp_p, java_p, cor_threshold = 0.999)
    cmp_lenient <- compare_maxent_predictions(cpp_p, java_p, cor_threshold = 0.5)

    # High noise-free correlation should pass lenient but may fail strict
    expect_true(cmp_lenient$agreement)
    expect_true(cmp_strict$spearman_cor > 0.99)
})

test_that("full comparison workflow: maxentcpp vs Java maxent", {
    skip_if_not_installed("maxentcpp")
    skip_if_not_installed("dismo")
    skip_if_not_installed("rJava")
    skip_if_not_installed("raster")

    jar_path <- file.path(system.file(package = "dismo"), "java", "maxent.jar")
    skip_if(!file.exists(jar_path), "maxent.jar not found")

    cpp_result  <- run_maxentcpp(max_iter = 500L)
    java_result <- run_maxent_java()

    expect_equal(cpp_result$n_points,  java_result$n_points)
    expect_equal(cpp_result$n_samples, java_result$n_samples)

    cmp <- compare_maxent_predictions(
        cpp_result$predictions,
        java_result$predictions,
        cor_threshold = 0.95
    )

    expect_true(cmp$spearman_cor >= 0.95,
                label = paste("Spearman correlation =", round(cmp$spearman_cor, 4),
                              "< 0.95"))
    expect_true(cmp$agreement)
})
