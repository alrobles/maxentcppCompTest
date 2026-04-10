test_that("run_maxent_java stops when dismo is not installed", {
    skip_if(requireNamespace("dismo", quietly = TRUE),
            "dismo is installed; skipping absence check")

    expect_error(run_maxent_java(), "dismo")
})

test_that("run_maxent_java returns correct structure", {
    skip_if_not_installed("dismo")
    skip_if_not_installed("rJava")
    skip_if_not_installed("raster")

    jar_path <- file.path(system.file(package = "dismo"), "java", "maxent.jar")
    skip_if(!file.exists(jar_path), "maxent.jar not found")

    result <- run_maxent_java()

    expect_type(result, "list")
    expect_named(result, c("model", "predictions", "n_points", "n_samples"))
})

test_that("run_maxent_java predictions cover 100 cells", {
    skip_if_not_installed("dismo")
    skip_if_not_installed("rJava")
    skip_if_not_installed("raster")

    jar_path <- file.path(system.file(package = "dismo"), "java", "maxent.jar")
    skip_if(!file.exists(jar_path), "maxent.jar not found")

    result <- run_maxent_java()

    expect_equal(result$n_points, 100L)
    expect_equal(length(result$predictions), 100L)
})

test_that("run_maxent_java predictions are in [0, 1]", {
    skip_if_not_installed("dismo")
    skip_if_not_installed("rJava")
    skip_if_not_installed("raster")

    jar_path <- file.path(system.file(package = "dismo"), "java", "maxent.jar")
    skip_if(!file.exists(jar_path), "maxent.jar not found")

    result <- run_maxent_java()

    non_na <- result$predictions[!is.na(result$predictions)]
    expect_true(all(non_na >= 0 & non_na <= 1))
})

test_that("run_maxent_java records 10 occurrence samples", {
    skip_if_not_installed("dismo")
    skip_if_not_installed("rJava")
    skip_if_not_installed("raster")

    jar_path <- file.path(system.file(package = "dismo"), "java", "maxent.jar")
    skip_if(!file.exists(jar_path), "maxent.jar not found")

    result <- run_maxent_java()

    expect_equal(result$n_samples, 10L)
})
