test_that("mock_raster_paths returns named character vector with two elements", {
    paths <- mock_raster_paths()

    expect_type(paths, "character")
    expect_length(paths, 2)
    expect_named(paths, c("bio1", "bio2"))
})

test_that("bio1.asc file exists and is readable", {
    paths <- mock_raster_paths()

    expect_true(file.exists(paths["bio1"]))

    lines <- readLines(paths["bio1"])
    expect_true(length(lines) >= 7)  # 6 header lines + at least 1 data row
})

test_that("bio2.asc file exists and is readable", {
    paths <- mock_raster_paths()

    expect_true(file.exists(paths["bio2"]))

    lines <- readLines(paths["bio2"])
    expect_true(length(lines) >= 7)
})

test_that("bio1.asc header declares a 10x10 grid", {
    paths <- mock_raster_paths()
    lines <- readLines(paths["bio1"])

    header <- tolower(trimws(lines[1:6]))
    ncols_line <- grep("^ncols", header, value = TRUE)
    nrows_line <- grep("^nrows", header, value = TRUE)

    ncols_val <- as.integer(strsplit(ncols_line, "\\s+")[[1]][2])
    nrows_val <- as.integer(strsplit(nrows_line, "\\s+")[[1]][2])

    expect_equal(ncols_val, 10L)
    expect_equal(nrows_val, 10L)
})

test_that("bio2.asc header declares a 10x10 grid", {
    paths <- mock_raster_paths()
    lines <- readLines(paths["bio2"])

    header <- tolower(trimws(lines[1:6]))
    ncols_line <- grep("^ncols", header, value = TRUE)
    nrows_line <- grep("^nrows", header, value = TRUE)

    ncols_val <- as.integer(strsplit(ncols_line, "\\s+")[[1]][2])
    nrows_val <- as.integer(strsplit(nrows_line, "\\s+")[[1]][2])

    expect_equal(ncols_val, 10L)
    expect_equal(nrows_val, 10L)
})

test_that("bio1.asc contains 100 numeric values", {
    paths <- mock_raster_paths()
    lines <- readLines(paths["bio1"])

    data_lines <- lines[7:length(lines)]
    vals <- as.numeric(unlist(strsplit(paste(data_lines, collapse = " "), "\\s+")))
    vals <- vals[!is.na(vals)]

    expect_equal(length(vals), 100L)
})

test_that("bio2.asc contains 100 numeric values", {
    paths <- mock_raster_paths()
    lines <- readLines(paths["bio2"])

    data_lines <- lines[7:length(lines)]
    vals <- as.numeric(unlist(strsplit(paste(data_lines, collapse = " "), "\\s+")))
    vals <- vals[!is.na(vals)]

    expect_equal(length(vals), 100L)
})

test_that("mock_occurrences returns a data frame with 10 rows", {
    occ <- mock_occurrences()

    expect_s3_class(occ, "data.frame")
    expect_equal(nrow(occ), 10L)
})

test_that("mock_occurrences has required columns", {
    occ <- mock_occurrences()

    expect_true(all(c("species", "lon", "lat") %in% names(occ)))
})

test_that("occurrence coordinates fall within the raster extent", {
    occ <- mock_occurrences()

    # Raster covers lon [0, 10], lat [0, 10]
    expect_true(all(occ$lon >= 0 & occ$lon <= 10))
    expect_true(all(occ$lat >= 0 & occ$lat <= 10))
})

test_that("all 10 occurrences belong to the same species", {
    occ <- mock_occurrences()

    expect_equal(length(unique(occ$species)), 1L)
})

test_that("Java source files exist in inst/java", {
    java_dir <- system.file("java", "com", "macfaq", "io",
                            package = "maxentcppCompTest")

    expect_true(nchar(java_dir) > 0)
    expect_true(dir.exists(java_dir))

    lei_file <- file.path(java_dir, "LittleEndianInputStream.java")
    leo_file <- file.path(java_dir, "LittleEndianOutputStream.java")

    expect_true(file.exists(lei_file))
    expect_true(file.exists(leo_file))
})
