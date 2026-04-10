# tests/testthat/helper-java-bridge.R
#
# Shared fixtures and helpers for the Java-bridge test batteries
# (Tests 1-22).  Sourced automatically by testthat before each test file.
# ---------------------------------------------------------------------------

# ---- One-time Java bridge initialization (cached per session) --------------

.java_maxent_initialized <- local({
    result <- NULL
    function() {
        if (is.null(result)) {
            result <<- tryCatch(
                maxentcppCompTest::ensure_java_maxent(),
                error = function(e) FALSE
            )
        }
        result
    }
})

#' Skip a Test if the Java Bridge is Unavailable
#'
#' Skips the current test with an informative message when either
#' \pkg{rJava} is not installed or the \code{maxent_mini.jar} JAR cannot
#' be found or compiled.
skip_if_no_java <- function() {
    if (!requireNamespace("rJava", quietly = TRUE)) {
        testthat::skip("rJava not installed")
    }
    if (!.java_maxent_initialized()) {
        testthat::skip(
            "MaxentMini JAR not available ",
            "(Java SDK required — run inst/java/build.sh)"
        )
    }
}

# ---- Shared data helpers ---------------------------------------------------

#' Read the Mock bio1 Vector
#'
#' Returns the 100-element background bio1 vector in row-major order
#' (top row first, left-to-right within each row).  Uses
#' \code{maxentcpp::maxent_read_asc()} when available, otherwise parses
#' the ASC file directly.
.get_bio1_vec <- function() {
    paths <- maxentcppCompTest::mock_raster_paths()
    if (requireNamespace("maxentcpp", quietly = TRUE)) {
        g <- maxentcpp::maxent_read_asc(paths["bio1"])
        m <- maxentcpp::maxent_grid_to_matrix(g)
        as.numeric(t(m))
    } else {
        .parse_asc(paths["bio1"])
    }
}

#' Read the Mock bio2 Vector
.get_bio2_vec <- function() {
    paths <- maxentcppCompTest::mock_raster_paths()
    if (requireNamespace("maxentcpp", quietly = TRUE)) {
        g <- maxentcpp::maxent_read_asc(paths["bio2"])
        m <- maxentcpp::maxent_grid_to_matrix(g)
        as.numeric(t(m))
    } else {
        .parse_asc(paths["bio2"])
    }
}

#' Parse a 6-header-line ASC File to a Flat Numeric Vector
.parse_asc <- function(path) {
    lines      <- readLines(path, warn = FALSE)
    data_lines <- lines[7:length(lines)]
    as.numeric(unlist(strsplit(trimws(data_lines), "\\s+")))
}

#' Compute 0-Based Occurrence Indices from Mock Occurrences
.get_sample_indices <- function() {
    occ      <- maxentcppCompTest::mock_occurrences()
    nrows    <- 10L; ncols <- 10L
    xll      <- 0.0; yll   <- 0.0; cellsize <- 1.0
    vapply(seq_len(nrow(occ)), function(i) {
        row_i <- floor((yll + nrows * cellsize - occ$lat[i]) / cellsize)
        col_i <- floor((occ$lon[i] - xll) / cellsize)
        row_i <- max(0L, min(as.integer(row_i), nrows - 1L))
        col_i <- max(0L, min(as.integer(col_i), ncols - 1L))
        as.integer(row_i * ncols + col_i)
    }, integer(1L))
}

#' Build the Standard Mock Fixture Used Across All Test Batteries
#'
#' Returns a named list with all shared test data and hyper-parameters.
make_mock_fixture <- function() {
    list(
        bio1_vec        = .get_bio1_vec(),
        bio2_vec        = .get_bio2_vec(),
        sample_indices  = .get_sample_indices(),
        n               = 100L,
        n_samples       = 10L,
        beta_multiplier = 1.0,
        min_deviation   = 0.001,
        max_iter        = 500L,
        convergence     = 1e-5
    )
}

# ---- C++ FeaturedSpace builder ---------------------------------------------

#' Build and Optionally Train a C++ FeaturedSpace from the Mock Fixture
#'
#' @param fx      Named list as returned by \code{make_mock_fixture()}.
#' @param do_fit  If \code{TRUE} (default), trains to convergence.
#' @return External pointer to a (possibly trained) FeaturedSpace object.
.make_cpp_fs <- function(fx, do_fit = TRUE) {
    f1 <- maxentcpp::maxent_linear_feature(fx$bio1_vec, "bio1")
    f2 <- maxentcpp::maxent_linear_feature(fx$bio2_vec, "bio2")
    fs <- maxentcpp::maxent_featured_space(fx$n, fx$sample_indices,
                                            list(f1, f2))
    if (do_fit) {
        maxentcpp::maxent_fit(fs,
                              max_iter        = fx$max_iter,
                              convergence     = fx$convergence,
                              beta_multiplier = fx$beta_multiplier,
                              min_deviation   = fx$min_deviation)
    }
    fs
}
