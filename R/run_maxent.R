#' Run maxentcpp on Mock Data
#'
#' Trains a MaxEnt model using the C++ implementation (\pkg{maxentcpp})
#' on the package's mock 10-by-10 raster data and 10 occurrence records.
#' Returns the fitted model as a named list including predictions over
#' all background cells.
#'
#' @param max_iter  Maximum training iterations (default 500).
#' @param convergence  Convergence threshold (default 1e-5).
#' @param beta_multiplier  Regularization multiplier (default 1.0).
#'
#' @return A named list with:
#'   \describe{
#'     \item{featured_space}{The trained FeaturedSpace external pointer.}
#'     \item{predictions}{Numeric vector of raw predictions for all 100
#'       background cells.}
#'     \item{fit}{The list returned by \code{maxent_fit()}.}
#'     \item{n_points}{Total number of background points (100).}
#'     \item{n_samples}{Number of occurrence samples (10).}
#'   }
#'
#' @seealso \code{\link{run_maxent_java}}, \code{\link{compare_maxent_predictions}}
#' @export
#' @examples
#' \dontrun{
#' result <- run_maxentcpp()
#' hist(result$predictions)
#' }
run_maxentcpp <- function(max_iter       = 500L,
                          convergence    = 1e-5,
                          beta_multiplier = 1.0) {
    if (!requireNamespace("maxentcpp", quietly = TRUE)) {
        stop(
            "Package 'maxentcpp' is required. ",
            "Install it with remotes::install_github(",
            "'alrobles/Maxent', subdir = 'R-package')."
        )
    }

    paths <- mock_raster_paths()
    occ   <- mock_occurrences()

    # Read rasters via maxentcpp
    g1 <- maxentcpp::maxent_read_asc(paths["bio1"])
    g2 <- maxentcpp::maxent_read_asc(paths["bio2"])

    m1 <- maxentcpp::maxent_grid_to_matrix(g1)
    m2 <- maxentcpp::maxent_grid_to_matrix(g2)

    info <- maxentcpp::maxent_grid_info(g1)
    nrows    <- info$nrows
    ncols    <- info$ncols
    xll      <- info$xll
    yll      <- info$yll
    cellsize <- info$cellsize

    # Flatten matrices to background vectors (row-major)
    bio1_vec <- as.numeric(t(m1))
    bio2_vec <- as.numeric(t(m2))
    n <- length(bio1_vec)

    # Map occurrence lon/lat to 0-based background indices
    # ASC row 0 = northernmost; col 0 = westernmost
    # row index = floor((yll + nrows*cellsize - lat) / cellsize)
    # col index = floor((lon - xll) / cellsize)
    occ_indices <- vapply(seq_len(nrow(occ)), function(i) {
        row_i <- floor((yll + nrows * cellsize - occ$lat[i]) / cellsize)
        col_i <- floor((occ$lon[i] - xll) / cellsize)
        row_i <- max(0L, min(as.integer(row_i), nrows - 1L))
        col_i <- max(0L, min(as.integer(col_i), ncols - 1L))
        as.integer(row_i * ncols + col_i)
    }, integer(1L))

    # Build features (linear for both variables)
    f1 <- maxentcpp::maxent_linear_feature(bio1_vec, "bio1")
    f2 <- maxentcpp::maxent_linear_feature(bio2_vec, "bio2")

    # Create and train the FeaturedSpace
    fs  <- maxentcpp::maxent_featured_space(n, occ_indices, list(f1, f2))
    fit <- maxentcpp::maxent_fit(fs,
                                 max_iter        = as.integer(max_iter),
                                 convergence     = convergence,
                                 beta_multiplier = beta_multiplier)

    # Predict on all background points
    f1b  <- maxentcpp::maxent_linear_feature(bio1_vec, "bio1")
    f2b  <- maxentcpp::maxent_linear_feature(bio2_vec, "bio2")
    feat_mat <- cbind(
        vapply(seq_len(n), function(i) maxentcpp::maxent_feature_eval(f1b, i), numeric(1)),
        vapply(seq_len(n), function(i) maxentcpp::maxent_feature_eval(f2b, i), numeric(1))
    )
    preds <- maxentcpp::maxent_predict_model(fs, feat_mat)

    list(
        featured_space = fs,
        predictions    = preds,
        fit            = fit,
        n_points       = n,
        n_samples      = nrow(occ)
    )
}

#' Run Java Maxent (via dismo) on Mock Data
#'
#' Trains a MaxEnt model using the Java implementation via the \pkg{dismo}
#' package on the package's mock 10-by-10 raster data and 10 occurrence
#' records.  Returns the fitted model and predictions.
#'
#' @return A named list with:
#'   \describe{
#'     \item{model}{The \code{MaxEnt} model object from \code{dismo::maxent()}.}
#'     \item{predictions}{Numeric vector of predictions for all 100 raster cells.}
#'     \item{n_points}{Number of background cells used (100).}
#'     \item{n_samples}{Number of occurrence records (10).}
#'   }
#'
#' @seealso \code{\link{run_maxentcpp}}, \code{\link{compare_maxent_predictions}}
#' @export
#' @examples
#' \dontrun{
#' result <- run_maxent_java()
#' hist(result$predictions)
#' }
run_maxent_java <- function() {
    if (!requireNamespace("dismo", quietly = TRUE)) {
        stop("Package 'dismo' is required. Install it with install.packages('dismo').")
    }
    if (!requireNamespace("rJava", quietly = TRUE)) {
        stop("Package 'rJava' is required. Install it with install.packages('rJava').")
    }

    paths <- mock_raster_paths()
    occ   <- mock_occurrences()
    occ_coords <- occ[, c("lon", "lat")]

    # Run Java Maxent via dismo (requires maxent.jar in the dismo java folder)
    jar_path <- file.path(system.file(package = "dismo"), "java", "maxent.jar")
    if (!file.exists(jar_path)) {
        stop("maxent.jar not found at: ", jar_path,
             "\nDownload maxent.jar from https://biodiversityinformatics.amnh.org/open_source/maxent/",
             " and place it in the dismo java folder.")
    }

    # Load rasters with raster for dismo compatibility
    if (!requireNamespace("raster", quietly = TRUE)) {
        stop("Package 'raster' is required. Install it with install.packages('raster').")
    }

    env_raster <- raster::stack(paths["bio1"], paths["bio2"])

    model <- dismo::maxent(
        x        = env_raster,
        p        = occ_coords,
        silent   = TRUE,
        args     = c("nowarnings", "notooltips",
                     "maximumiterations=500",
                     "convergencethreshold=1e-5",
                     "betamultiplier=1.0")
    )

    preds_raster <- dismo::predict(model, env_raster)
    preds_vec    <- as.numeric(raster::values(preds_raster))

    list(
        model      = model,
        predictions = preds_vec,
        n_points   = length(preds_vec),
        n_samples  = nrow(occ)
    )
}
