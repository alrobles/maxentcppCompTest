#' Get Paths to Mock Raster Files
#'
#' Returns the file system paths to the two bundled 10-by-10 ESRI ASCII
#' raster files (bio1 and bio2) included in this package's \code{extdata}
#' directory.
#'
#' @return A named character vector with elements \code{"bio1"} and
#'   \code{"bio2"}, each giving the full path to the corresponding
#'   \code{.asc} file.
#' @export
#' @examples
#' paths <- mock_raster_paths()
#' file.exists(paths["bio1"])
#' file.exists(paths["bio2"])
mock_raster_paths <- function() {
    extdata <- system.file("extdata", package = "maxentcppCompTest")
    c(
        bio1 = file.path(extdata, "bio1.asc"),
        bio2 = file.path(extdata, "bio2.asc")
    )
}

#' Load Mock Occurrence Records
#'
#' Reads the bundled occurrence CSV file and returns a data frame with
#' species name, longitude, and latitude for 10 mock occurrence points.
#'
#' @return A \code{data.frame} with columns \code{species}, \code{lon},
#'   and \code{lat} (10 rows).
#' @export
#' @examples
#' occ <- mock_occurrences()
#' head(occ)
mock_occurrences <- function() {
    csv_path <- system.file("extdata", "occurrences.csv",
                            package = "maxentcppCompTest")
    read.csv(csv_path, stringsAsFactors = FALSE)
}
