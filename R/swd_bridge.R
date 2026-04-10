#' Convert Mock Data to SWD Format
#'
#' Converts the package's mock raster data and occurrence records to the
#' Samples-With-Data (SWD) CSV format accepted by Java Maxent.
#'
#' SWD columns: \code{species, lon, lat, bio1, bio2}.
#' Environmental values are extracted at the occurrence point locations.
#'
#' @param output_path Character: path for the output CSV file.
#'   Defaults to a temporary file with extension \code{.csv}.
#' @return Invisibly, the character path to the written CSV file.
#' @seealso \code{\link{swd_to_features}}
#' @export
#' @examples
#' \dontrun{
#' swd_file <- mock_to_swd()
#' head(read.csv(swd_file))
#' }
mock_to_swd <- function(output_path = tempfile(fileext = ".csv")) {
    occ   <- mock_occurrences()
    paths <- mock_raster_paths()

    bio1_vals <- parse_asc_raster(paths["bio1"])
    bio2_vals <- parse_asc_raster(paths["bio2"])

    # Grid parameters match the mock data (10x10, cellsize 1, origin (0,0))
    nrows    <- 10L
    ncols    <- 10L
    xll      <- 0.0
    yll      <- 0.0
    cellsize <- 1.0

    rows_occ <- floor((yll + nrows * cellsize - occ$lat) / cellsize)
    cols_occ <- floor((occ$lon - xll) / cellsize)
    rows_occ <- pmax(0L, pmin(as.integer(rows_occ), nrows - 1L))
    cols_occ <- pmax(0L, pmin(as.integer(cols_occ), ncols - 1L))
    idx_occ  <- rows_occ * ncols + cols_occ + 1L   # convert to 1-based R index

    swd <- data.frame(
        species = occ$species,
        lon     = occ$lon,
        lat     = occ$lat,
        bio1    = bio1_vals[idx_occ],
        bio2    = bio2_vals[idx_occ],
        stringsAsFactors = FALSE
    )

    utils::write.csv(swd, output_path, row.names = FALSE)
    invisible(output_path)
}

#' Convert SWD CSV to Feature Vectors
#'
#' Parses a Samples-With-Data CSV file and returns a named list of
#' feature vectors, one per environmental variable column.
#'
#' @param swd_path Character: path to the SWD CSV file (as produced by
#'   \code{\link{mock_to_swd}}).
#' @return Named list with \code{species}, \code{lon}, \code{lat}, and one
#'   numeric vector per environmental variable column.
#' @seealso \code{\link{mock_to_swd}}
#' @export
#' @examples
#' \dontrun{
#' swd_file <- mock_to_swd()
#' feats    <- swd_to_features(swd_file)
#' str(feats$bio1)
#' }
swd_to_features <- function(swd_path) {
    swd      <- utils::read.csv(swd_path, stringsAsFactors = FALSE)
    env_cols <- setdiff(names(swd), c("species", "lon", "lat"))

    result <- list(
        species = swd$species,
        lon     = swd$lon,
        lat     = swd$lat
    )
    for (col in env_cols) {
        result[[col]] <- swd[[col]]
    }
    result
}

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

#' Parse an ASC Raster File to a Flat Numeric Vector
#'
#' Reads the data portion of an ASC raster file (skipping the 6 header lines)
#' and returns all cell values as a flat numeric vector in row-major order
#' (first row first, left to right within each row).
#'
#' @param path Character: path to the ASC file.
#' @return Numeric vector of cell values.
#' @keywords internal
parse_asc_raster <- function(path) {
    lines      <- readLines(path, warn = FALSE)
    data_lines <- lines[7:length(lines)]
    as.numeric(unlist(strsplit(trimws(data_lines), "\\s+")))
}
