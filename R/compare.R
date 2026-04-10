#' Compare Predictions from maxentcpp and Java Maxent
#'
#' Computes several agreement metrics between the raw prediction vectors
#' produced by \code{\link{run_maxentcpp}} and \code{\link{run_maxent_java}}.
#' Both vectors are rank-normalized before comparison so that differences in
#' absolute scale (which are expected between implementations) do not mask
#' ordinal agreement.
#'
#' @param cpp_preds  Numeric vector of predictions from the C++ implementation.
#' @param java_preds Numeric vector of predictions from the Java implementation.
#'   Must have the same length as \code{cpp_preds}.
#' @param cor_threshold Minimum acceptable Spearman rank correlation
#'   (default 0.95). Used only when called from tests.
#'
#' @return A named list with:
#'   \describe{
#'     \item{pearson_cor}{Pearson correlation of the two prediction vectors.}
#'     \item{spearman_cor}{Spearman rank correlation.}
#'     \item{max_abs_diff_rank}{Maximum absolute difference of rank-normalized
#'       predictions (values in [0, 1]).}
#'     \item{n}{Number of non-missing paired predictions used in the
#'       comparison.}
#'     \item{agreement}{Logical: \code{TRUE} if Spearman correlation is
#'       at least \code{cor_threshold}.}
#'   }
#'
#' @seealso \code{\link{run_maxentcpp}}, \code{\link{run_maxent_java}}
#' @export
#' @examples
#' \dontrun{
#' cpp  <- run_maxentcpp()
#' java <- run_maxent_java()
#' cmp  <- compare_maxent_predictions(cpp$predictions, java$predictions)
#' cmp$spearman_cor
#' cmp$agreement
#' }
compare_maxent_predictions <- function(cpp_preds,
                                       java_preds,
                                       cor_threshold = 0.95) {
    if (length(cpp_preds) != length(java_preds)) {
        stop("cpp_preds and java_preds must have the same length.")
    }

    valid <- stats::complete.cases(cpp_preds, java_preds)
    cpp_preds <- cpp_preds[valid]
    java_preds <- java_preds[valid]
    n <- length(cpp_preds)

    if (n < 2) {
        stop("Need at least 2 non-missing paired predictions to compare.")
    }

    pearson_cor  <- stats::cor(cpp_preds, java_preds, method = "pearson")
    spearman_cor <- stats::cor(cpp_preds, java_preds, method = "spearman")

    # Rank-normalize to [0, 1]
    rank_norm <- function(x) {
        r <- rank(x, ties.method = "average")
        (r - 1) / (length(r) - 1)
    }
    cpp_rank  <- rank_norm(cpp_preds)
    java_rank <- rank_norm(java_preds)

    max_abs_diff_rank <- max(abs(cpp_rank - java_rank))

    list(
        pearson_cor        = pearson_cor,
        spearman_cor       = spearman_cor,
        max_abs_diff_rank  = max_abs_diff_rank,
        n                  = n,
        agreement          = spearman_cor >= cor_threshold
    )
}
