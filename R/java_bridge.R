#' Ensure Java Maxent Mini JAR is Available
#'
#' Checks that \pkg{rJava} is installed and that the \code{MaxentMini.jar}
#' can be found or compiled from source.  If the JAR is absent but
#' \code{javac} is available, the function attempts to compile
#' \code{MaxentMini.java} using the bundled \code{build.sh} script.
#'
#' This function is idempotent: subsequent calls within the same R session
#' return immediately after the first successful initialization.
#'
#' @return Logical: \code{TRUE} if the JAR is loaded and the JVM is ready,
#'   \code{FALSE} otherwise (with an informative \code{message()}).
#' @export
ensure_java_maxent <- function() {
    if (!requireNamespace("rJava", quietly = TRUE)) {
        message("rJava is not installed. Install it with install.packages('rJava').")
        return(FALSE)
    }

    java_dir <- system.file("java", package = "maxentcppCompTest")
    if (!nzchar(java_dir)) {
        message("Cannot locate inst/java/ directory in package 'maxentcppCompTest'.")
        return(FALSE)
    }

    jar_path <- file.path(java_dir, "maxent_mini.jar")

    # Try to compile if JAR is missing
    if (!file.exists(jar_path)) {
        src_file   <- file.path(java_dir, "MaxentMini.java")
        build_sh   <- file.path(java_dir, "build.sh")

        if (!file.exists(src_file)) {
            message("MaxentMini.java not found in ", java_dir)
            return(FALSE)
        }

        message("maxent_mini.jar not found — attempting to compile MaxentMini.java ...")

        ok <- tryCatch({
            if (file.exists(build_sh)) {
                status <- system2("bash", args = build_sh,
                                  stdout = TRUE, stderr = TRUE)
            } else {
                class_dir <- file.path(java_dir, "classes")
                dir.create(class_dir, showWarnings = FALSE, recursive = TRUE)
                status <- system2("javac",
                                  args = c("-d", class_dir, src_file),
                                  stdout = TRUE, stderr = TRUE)
                if (isTRUE(attr(status, "status") == 0) ||
                    is.null(attr(status, "status"))) {
                    status <- system2("jar",
                                     args = c("cf", jar_path, "-C", class_dir, "."),
                                     stdout = TRUE, stderr = TRUE)
                }
            }
            file.exists(jar_path)
        }, error = function(e) {
            message("Compilation error: ", conditionMessage(e))
            FALSE
        })

        if (!isTRUE(ok)) {
            message("Could not compile MaxentMini.java. ",
                    "Please run inst/java/build.sh manually (requires JDK >= 8).")
            return(FALSE)
        }
    }

    # Initialize JVM and load the JAR
    tryCatch({
        rJava::.jinit()
        rJava::.jaddClassPath(jar_path)
        TRUE
    }, error = function(e) {
        message("JVM initialization failed: ", conditionMessage(e))
        FALSE
    })
}

# ---------------------------------------------------------------------------
# Linear feature helpers
# ---------------------------------------------------------------------------

#' Evaluate Java Linear Feature for All Points
#'
#' Creates a linear feature from a numeric vector and evaluates it at every
#' point using the Java \code{MaxentMini} implementation.
#'
#' Formula: \code{(values[i] - min) / (max - min)}.
#' When \code{min == max} the feature is identically 0.
#'
#' @param values Numeric vector of raw environmental values (length \code{n}).
#' @param name   Character string (for documentation only; not used internally).
#' @return Numeric vector of normalized feature values, length \code{n}.
#' @export
java_linear_feature_eval <- function(values, name = "") {
    values <- as.double(values)
    result <- rJava::.jcall(
        "density/MaxentMini", "[D", "evalLinearFeature",
        values
    )
    as.numeric(result)
}

# ---------------------------------------------------------------------------
# Sample statistics
# ---------------------------------------------------------------------------

#' Compute Java Sample Expectation
#'
#' Returns the mean of the normalized linear feature over the occurrence
#' sample indices, computed by the Java \code{MaxentMini} implementation.
#'
#' @param values        Numeric vector of raw environmental values.
#' @param sample_indices Integer vector of 0-based occurrence point indices.
#' @return Numeric scalar: sample expectation.
#' @export
java_sample_expectation <- function(values, sample_indices) {
    rJava::.jcall(
        "density/MaxentMini", "D", "computeSampleExpectation",
        as.double(values),
        as.integer(sample_indices)
    )
}

#' Compute Java Sample Deviation
#'
#' Returns the population standard deviation of the normalized linear feature
#' over the occurrence sample indices, computed by the Java
#' \code{MaxentMini} implementation.
#'
#' @param values        Numeric vector of raw environmental values.
#' @param sample_indices Integer vector of 0-based occurrence point indices.
#' @return Numeric scalar: sample standard deviation.
#' @export
java_sample_deviation <- function(values, sample_indices) {
    rJava::.jcall(
        "density/MaxentMini", "D", "computeSampleDeviation",
        as.double(values),
        as.integer(sample_indices)
    )
}

# ---------------------------------------------------------------------------
# Optimizer building blocks
# ---------------------------------------------------------------------------

#' Compute Java goodAlpha
#'
#' Computes the goodAlpha closed-form coordinate-ascent step size.
#' Mirrors \code{Sequential.goodAlpha()} from the Java Maxent source.
#'
#' @param N1     Sample expectation (empirical mean over occurrences, in [0,1]).
#' @param W1     Model expectation (density-weighted mean, in [0,1]).
#' @param beta   Regularization parameter (sample deviation * betaMultiplier / sqrt(n)).
#' @param lambda Current lambda value for this feature.
#' @return Numeric scalar: the goodAlpha step size.
#' @export
java_good_alpha <- function(N1, W1, beta, lambda) {
    rJava::.jcall(
        "density/MaxentMini", "D", "computeGoodAlpha",
        as.double(N1),
        as.double(W1),
        as.double(beta),
        as.double(lambda)
    )
}

#' Compute Java reduceAlpha
#'
#' Applies the iteration-dependent damping schedule:
#' \itemize{
#'   \item iteration < 10: \code{alpha / 50}
#'   \item 10 <= iteration < 20: \code{alpha / 10}
#'   \item 20 <= iteration < 50: \code{alpha / 3}
#'   \item iteration >= 50: \code{alpha} (unchanged)
#' }
#'
#' @param alpha     Undamped step size.
#' @param iteration Current 0-based iteration index.
#' @return Numeric scalar: damped step size.
#' @export
java_reduce_alpha <- function(alpha, iteration) {
    rJava::.jcall(
        "density/MaxentMini", "D", "computeReduceAlpha",
        as.double(alpha),
        as.integer(iteration)
    )
}

# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

#' Get Java Initial Loss
#'
#' Returns the loss value before any training iterations (all lambdas = 0).
#' The expected value is \code{log(n)} where \code{n} is the number of
#' background points.
#'
#' @param bio1_vec       Numeric vector of bio1 values (length \code{n}).
#' @param bio2_vec       Numeric vector of bio2 values (length \code{n}).
#' @param sample_indices Integer vector of 0-based occurrence indices.
#' @param beta_multiplier Regularization multiplier (default 1.0).
#' @param min_deviation  Minimum sample deviation floor (default 0.001).
#' @return Numeric scalar: initial loss.
#' @export
java_get_initial_loss <- function(bio1_vec, bio2_vec, sample_indices,
                                   beta_multiplier = 1.0, min_deviation = 0.001) {
    rJava::.jcall(
        "density/MaxentMini", "D", "getInitialLoss",
        as.double(bio1_vec),
        as.double(bio2_vec),
        as.integer(sample_indices),
        as.double(beta_multiplier),
        as.double(min_deviation)
    )
}

#' Train Java MaxEnt Featured Space
#'
#' Runs the goodAlpha-based sequential coordinate-ascent MaxEnt optimizer
#' (mirroring the C++ implementation) and returns training results.
#'
#' @param bio1_vec       Numeric vector of bio1 values (all background points).
#' @param bio2_vec       Numeric vector of bio2 values (all background points).
#' @param sample_indices Integer vector of 0-based occurrence indices.
#' @param beta_multiplier Regularization multiplier (default 1.0).
#' @param min_deviation  Minimum sample deviation floor (default 0.001).
#' @param max_iter       Maximum training iterations (default 500).
#' @param convergence    Convergence threshold (default 1e-5).
#' @return Named list with:
#'   \describe{
#'     \item{loss}{Final regularized loss.}
#'     \item{entropy}{Shannon entropy of the trained distribution.}
#'     \item{gain}{\code{log(n) - loss}.}
#'     \item{lambdas}{Numeric vector of length 2: final lambda values.}
#'     \item{iterations}{Number of training iterations completed.}
#'   }
#' @export
java_featured_space_train <- function(bio1_vec, bio2_vec, sample_indices,
                                       beta_multiplier = 1.0, min_deviation = 0.001,
                                       max_iter = 500L, convergence = 1e-5) {
    result <- rJava::.jcall(
        "density/MaxentMini", "[D", "trainModel",
        as.double(bio1_vec),
        as.double(bio2_vec),
        as.integer(sample_indices),
        as.double(beta_multiplier),
        as.double(min_deviation),
        as.integer(max_iter),
        as.double(convergence)
    )
    result <- as.numeric(result)
    list(
        loss       = result[1],
        entropy    = result[2],
        gain       = result[3],
        lambdas    = result[4:5],
        iterations = as.integer(result[6])
    )
}

#' Train Java Model for Exactly N Iterations
#'
#' Trains without any convergence check, running exactly \code{n_iterations}
#' steps of the goodAlpha-based optimizer.  Useful for comparing per-iteration
#' loss trajectories between Java and C++.
#'
#' @param bio1_vec       Numeric vector of bio1 values.
#' @param bio2_vec       Numeric vector of bio2 values.
#' @param sample_indices Integer vector of 0-based occurrence indices.
#' @param n_iterations   Number of optimizer steps to take.
#' @param beta_multiplier Regularization multiplier (default 1.0).
#' @param min_deviation  Minimum sample deviation floor (default 0.001).
#' @return Named list with: \code{loss}, \code{entropy}, \code{lambdas}.
#' @export
java_train_n_iterations <- function(bio1_vec, bio2_vec, sample_indices,
                                     n_iterations,
                                     beta_multiplier = 1.0, min_deviation = 0.001) {
    result <- rJava::.jcall(
        "density/MaxentMini", "[D", "trainNIterations",
        as.double(bio1_vec),
        as.double(bio2_vec),
        as.integer(sample_indices),
        as.double(beta_multiplier),
        as.double(min_deviation),
        as.integer(n_iterations)
    )
    result <- as.numeric(result)
    list(
        loss    = result[1],
        entropy = result[2],
        lambdas = result[3:4]
    )
}

# ---------------------------------------------------------------------------
# Post-training queries
# ---------------------------------------------------------------------------

#' Get Java Density Weights
#'
#' Returns normalized density weights for all background points after
#' training to convergence.  Weights sum to 1 and represent the
#' probability \eqn{p_i = \exp(LP_i) / Z} under the trained Gibbs
#' distribution.
#'
#' @param bio1_vec       Numeric vector of bio1 values.
#' @param bio2_vec       Numeric vector of bio2 values.
#' @param sample_indices Integer vector of 0-based occurrence indices.
#' @param beta_multiplier Regularization multiplier (default 1.0).
#' @param min_deviation  Minimum sample deviation floor (default 0.001).
#' @param max_iter       Maximum training iterations (default 500).
#' @param convergence    Convergence threshold (default 1e-5).
#' @return Numeric vector of length \code{n}: normalized density weights.
#' @export
java_get_density <- function(bio1_vec, bio2_vec, sample_indices,
                              beta_multiplier = 1.0, min_deviation = 0.001,
                              max_iter = 500L, convergence = 1e-5) {
    result <- rJava::.jcall(
        "density/MaxentMini", "[D", "getWeights",
        as.double(bio1_vec),
        as.double(bio2_vec),
        as.integer(sample_indices),
        as.double(beta_multiplier),
        as.double(min_deviation),
        as.integer(max_iter),
        as.double(convergence)
    )
    as.numeric(result)
}

#' Get Java Sample Expectations
#'
#' Returns the empirical sample expectations for both features immediately
#' after FeaturedSpace construction (before training).
#'
#' @param bio1_vec       Numeric vector of bio1 values.
#' @param bio2_vec       Numeric vector of bio2 values.
#' @param sample_indices Integer vector of 0-based occurrence indices.
#' @param beta_multiplier Regularization multiplier (default 1.0).
#' @param min_deviation  Minimum sample deviation floor (default 0.001).
#' @return Numeric vector of length 2: sample expectations for bio1 and bio2.
#' @export
java_get_sample_expectations <- function(bio1_vec, bio2_vec, sample_indices,
                                          beta_multiplier = 1.0, min_deviation = 0.001) {
    result <- rJava::.jcall(
        "density/MaxentMini", "[D", "getSampleExpectations",
        as.double(bio1_vec),
        as.double(bio2_vec),
        as.integer(sample_indices),
        as.double(beta_multiplier),
        as.double(min_deviation)
    )
    as.numeric(result)
}

# ---------------------------------------------------------------------------
# Cloglog transform
# ---------------------------------------------------------------------------

#' Apply Java Cloglog Transformation
#'
#' Applies the complementary log-log transformation used in Maxent v3.4+:
#' \deqn{cloglog(x) = 1 - \exp(-x \cdot \exp(H))}
#' where \eqn{x} is the raw prediction (normalized density weight) and
#' \eqn{H} is the Shannon entropy of the trained distribution.
#'
#' @param raw_values Numeric vector of raw prediction values (normalized
#'   density weights, should sum to 1 over the background).
#' @param entropy    Scalar: Shannon entropy of the trained distribution.
#' @return Numeric vector of cloglog values in [0, 1].
#' @export
java_cloglog_transform <- function(raw_values, entropy) {
    result <- rJava::.jcall(
        "density/MaxentMini", "[D", "cloglogTransform",
        as.double(raw_values),
        as.double(entropy)
    )
    as.numeric(result)
}
