package density;

/**
 * Minimal standalone MaxEnt implementation for side-by-side testing with
 * the C++ maxentcpp package.
 *
 * <p>Implements linear features and cloglog output using a goodAlpha-based
 * sequential coordinate ascent optimizer that mirrors the C++ implementation
 * in {@code featured_space.hpp}. This class is called via rJava from R test
 * code in the maxentcppCompTest validation package.</p>
 *
 * <p>Key algorithms faithfully reproduced from {@code density/Sequential.java}
 * and {@code density/FeaturedSpace.java} in the alrobles/Maxent repository:</p>
 * <ul>
 *   <li>Linear feature normalization: {@code (val - min) / (max - min)}</li>
 *   <li>Sample expectations and deviations (regularization betas)</li>
 *   <li>goodAlpha closed-form step size</li>
 *   <li>reduceAlpha damping schedule ({@code /50, /10, /3, x1})</li>
 *   <li>Density normalization: {@code density[i] = exp(LP[i])}</li>
 *   <li>Loss: {@code -sum(lambda*sampleExp) + log(dNorm) + reg}</li>
 *   <li>Cloglog: {@code 1 - exp(-raw * exp(entropy))}</li>
 * </ul>
 */
public class MaxentMini {

    // -------------------------------------------------------------------------
    // Constants
    // -------------------------------------------------------------------------
    static final double EPS = 1e-6;
    static final int CONVERGENCE_TEST_FREQ = 20;

    // -------------------------------------------------------------------------
    // Instance fields
    // -------------------------------------------------------------------------
    final int n;
    final int numSamples;
    final int numFeatures;
    final int[] sampleIndices;

    /** Normalized feature values: featVals[j][i] = f_j(i). */
    final double[][] featVals;

    /** Per-feature normalization metadata. */
    final double[] featureMin;
    final double[] featureMax;
    final double[] featureScale;

    /** Current lambda (weight) for each feature. */
    double[] lambda;

    /** Unnormalized density: density[i] = exp(sum_j lambda[j] * f_j(i)). */
    double[] density;

    /** sum of density[i] over all background points. */
    double densityNormalizer;

    /** Density-weighted expectation of each feature. */
    double[] expectation;

    /** Empirical mean of each feature over occurrence samples. */
    double[] sampleExpectation;

    /** Regularization parameter for each feature (beta). */
    double[] sampleDeviation;

    /** L1 regularization term: sum_j |lambda[j]| * sampleDeviation[j]. */
    double reg;

    /** Number of training iterations completed. */
    int iterations;

    // -------------------------------------------------------------------------
    // Constructor
    // -------------------------------------------------------------------------

    /**
     * Creates a MaxentMini instance from two raw environmental variable vectors
     * and a set of occurrence sample indices.
     *
     * @param feat0           Raw values for feature 0 (e.g., bio1), length n.
     * @param feat1           Raw values for feature 1 (e.g., bio2), length n.
     * @param sampleIdx       0-based indices of occurrence points in feat0/feat1.
     * @param betaMultiplier  Regularization multiplier (typically 1.0).
     * @param minDeviation    Minimum sample deviation floor (typically 0.001).
     */
    public MaxentMini(double[] feat0, double[] feat1,
                      int[] sampleIdx,
                      double betaMultiplier, double minDeviation) {
        this.n           = feat0.length;
        this.numSamples  = sampleIdx.length;
        this.numFeatures = 2;
        this.sampleIndices = sampleIdx.clone();

        featureMin   = new double[numFeatures];
        featureMax   = new double[numFeatures];
        featureScale = new double[numFeatures];
        featVals     = new double[numFeatures][n];

        buildFeature(0, feat0);
        buildFeature(1, feat1);

        lambda          = new double[numFeatures];
        density         = new double[n];
        expectation     = new double[numFeatures];
        sampleExpectation = new double[numFeatures];
        sampleDeviation   = new double[numFeatures];

        // Uniform initial distribution: density[i] = 1, dNorm = n
        for (int i = 0; i < n; i++) density[i] = 1.0;
        densityNormalizer = n;

        setSampleExpectations();

        // Compute regularization betas
        for (int j = 0; j < numFeatures; j++) {
            double mean = sampleExpectation[j];
            double var  = 0.0;
            for (int k = 0; k < numSamples; k++) {
                double diff = featVals[j][sampleIndices[k]] - mean;
                var += diff * diff;
            }
            var /= numSamples;
            double dev = Math.sqrt(var);
            if (dev < minDeviation) dev = minDeviation;
            sampleDeviation[j] = betaMultiplier * dev / Math.sqrt(numSamples);
        }

        updateExpectations();
        reg        = 0.0;
        iterations = 0;
    }

    // -------------------------------------------------------------------------
    // Private / package helpers
    // -------------------------------------------------------------------------

    /**
     * Normalizes a raw feature vector: {@code (raw[i] - min) / (max - min)}.
     * When {@code min == max}, {@code scale = 1} so the result is 0.
     */
    private void buildFeature(int j, double[] raw) {
        double min = raw[0], max = raw[0];
        for (double v : raw) {
            if (v < min) min = v;
            if (v > max) max = v;
        }
        featureMin[j]   = min;
        featureMax[j]   = max;
        featureScale[j] = (max == min) ? 1.0 : (max - min);
        for (int i = 0; i < n; i++) {
            featVals[j][i] = (raw[i] - min) / featureScale[j];
        }
    }

    private void setSampleExpectations() {
        for (int j = 0; j < numFeatures; j++) {
            double sum = 0.0;
            for (int k = 0; k < numSamples; k++) {
                sum += featVals[j][sampleIndices[k]];
            }
            sampleExpectation[j] = sum / numSamples;
        }
    }

    private void updateExpectations() {
        for (int j = 0; j < numFeatures; j++) {
            double sum = 0.0;
            for (int i = 0; i < n; i++) sum += density[i] * featVals[j][i];
            expectation[j] = sum / densityNormalizer;
        }
    }

    private double getN1() {
        double s = 0.0;
        for (int j = 0; j < numFeatures; j++) {
            s -= lambda[j] * sampleExpectation[j];
        }
        return s;
    }

    // -------------------------------------------------------------------------
    // Public instance API (mirrors FeaturedSpace / Sequential)
    // -------------------------------------------------------------------------

    /** Returns the current regularized loss. */
    public double getLoss() {
        return getN1() + Math.log(densityNormalizer) + reg;
    }

    /**
     * Computes the goodAlpha closed-form step size for feature j.
     * Mirrors {@code Sequential.goodAlpha()} in the Java Maxent source.
     */
    double goodAlpha(int j) {
        double W1  = expectation[j];
        double W0  = 1.0 - W1;
        double N1  = sampleExpectation[j];
        double N0  = 1.0 - N1;
        double lam  = lambda[j];
        double beta = sampleDeviation[j];

        if (W0 < EPS || W1 < EPS) return 0.0;

        double alpha;
        if (N1 - beta > EPS) {
            alpha = Math.log((N1 - beta) * W0 / ((N0 + beta) * W1));
            if (alpha + lam > 0.0) return alpha;
        }
        if (N0 - beta > EPS) {
            alpha = Math.log((N1 + beta) * W0 / ((N0 - beta) * W1));
            if (alpha + lam < 0.0) return alpha;
        }
        return -lam;
    }

    /**
     * Applies iteration-dependent damping to a step size.
     * Mirrors {@code Sequential.reduceAlpha()}.
     */
    double reduceAlpha(double alpha, int iteration) {
        if (iteration < 10) return alpha / 50.0;
        if (iteration < 20) return alpha / 10.0;
        if (iteration < 50) return alpha / 3.0;
        return alpha;
    }

    /**
     * Returns a bound on the loss decrease achievable by updating feature j.
     * Mirrors {@code Sequential.deltaLossBound()}.
     */
    double deltaLossBound(int j) {
        double alpha = goodAlpha(j);
        if (alpha == 0.0) return 0.0;

        double W0   = 1.0 - expectation[j];
        double W1   = expectation[j];
        double N1   = sampleExpectation[j];
        double beta = sampleDeviation[j];
        double lam  = lambda[j];
        double arg  = W0 + W1 * Math.exp(alpha);
        if (arg <= 0.0) return 0.0;
        double bound = -N1 * alpha
                + Math.log(arg)
                + beta * (Math.abs(lam + alpha) - Math.abs(lam));
        return Double.isNaN(bound) ? 0.0 : bound;
    }

    /** Returns the index of the feature with the smallest deltaLossBound. */
    int getBestFeature() {
        int    best    = -1;
        double bestDlb = 1.0;
        for (int j = 0; j < numFeatures; j++) {
            double dlb = deltaLossBound(j);
            if (dlb < bestDlb) {
                bestDlb = dlb;
                best    = j;
            }
        }
        return best;
    }

    /**
     * Updates lambda[j] by alpha, recomputes density, normalizer,
     * regularization, and all feature expectations.
     */
    void increaseLambda(int j, double alpha) {
        for (int i = 0; i < n; i++) {
            density[i] *= Math.exp(alpha * featVals[j][i]);
        }
        densityNormalizer = 0.0;
        for (int i = 0; i < n; i++) densityNormalizer += density[i];
        reg += (Math.abs(lambda[j] + alpha) - Math.abs(lambda[j])) * sampleDeviation[j];
        lambda[j] += alpha;
        updateExpectations();
    }

    /** Returns the Shannon entropy of the current distribution. */
    public double getEntropy() {
        double h = 0.0;
        for (int i = 0; i < n; i++) {
            double p = density[i] / densityNormalizer;
            if (p > 0.0) h -= p * Math.log(p);
        }
        return h;
    }

    /** Returns the normalized density weights (sum to 1). */
    public double[] getNormalizedDensity() {
        double[] w = new double[n];
        for (int i = 0; i < n; i++) w[i] = density[i] / densityNormalizer;
        return w;
    }

    public double[] getLambdas()           { return lambda.clone(); }
    public double[] getSampleExpectations(){ return sampleExpectation.clone(); }
    public double[] getSampleDeviations()  { return sampleDeviation.clone(); }

    /**
     * Runs the goodAlpha-based sequential coordinate ascent optimizer.
     * Mirrors {@code Sequential.run()} using goodAlpha as the step function.
     *
     * @param maxIter               Maximum iterations.
     * @param convergenceThreshold  Stop when loss improvement over 20
     *                              iterations is below this value.
     * @return Final loss.
     */
    public double train(int maxIter, double convergenceThreshold) {
        double newLoss      = getLoss();
        double previousLoss = Double.POSITIVE_INFINITY;

        for (iterations = 0; iterations < maxIter; iterations++) {
            int j = getBestFeature();
            if (j < 0) break;

            double alpha = goodAlpha(j);
            alpha = reduceAlpha(alpha, iterations);

            if (Math.abs(alpha) > 1e-15) {
                increaseLambda(j, alpha);
                newLoss = getLoss();
            }

            // Convergence test every CONVERGENCE_TEST_FREQ iterations
            if (iterations == 0) {
                previousLoss = newLoss;
            } else if (iterations % CONVERGENCE_TEST_FREQ == 0) {
                if (previousLoss - newLoss < convergenceThreshold) {
                    break;
                }
                previousLoss = newLoss;
            }
        }
        return newLoss;
    }

    // =========================================================================
    // Static API for rJava calls
    // =========================================================================

    /**
     * Evaluates a linear feature for all n points.
     * Formula: {@code (values[i] - min) / (max - min)}.
     * When {@code min == max}, returns 0 for all points.
     *
     * @param values Raw environmental variable values (length n).
     * @return Normalized feature values (length n).
     */
    public static double[] evalLinearFeature(double[] values) {
        double min = values[0], max = values[0];
        for (double v : values) {
            if (v < min) min = v;
            if (v > max) max = v;
        }
        double scale = (max == min) ? 1.0 : (max - min);
        double[] result = new double[values.length];
        for (int i = 0; i < values.length; i++) {
            result[i] = (values[i] - min) / scale;
        }
        return result;
    }

    /**
     * Computes the sample expectation (mean of normalized feature over
     * occurrence sample indices).
     *
     * @param values        Raw environmental variable values.
     * @param sampleIndices 0-based occurrence indices.
     * @return Scalar sample expectation.
     */
    public static double computeSampleExpectation(double[] values, int[] sampleIndices) {
        double min = values[0], max = values[0];
        for (double v : values) {
            if (v < min) min = v;
            if (v > max) max = v;
        }
        double scale = (max == min) ? 1.0 : (max - min);
        double sum = 0.0;
        for (int idx : sampleIndices) {
            sum += (values[idx] - min) / scale;
        }
        return sum / sampleIndices.length;
    }

    /**
     * Computes the sample deviation (population standard deviation of the
     * normalized feature over occurrence sample indices).
     *
     * @param values        Raw environmental variable values.
     * @param sampleIndices 0-based occurrence indices.
     * @return Scalar sample standard deviation.
     */
    public static double computeSampleDeviation(double[] values, int[] sampleIndices) {
        double mean  = computeSampleExpectation(values, sampleIndices);
        double min   = values[0], max = values[0];
        for (double v : values) {
            if (v < min) min = v;
            if (v > max) max = v;
        }
        double scale = (max == min) ? 1.0 : (max - min);
        double var   = 0.0;
        for (int idx : sampleIndices) {
            double diff = (values[idx] - min) / scale - mean;
            var += diff * diff;
        }
        var /= sampleIndices.length;
        return Math.sqrt(var);
    }

    /**
     * Computes the goodAlpha step size for given parameters.
     * Exposed as a static method for isolated unit testing of the formula.
     *
     * @param N1     Sample expectation (empirical mean over occurrences).
     * @param W1     Model expectation (density-weighted mean).
     * @param beta   Regularization parameter.
     * @param lambda Current lambda value.
     * @return The goodAlpha step size.
     */
    public static double computeGoodAlpha(double N1, double W1,
                                          double beta, double lambda) {
        double W0 = 1.0 - W1;
        double N0 = 1.0 - N1;
        if (W0 < EPS || W1 < EPS) return 0.0;
        double alpha;
        if (N1 - beta > EPS) {
            alpha = Math.log((N1 - beta) * W0 / ((N0 + beta) * W1));
            if (alpha + lambda > 0.0) return alpha;
        }
        if (N0 - beta > EPS) {
            alpha = Math.log((N1 + beta) * W0 / ((N0 - beta) * W1));
            if (alpha + lambda < 0.0) return alpha;
        }
        return -lambda;
    }

    /**
     * Applies the reduceAlpha damping schedule.
     *
     * @param alpha     Undamped step size.
     * @param iteration 0-based iteration index.
     * @return Damped step size.
     */
    public static double computeReduceAlpha(double alpha, int iteration) {
        if (iteration < 10) return alpha / 50.0;
        if (iteration < 20) return alpha / 10.0;
        if (iteration < 50) return alpha / 3.0;
        return alpha;
    }

    /**
     * Applies cloglog transformation: {@code 1 - exp(-raw[i] * exp(entropy))}.
     *
     * @param raw     Raw prediction values (normalized density weights).
     * @param entropy Shannon entropy of the trained distribution.
     * @return Cloglog-transformed values in [0, 1].
     */
    public static double[] cloglogTransform(double[] raw, double entropy) {
        double expEntropy = Math.exp(entropy);
        double[] result   = new double[raw.length];
        for (int i = 0; i < raw.length; i++) {
            result[i] = 1.0 - Math.exp(-raw[i] * expEntropy);
        }
        return result;
    }

    /**
     * Trains a 2-feature model and returns key results.
     *
     * @param bio1              Raw feature 0 values.
     * @param bio2              Raw feature 1 values.
     * @param sampleIndices     0-based occurrence indices.
     * @param betaMultiplier    Regularization multiplier.
     * @param minDeviation      Minimum sample deviation floor.
     * @param maxIter           Maximum iterations.
     * @param convergenceThreshold Convergence threshold.
     * @return Array: [loss, entropy, gain, lambda0, lambda1, (double) iterations].
     */
    public static double[] trainModel(double[] bio1, double[] bio2,
                                      int[] sampleIndices,
                                      double betaMultiplier, double minDeviation,
                                      int maxIter, double convergenceThreshold) {
        MaxentMini m    = new MaxentMini(bio1, bio2, sampleIndices,
                                         betaMultiplier, minDeviation);
        double loss     = m.train(maxIter, convergenceThreshold);
        double entropy  = m.getEntropy();
        double gain     = Math.log(m.n) - loss;
        return new double[] {
            loss, entropy, gain,
            m.lambda[0], m.lambda[1],
            m.iterations
        };
    }

    /**
     * Trains for exactly {@code nIterations} steps without any convergence
     * check. Useful for comparing per-iteration loss trajectories.
     *
     * @return Array: [loss, entropy, lambda0, lambda1].
     */
    public static double[] trainNIterations(double[] bio1, double[] bio2,
                                            int[] sampleIndices,
                                            double betaMultiplier, double minDeviation,
                                            int nIterations) {
        MaxentMini m = new MaxentMini(bio1, bio2, sampleIndices,
                                       betaMultiplier, minDeviation);
        for (int iter = 0; iter < nIterations; iter++) {
            int j = m.getBestFeature();
            if (j < 0) break;
            double alpha = m.goodAlpha(j);
            alpha = m.reduceAlpha(alpha, iter);
            if (Math.abs(alpha) > 1e-15) {
                m.increaseLambda(j, alpha);
            }
        }
        return new double[] {
            m.getLoss(), m.getEntropy(),
            m.lambda[0], m.lambda[1]
        };
    }

    /**
     * Returns normalized density weights for all background points after
     * training to convergence.
     *
     * @return Normalized density array (sums to 1), length n.
     */
    public static double[] getWeights(double[] bio1, double[] bio2,
                                      int[] sampleIndices,
                                      double betaMultiplier, double minDeviation,
                                      int maxIter, double convergenceThreshold) {
        MaxentMini m = new MaxentMini(bio1, bio2, sampleIndices,
                                       betaMultiplier, minDeviation);
        m.train(maxIter, convergenceThreshold);
        return m.getNormalizedDensity();
    }

    /**
     * Returns sample expectations for both features (before training).
     *
     * @return Array of length 2: [sampleExp_bio1, sampleExp_bio2].
     */
    public static double[] getSampleExpectations(double[] bio1, double[] bio2,
                                                  int[] sampleIndices,
                                                  double betaMultiplier, double minDeviation) {
        MaxentMini m = new MaxentMini(bio1, bio2, sampleIndices,
                                       betaMultiplier, minDeviation);
        return m.getSampleExpectations();
    }

    /**
     * Returns the model loss before any training (all lambdas = 0).
     * Expected value: {@code log(n)}.
     */
    public static double getInitialLoss(double[] bio1, double[] bio2,
                                         int[] sampleIndices,
                                         double betaMultiplier, double minDeviation) {
        MaxentMini m = new MaxentMini(bio1, bio2, sampleIndices,
                                       betaMultiplier, minDeviation);
        return m.getLoss();
    }
}
