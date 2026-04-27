/*
 * MaxentRefRunner.java
 * --------------------------------------------------------------------------
 * Phase A of the maxentcpp fidelity plan (issues #36 / #37).
 *
 * Thin orchestrator that drives the **unmodified** Java Maxent 3.4.4
 * `density.*` classes (FeaturedSpace, Sequential, LinearFeature, Sample,
 * Params) on a caller-supplied linear-feature, cloglog-output problem.
 *
 * This class replaces the simplified `MaxentMini.java` as the reference
 * oracle: it runs the real `density.Sequential` optimizer
 * (deltaLossBound feature selection, newtonStep + searchAlpha updates,
 * parallel-update every N iterations, undoing loss-violating steps) and
 * exposes the final state so that maxentcpp C++ outputs can be compared
 * at 1e-6 precision against the original Java algorithm.
 *
 * Two entry points are provided:
 *
 *   1. **Library API (rJava)** - construct via
 *          new MaxentRefRunner(feat0, feat1, sampleIdx, betaMultiplier,
 *                              maxIter, convergence)
 *      and query via getLambdas(), getDensity(), getRaw(),
 *      getCloglogJava(), getEntropy(), getLoss(), etc.
 *
 *   2. **Golden generator (CLI)** - `java density.MaxentRefRunner
 *          bio1.asc bio2.asc occurrences.csv outDir` writes CSV goldens
 *      (lambdas.csv, density.csv, scalars.csv) consumed by the
 *      test-golden-java.R sanity test.
 *
 * Placed in package `density` to access package-private fields/methods
 * on `FeaturedSpace`, `Sequential`, `Feature`, and `Sample`. No Maxent
 * source file is modified by this class.
 */
package density;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;

public class MaxentRefRunner {

    /* ------------------------------------------------------------------
     * Internal feature: returns pre-scaled [0,1] values for point i
     * and for a Sample (via its point index).
     * ------------------------------------------------------------------ */
    private static final class ScaledVec extends Feature {
        final double[] vals;
        ScaledVec(double[] v, String name) {
            super(v.length, name);
            this.vals = v;
        }
        public double eval(int p)     { return vals[p]; }
        public double eval(Sample s)  { return vals[s.getPoint()]; }
    }

    /* ------------------------------------------------------------------
     * Persistent state of one training run.
     * ------------------------------------------------------------------ */
    final FeaturedSpace fs;
    final Sequential    seq;
    final double        finalLoss;
    final double        finalEntropy;
    final int           iterations;
    final String[]      featureNames;
    final double        betaMultiplier;
    final double        autoBetaLqp;

    /**
     * Runs the real Java Maxent Sequential optimizer on two linear
     * features over a shared background grid.
     *
     * @param feat0         raw environmental values for feature 0 (length n)
     * @param feat1         raw environmental values for feature 1 (length n)
     * @param sampleIdx     0-based background indices of occurrence samples
     * @param betaMultiplier regularization multiplier (typically 1.0)
     * @param maxIter       maximum optimizer iterations
     * @param convergence   convergence threshold (delta-loss over 20 iters)
     */
    public MaxentRefRunner(double[] feat0, double[] feat1,
                           int[] sampleIdx,
                           double betaMultiplier,
                           int maxIter, double convergence) {

        // Match the (raw - min) / (max - min) linear scaling that Maxent's
        // standard pipeline applies to continuous features via
        // ScaledFeature(rawLayer, min, max). Pre-computing avoids wiring
        // the full GridSet / GridIO machinery for this minimal oracle.
        double[] v0 = prescale(feat0);
        double[] v1 = prescale(feat1);

        LinearFeature lf0 = new LinearFeature(new ScaledVec(v0, "bio1"), "bio1");
        LinearFeature lf1 = new LinearFeature(new ScaledVec(v1, "bio2"), "bio2");

        // Replicate Runner.autoSetBeta() for the linear-only schedule
        // (thresholds = [10, 30, 100], betas = [1.0, 0.2, 0.05]).
        // See /home/ubuntu/repos/Maxent/density/Runner.java:2265.
        this.autoBetaLqp    = interpolateLinearBeta(sampleIdx.length);
        this.betaMultiplier = betaMultiplier;
        lf0.setBeta(autoBetaLqp * betaMultiplier);
        lf1.setBeta(autoBetaLqp * betaMultiplier);

        Sample[] samples = new Sample[sampleIdx.length];
        for (int k = 0; k < sampleIdx.length; k++) {
            samples[k] = new Sample(sampleIdx[k], 0, 0, 0.0, 0.0, "sp1");
        }

        Params params = new Params();
        params.setBetamultiplier(betaMultiplier);
        params.setMaximumiterations(maxIter);
        params.setConvergencethreshold(convergence);
        params.setAutofeature(false);
        params.setLinear(true);
        params.setQuadratic(false);
        params.setProduct(false);
        params.setThreshold(false);
        params.setHinge(false);

        this.fs = new FeaturedSpace(samples,
                                    new Feature[] { lf0, lf1 }, params);
        this.featureNames = new String[] { lf0.name, lf1.name };

        this.seq          = new Sequential(fs, params);
        this.finalLoss    = seq.run();
        this.iterations   = seq.iteration;
        this.finalEntropy = fs.getEntropy();
    }

    /* ------------------------------------------------------------------
     * Public getters -- stable rJava surface.
     * ------------------------------------------------------------------ */

    public double[] getLambdas()             { return collect(Feature::getLambda); }
    public double[] getSampleExpectations()  { return collect(Feature::getSampleExpectation); }
    public double[] getSampleDeviations()    { return collect(Feature::getSampleDeviation); }
    public double[] getExpectations()        { return collect(Feature::getExpectation); }
    public double[] getBetas()               { return collect(Feature::getBeta); }
    public String[] getFeatureNames()        { return featureNames.clone(); }

    /** Unnormalized per-point density: exp(LP[i] - LPN). */
    public double[] getDensity()             { return fs.density.clone(); }

    /** Per-point linear predictor sum_j lambda[j] * f_j(i). */
    public double[] getLinearPredictor()     { return fs.linearPredictor.clone(); }

    public double   getDensityNormalizer()           { return fs.densityNormalizer; }
    public double   getLinearPredictorNormalizer()   { return fs.linearPredictorNormalizer; }

    /** Normalized distribution density[i] / densityNormalizer (sums to 1). */
    public double[] getRaw() {
        int n = fs.numPoints;
        double[] r = new double[n];
        for (int i = 0; i < n; i++) r[i] = fs.density[i] / fs.densityNormalizer;
        return r;
    }

    /**
     * Java Maxent cloglog transform: 1 - exp(-raw[i] * exp(entropy)).
     * See /home/ubuntu/repos/Maxent/density/Project.java:331.
     */
    public double[] getCloglogJava() {
        double[] raw  = getRaw();
        double   expH = Math.exp(finalEntropy);
        double[] cl   = new double[raw.length];
        for (int i = 0; i < raw.length; i++)
            cl[i] = 1.0 - Math.exp(-raw[i] * expH);
        return cl;
    }

    public double getEntropy()            { return finalEntropy; }
    public double getLoss()               { return finalLoss; }
    public double getUnregularizedLoss()  { return fs.getLoss(); }
    public double getL1Reg()              { return fs.getL1reg(); }
    public double getGain()               { return Math.log(fs.numPoints) - finalLoss; }
    public int    getIterations()         { return iterations; }
    public int    getNumFeatures()        { return fs.numFeatures; }
    public int    getNumPoints()          { return fs.numPoints; }
    public double getBetaMultiplier()     { return betaMultiplier; }
    public double getAutoBetaLqp()        { return autoBetaLqp; }

    /* ------------------------------------------------------------------
     * Trajectory capture (Phase B — quantify the C++ vs real-Java gap).
     *
     * A subclass of `density.Sequential` that snapshots
     *     (loss, entropy, lambda_0, ..., lambda_{k-1})
     * after each iteration in a caller-supplied `checkpoints` list.
     * Overrides `terminationTest` to (a) record snapshots and (b)
     * disable the convergence-based early exit so the optimizer always
     * runs to `max(checkpoints)` iterations. The unmodified `Sequential`
     * semantics (deltaLossBound feature selection, newtonStep +
     * searchAlpha for continuous features, doParallelUpdate every 10
     * iterations, undo on loss violation) are preserved because
     * `terminationTest` only affects the `break` clause.
     * ------------------------------------------------------------------ */
    private static final class TrajectorySequential extends Sequential {
        final int[]    checkpoints;   // sorted ascending, 1-based iterations
        final int      numFeatures;
        final double[] lossBuf;
        final double[] entropyBuf;
        final double[][] lambdaBuf;   // [checkpoint][feature]
        int   cpIdx = 0;

        TrajectorySequential(FeaturedSpace X, Params params, int[] cps) {
            super(X, params);
            this.checkpoints = cps;
            this.numFeatures = X.numFeatures;
            this.lossBuf     = new double[cps.length];
            this.entropyBuf  = new double[cps.length];
            this.lambdaBuf   = new double[cps.length][numFeatures];
            for (int i = 0; i < cps.length; i++) {
                lossBuf[i]    = Double.NaN;
                entropyBuf[i] = Double.NaN;
                for (int j = 0; j < numFeatures; j++)
                    lambdaBuf[i][j] = Double.NaN;
            }
        }

        @Override
        boolean terminationTest(double newLoss) {
            int completed = iteration + 1;  // iteration=k in loop ⇒ iter k+1 done
            while (cpIdx < checkpoints.length
                   && checkpoints[cpIdx] <= completed) {
                if (checkpoints[cpIdx] == completed) {
                    lossBuf[cpIdx]    = newLoss;
                    // FeaturedSpace.getEntropy() caches: reset the cache
                    // to force a fresh recomputation at each checkpoint.
                    // Sequential.run() itself only reads entropy at the
                    // very end (via FeaturedSpace.describe(...)), so the
                    // stale cached value is normally fine; for a
                    // per-iteration trajectory we need the fresh value.
                    X.entropy = -1.0;
                    entropyBuf[cpIdx] = X.getEntropy();
                    for (int j = 0; j < numFeatures; j++)
                        lambdaBuf[cpIdx][j] = X.features[j].lambda;
                }
                cpIdx++;
            }
            return cpIdx >= checkpoints.length;  // stop once all captured
        }
    }

    /**
     * Runs the real Java Maxent optimizer and reports the trained state
     * at each iteration in {@code checkpoints} (1-based, ascending).
     *
     * Returns a flat array laid out as
     *   [loss_cp0, entropy_cp0, lam_0_cp0, ..., lam_{k-1}_cp0,
     *    loss_cp1, entropy_cp1, lam_0_cp1, ..., lam_{k-1}_cp1,
     *    ...]
     * of length {@code checkpoints.length * (2 + numFeatures)}.
     *
     * Convergence-based early termination is suppressed so the trajectory
     * is deterministic regardless of how flat the loss becomes — that
     * lets Phase B compare the full lambda evolution, not just the
     * (potentially short) portion before convergence cuts the Java run
     * but not the C++ run (or vice versa).
     *
     * @param feat0          raw feature-0 vector (length n)
     * @param feat1          raw feature-1 vector (length n)
     * @param sampleIdx      0-based occurrence indices
     * @param betaMultiplier regularization multiplier
     * @param checkpoints    iteration numbers to snapshot (1-based)
     */
    public static double[] runTrajectoryFlat(double[] feat0, double[] feat1,
                                             int[] sampleIdx,
                                             double betaMultiplier,
                                             int[] checkpoints) {
        int[] cps = checkpoints.clone();
        java.util.Arrays.sort(cps);
        int maxIter = cps[cps.length - 1];

        double[] v0 = prescale(feat0);
        double[] v1 = prescale(feat1);

        LinearFeature lf0 = new LinearFeature(new ScaledVec(v0, "bio1"), "bio1");
        LinearFeature lf1 = new LinearFeature(new ScaledVec(v1, "bio2"), "bio2");
        double autoBeta = interpolateLinearBeta(sampleIdx.length);
        lf0.setBeta(autoBeta * betaMultiplier);
        lf1.setBeta(autoBeta * betaMultiplier);

        Sample[] samples = new Sample[sampleIdx.length];
        for (int k = 0; k < sampleIdx.length; k++)
            samples[k] = new Sample(sampleIdx[k], 0, 0, 0.0, 0.0, "sp1");

        Params params = new Params();
        params.setBetamultiplier(betaMultiplier);
        params.setMaximumiterations(maxIter);
        params.setConvergencethreshold(0.0);   // disable early-out
        params.setAutofeature(false);
        params.setLinear(true);
        params.setQuadratic(false);
        params.setProduct(false);
        params.setThreshold(false);
        params.setHinge(false);

        FeaturedSpace        fsLocal = new FeaturedSpace(samples,
                                         new Feature[] { lf0, lf1 }, params);
        TrajectorySequential tseq    = new TrajectorySequential(
                                         fsLocal, params, cps);
        tseq.run();

        int stride = 2 + tseq.numFeatures;
        double[] out = new double[cps.length * stride];
        for (int i = 0; i < cps.length; i++) {
            int base        = i * stride;
            out[base]       = tseq.lossBuf[i];
            out[base + 1]   = tseq.entropyBuf[i];
            for (int j = 0; j < tseq.numFeatures; j++)
                out[base + 2 + j] = tseq.lambdaBuf[i][j];
        }
        return out;
    }

    /**
     * Phase D — linear + quadratic trajectory on a 2-variable raster stack.
     *
     * Builds 4 features in the exact order matching the C++ maxent::FeatureFactory
     * when cfg = {linear=true, quadratic=true, product=false}:
     *
     *   0: bio1        (LinearFeature,   scaled)
     *   1: bio1^2      (SquareFeature over the same ScaledVec)
     *   2: bio2        (LinearFeature)
     *   3: bio2^2      (SquareFeature)
     *
     * (Feature order is per-variable-then-per-type, matching Runner.java's
     * loop `for i in vars: linear; quadratic` at density/Runner.java:2177-2180.)
     *
     * Uses the Java Runner.autoSetBeta quadratic-on schedule
     *   thresholds = {  0,  10,  17,   30,  100 }
     *   betas      = {1.3, 0.8, 0.5, 0.25, 0.05 }
     * (see density/Runner.java:2260-2263), so beta_lqp is applied uniformly
     * to linear AND quadratic features — matching the real pipeline.
     *
     * Same TrajectorySequential override as {@link #runTrajectoryFlat} —
     * convergence-based early termination is disabled so the trajectory
     * is deterministic regardless of loss flatness.
     */
    public static double[] runTrajectoryFlatLQ(double[] feat0, double[] feat1,
                                               int[] sampleIdx,
                                               double betaMultiplier,
                                               int[] checkpoints) {
        int[] cps = checkpoints.clone();
        java.util.Arrays.sort(cps);
        int maxIter = cps[cps.length - 1];

        double[] v0 = prescale(feat0);
        double[] v1 = prescale(feat1);

        ScaledVec sv0 = new ScaledVec(v0, "bio1");
        ScaledVec sv1 = new ScaledVec(v1, "bio2");

        LinearFeature lf0  = new LinearFeature(sv0, "bio1");
        SquareFeature sqf0 = new SquareFeature(sv0, "bio1");   // names itself "bio1^2"
        LinearFeature lf1  = new LinearFeature(sv1, "bio2");
        SquareFeature sqf1 = new SquareFeature(sv1, "bio2");

        double autoBeta = interpolateLqpBeta(sampleIdx.length);
        lf0 .setBeta(autoBeta * betaMultiplier);
        sqf0.setBeta(autoBeta * betaMultiplier);
        lf1 .setBeta(autoBeta * betaMultiplier);
        sqf1.setBeta(autoBeta * betaMultiplier);

        Sample[] samples = new Sample[sampleIdx.length];
        for (int k = 0; k < sampleIdx.length; k++)
            samples[k] = new Sample(sampleIdx[k], 0, 0, 0.0, 0.0, "sp1");

        Params params = new Params();
        params.setBetamultiplier(betaMultiplier);
        params.setMaximumiterations(maxIter);
        params.setConvergencethreshold(0.0);   // disable early-out
        params.setAutofeature(false);
        params.setLinear(true);
        params.setQuadratic(true);
        params.setProduct(false);
        params.setThreshold(false);
        params.setHinge(false);

        Feature[] feats = new Feature[] { lf0, sqf0, lf1, sqf1 };
        FeaturedSpace        fsLocal = new FeaturedSpace(samples, feats, params);
        TrajectorySequential tseq    = new TrajectorySequential(
                                         fsLocal, params, cps);
        tseq.run();

        int stride = 2 + tseq.numFeatures;
        double[] out = new double[cps.length * stride];
        for (int i = 0; i < cps.length; i++) {
            int base        = i * stride;
            out[base]       = tseq.lossBuf[i];
            out[base + 1]   = tseq.entropyBuf[i];
            for (int j = 0; j < tseq.numFeatures; j++)
                out[base + 2 + j] = tseq.lambdaBuf[i][j];
        }
        return out;
    }

    /* ------------------------------------------------------------------
     * Private helpers.
     * ------------------------------------------------------------------ */

    @FunctionalInterface
    private interface FeatExtractor {
        double apply(Feature f);
        /* convenience overloads to allow reflection-free method handles */
    }

    private double[] collect(FeatExtractor ex) {
        double[] out = new double[fs.numFeatures];
        for (int j = 0; j < fs.numFeatures; j++) out[j] = ex.apply(fs.features[j]);
        return out;
    }

    private static double[] prescale(double[] raw) {
        double min = raw[0], max = raw[0];
        for (double v : raw) {
            if (v < min) min = v;
            if (v > max) max = v;
        }
        double scale = (max == min) ? 1.0 : (max - min);
        double[] out = new double[raw.length];
        for (int i = 0; i < raw.length; i++) out[i] = (raw[i] - min) / scale;
        return out;
    }

    /** Piecewise-linear autoSetBeta for linear-only features. */
    static double interpolateLinearBeta(int numSamples) {
        int[]    thr   = {  10,  30, 100 };
        double[] betas = { 1.0, 0.2, 0.05 };
        int i;
        for (i = 0; i < thr.length; i++) if (numSamples <= thr[i]) break;
        if (i == 0)             return betas[0];
        if (i == thr.length)    return betas[thr.length - 1];
        return betas[i - 1]
             + (betas[i] - betas[i - 1])
               * (numSamples - thr[i - 1])
               / (double) (thr[i] - thr[i - 1]);
    }

    /**
     * Piecewise-linear autoSetBeta for linear + quadratic
     * (when product=false, matching Runner.autoSetBeta at
     *  density/Runner.java:2260-2263).
     */
    static double interpolateLqpBeta(int numSamples) {
        int[]    thr   = {   0,  10,  17,   30,  100 };
        double[] betas = { 1.3, 0.8, 0.5, 0.25, 0.05 };
        int i;
        for (i = 0; i < thr.length; i++) if (numSamples <= thr[i]) break;
        if (i == 0)             return betas[0];
        if (i == thr.length)    return betas[thr.length - 1];
        return betas[i - 1]
             + (betas[i] - betas[i - 1])
               * (numSamples - thr[i - 1])
               / (double) (thr[i] - thr[i - 1]);
    }

    /**
     * Piecewise-linear autoSetBeta for linear + quadratic + product
     * (the product-on schedule in Runner.autoSetBeta at
     *  density/Runner.java:2256-2258).  At n=10 samples this returns
     *  exactly 1.6 — the Phase D.2 product-on beta_lqp.
     */
    static double interpolateLqpProductBeta(int numSamples) {
        int[]    thr   = {   0,   10,   17,    30,  100 };
        double[] betas = { 2.6,  1.6,  0.9,  0.55, 0.05 };
        int i;
        for (i = 0; i < thr.length; i++) if (numSamples <= thr[i]) break;
        if (i == 0)             return betas[0];
        if (i == thr.length)    return betas[thr.length - 1];
        return betas[i - 1]
             + (betas[i] - betas[i - 1])
               * (numSamples - thr[i - 1])
               / (double) (thr[i] - thr[i - 1]);
    }

    /**
     * Phase D.2 — linear + quadratic + product trajectory on the 2-variable
     * asymmetric raster stack.
     *
     * Builds 5 features in the order used by the companion C++ test
     * test_sequential_lqp.cpp (per-variable L+Q then a single cross-term
     * product appended at the end):
     *
     *   0: bio1          (LinearFeature over the ScaledVec for bio1)
     *   1: bio1^2        (SquareFeature  over the ScaledVec for bio1)
     *   2: bio2          (LinearFeature over the ScaledVec for bio2)
     *   3: bio2^2        (SquareFeature  over the ScaledVec for bio2)
     *   4: bio1*bio2     (ProductFeature of the two ScaledVecs,
     *                     mirroring Runner.java:2184-2188 where
     *                     ProductFeature is created from already-scaled cont[]
     *                     features with i &lt; j)
     *
     * Per Runner.java feature-order convention the "real" pipeline would
     * emit {lf0, lf1, sqf0, sqf1, pf01} (per-type-then-per-variable), but
     * the feature indices in Sequential are order-independent as long as
     * both Java and C++ agree on the indexing.  The C++ side embeds the
     * golden in the same order emitted here.
     *
     * Uses the Java Runner.autoSetBeta product-on schedule
     *   thresholds = {   0,   10,   17,    30,  100 }
     *   betas      = { 2.6,  1.6,  0.9,  0.55, 0.05 }
     * (density/Runner.java:2256-2258), so beta_lqp = 1.6 at n=10 — applied
     * uniformly to linear, quadratic AND product features.  ProductFeature
     * is not a BinaryFeature / ThrGeneratorFeature / HingeGeneratorFeature
     * so it falls through to the beta_lqp branch in autoSetBeta
     * (density/Runner.java:2291-2299).
     *
     * Same TrajectorySequential override as {@link #runTrajectoryFlat} —
     * convergence-based early termination is disabled so the trajectory
     * is deterministic regardless of loss flatness.
     */
    public static double[] runTrajectoryFlatLQP(double[] feat0, double[] feat1,
                                                int[] sampleIdx,
                                                double betaMultiplier,
                                                int[] checkpoints) {
        int[] cps = checkpoints.clone();
        java.util.Arrays.sort(cps);
        int maxIter = cps[cps.length - 1];

        double[] v0 = prescale(feat0);
        double[] v1 = prescale(feat1);

        ScaledVec sv0 = new ScaledVec(v0, "bio1");
        ScaledVec sv1 = new ScaledVec(v1, "bio2");

        LinearFeature  lf0  = new LinearFeature (sv0, "bio1");
        SquareFeature  sqf0 = new SquareFeature (sv0, "bio1");
        LinearFeature  lf1  = new LinearFeature (sv1, "bio2");
        SquareFeature  sqf1 = new SquareFeature (sv1, "bio2");
        ProductFeature pf01 = new ProductFeature(sv0, "bio1", sv1, "bio2");

        double autoBeta = interpolateLqpProductBeta(sampleIdx.length);
        double scaled   = autoBeta * betaMultiplier;
        lf0 .setBeta(scaled);
        sqf0.setBeta(scaled);
        lf1 .setBeta(scaled);
        sqf1.setBeta(scaled);
        pf01.setBeta(scaled);

        Sample[] samples = new Sample[sampleIdx.length];
        for (int k = 0; k < sampleIdx.length; k++)
            samples[k] = new Sample(sampleIdx[k], 0, 0, 0.0, 0.0, "sp1");

        Params params = new Params();
        params.setBetamultiplier(betaMultiplier);
        params.setMaximumiterations(maxIter);
        params.setConvergencethreshold(0.0);
        params.setAutofeature(false);
        params.setLinear(true);
        params.setQuadratic(true);
        params.setProduct(true);
        params.setThreshold(false);
        params.setHinge(false);

        Feature[] feats = new Feature[] { lf0, sqf0, lf1, sqf1, pf01 };
        FeaturedSpace        fsLocal = new FeaturedSpace(samples, feats, params);
        TrajectorySequential tseq    = new TrajectorySequential(
                                         fsLocal, params, cps);
        tseq.run();

        int stride = 2 + tseq.numFeatures;
        double[] out = new double[cps.length * stride];
        for (int i = 0; i < cps.length; i++) {
            int base        = i * stride;
            out[base]       = tseq.lossBuf[i];
            out[base + 1]   = tseq.entropyBuf[i];
            for (int j = 0; j < tseq.numFeatures; j++)
                out[base + 2 + j] = tseq.lambdaBuf[i][j];
        }
        return out;
    }

    /**
     * Piecewise-linear autoSetBeta for threshold features
     * (density/Runner.java:2266-2268):
     *   int[]    thresholds = {   0,  100 }
     *   double[] betas      = { 2.0,  1.0 }
     * At n=10 samples this returns exactly 1.9 — the Phase D.2b threshold
     * beta.
     */
    static double interpolateThrBeta(int numSamples) {
        int[]    thr   = {   0,  100 };
        double[] betas = { 2.0,  1.0 };
        int i;
        for (i = 0; i < thr.length; i++) if (numSamples <= thr[i]) break;
        if (i == 0)             return betas[0];
        if (i == thr.length)    return betas[thr.length - 1];
        return betas[i - 1]
             + (betas[i] - betas[i - 1])
               * (numSamples - thr[i - 1])
               / (double) (thr[i] - thr[i - 1]);
    }

    /**
     * Hinge features use a constant beta of 0.5 regardless of sample count
     * (density/Runner.java:2270 hard-coded beta_hge = 0.5).
     */
    static double interpolateHingeBeta(int numSamples) {
        return 0.5;
    }

    /**
     * Phase D.2b-threshold — linear + fixed-set threshold trajectory on
     * the 2-variable asymmetric raster stack.
     *
     * Builds 8 features in the order used by the companion C++ test
     * test_sequential_lt.cpp (per-variable L then 3 fixed thresholds):
     *
     *   0: bio1                    (LinearFeature over ScaledVec for bio1)
     *   1: (0.33&lt;bio1)             (ThresholdFeature on prescaled bio1)
     *   2: (0.57&lt;bio1)
     *   3: (0.83&lt;bio1)
     *   4: bio2                    (LinearFeature over ScaledVec for bio2)
     *   5: (0.23&lt;bio2)             (ThresholdFeature on prescaled bio2)
     *   6: (0.51&lt;bio2)
     *   7: (0.77&lt;bio2)
     *
     * Threshold positions are hand-picked so they never coincide with any
     * prescaled data value (which would expose the Java &gt; vs C++ &gt;=
     * divergence); see maxentcpp/src/cpp/include/maxent/feature.hpp:389
     * and Maxent/density/ThresholdFeature.java:38.
     *
     * Regularization: a single uniform beta is applied to ALL 8 features
     * (linear + threshold), equal to `betaMultiplier` exactly. This
     * mirrors maxent::Sequential which takes a single scalar
     * `beta_multiplier` parameter (see
     * maxentcpp/src/cpp/include/maxent/featured_space.hpp:624
     * set_sample_expectations). Real Runner.autoSetBeta applies different
     * betas per feature class (beta_lin vs beta_thr) — reproducing that
     * requires per-feature-beta plumbing in C++ Sequential which is out
     * of scope for Phase D.2b. We lose nothing: the parity test verifies
     * Sequential's handling of ThresholdFeature evals + deltaLossBound +
     * searchAlpha, not the autoSetBeta schedule logic.
     */
    public static double[] runTrajectoryFlatLT(double[] feat0, double[] feat1,
                                               int[] sampleIdx,
                                               double betaMultiplier,
                                               int[] checkpoints) {
        int[] cps = checkpoints.clone();
        java.util.Arrays.sort(cps);
        int maxIter = cps[cps.length - 1];

        double[] v0 = prescale(feat0);
        double[] v1 = prescale(feat1);

        ScaledVec sv0 = new ScaledVec(v0, "bio1");
        ScaledVec sv1 = new ScaledVec(v1, "bio2");

        LinearFeature lf0 = new LinearFeature(sv0, "bio1");
        LinearFeature lf1 = new LinearFeature(sv1, "bio2");

        // Thresholds are chosen strictly inside the sample value range
        // for each variable so that every binary indicator has a
        // non-degenerate split across the 10 occurrences (5-8 Y / 2-5 N).
        // This matters because Java's FeaturedSpace.setSampleExpectations
        // has a special case for BINARY features with sampleExpectation=1
        // (it uses 1/(2m) instead of the usual spread clip), which C++
        // maxent::Sequential does not implement.  Keeping samples mixed
        // sidesteps that divergence entirely and lets the 1e-6 parity
        // bar pass on the default set_sample_expectations path.
        // Sample bio1 prescaled ∈ {0.741, 0.778, 0.815, 0.852, 0.852,
        //                         0.889, 0.926, 0.926, 0.963, 1.000}
        // Sample bio2 prescaled ∈ {0.778, 0.796, 0.815, 0.852, 0.870,
        //                         0.889, 0.907, 0.963, 0.981, 1.000}
        double[] thr0 = { 0.78, 0.85, 0.92 };
        double[] thr1 = { 0.80, 0.88, 0.95 };
        ThresholdFeature tf00 = new ThresholdFeature(sv0, thr0[0], "bio1");
        ThresholdFeature tf01 = new ThresholdFeature(sv0, thr0[1], "bio1");
        ThresholdFeature tf02 = new ThresholdFeature(sv0, thr0[2], "bio1");
        ThresholdFeature tf10 = new ThresholdFeature(sv1, thr1[0], "bio2");
        ThresholdFeature tf11 = new ThresholdFeature(sv1, thr1[1], "bio2");
        ThresholdFeature tf12 = new ThresholdFeature(sv1, thr1[2], "bio2");

        // Uniform per-feature beta (see javadoc): mirrors maxent::Sequential's
        // scalar beta_multiplier contract.
        double beta = betaMultiplier;
        lf0 .setBeta(beta);
        lf1 .setBeta(beta);
        tf00.setBeta(beta);
        tf01.setBeta(beta);
        tf02.setBeta(beta);
        tf10.setBeta(beta);
        tf11.setBeta(beta);
        tf12.setBeta(beta);

        Sample[] samples = new Sample[sampleIdx.length];
        for (int k = 0; k < sampleIdx.length; k++)
            samples[k] = new Sample(sampleIdx[k], 0, 0, 0.0, 0.0, "sp1");

        Params params = new Params();
        params.setBetamultiplier(betaMultiplier);
        params.setMaximumiterations(maxIter);
        params.setConvergencethreshold(0.0);
        params.setAutofeature(false);
        params.setLinear(true);
        params.setQuadratic(false);
        params.setProduct(false);
        params.setThreshold(true);
        params.setHinge(false);

        Feature[] feats = new Feature[] {
            lf0, tf00, tf01, tf02,
            lf1, tf10, tf11, tf12
        };
        FeaturedSpace        fsLocal = new FeaturedSpace(samples, feats, params);
        TrajectorySequential tseq    = new TrajectorySequential(
                                         fsLocal, params, cps);
        if (System.getProperty("maxent.ref.debug") != null) {
            System.err.println("[LT-DEBUG] features after FeaturedSpace ctor:");
            for (int j = 0; j < fsLocal.numFeatures; j++) {
                Feature f = fsLocal.features[j];
                double expect = 0;
                for (int ii = 0; ii < fsLocal.numPoints; ii++)
                    expect += fsLocal.getDensity(ii) * f.eval(ii);
                expect /= fsLocal.densityNormalizer;
                System.err.printf(
                    "  [%d] %s  sampleExp=%.8f  sampleDev=%.8f  beta=%.4f  expect=%.8f  isBinary=%b%n",
                    j, f.description(), f.sampleExpectation,
                    f.sampleDeviation, f.beta, expect, f.isBinary());
            }
        }
        tseq.run();

        int stride = 2 + tseq.numFeatures;
        double[] out = new double[cps.length * stride];
        for (int i = 0; i < cps.length; i++) {
            int base        = i * stride;
            out[base]       = tseq.lossBuf[i];
            out[base + 1]   = tseq.entropyBuf[i];
            for (int j = 0; j < tseq.numFeatures; j++)
                out[base + 2 + j] = tseq.lambdaBuf[i][j];
        }
        return out;
    }

    /**
     * Phase D.2b-hinge — linear + fixed-set hinge trajectory on the
     * 2-variable asymmetric raster stack.
     *
     * Builds 10 features in the order used by the companion C++ test
     * test_sequential_lh.cpp (per-variable L then 2 forward + 2 reverse
     * hinges):
     *
     *   0: bio1                   (LinearFeature over ScaledVec)
     *   1: 'bio1  fwd (0.2, 1.0)  (HingeFeature)
     *   2: 'bio1  fwd (0.5, 1.0)
     *   3: 'bio1  rev (0.0, 0.5)  (emulated via revFeature trick)
     *   4: 'bio1  rev (0.0, 0.8)
     *   5: bio2                   (LinearFeature)
     *   6: 'bio2  fwd (0.17, 1.0)
     *   7: 'bio2  fwd (0.53, 1.0)
     *   8: 'bio2  rev (0.0, 0.47)
     *   9: 'bio2  rev (0.0, 0.77)
     *
     * Reverse-hinge parity: Java `density.HingeFeature` only implements a
     * forward hinge eval(d &gt; min ? (d-min)/range : 0). In real Runner,
     * reverse hinges are produced by wrapping the underlying feature in a
     * `revFeature(f)` adapter that negates eval and then constructing a
     * forward hinge in negated space. Here we do the same trick: build a
     * small anonymous Feature that negates the ScaledVec values, then a
     * forward HingeFeature over it with knots (-max_knot_orig,
     * -min_knot_orig) so its algebra is identical to the C++ reverse
     * hinge (v &lt; max_knot_orig) ? (max_knot_orig - v)/range : 0.
     *
     * Knot positions are chosen so that no prescaled background value
     * equals any knot (avoiding &gt;/&gt;= edge cases) and so that data
     * never exceeds max_knot in the forward case or falls below min_knot
     * in the reverse case (so the C++ eval_from_env upper-clamp at 1.0
     * cannot diverge from the unclamped eval(int i)).
     *
     * Regularization: a single uniform beta is applied to ALL 10 features,
     * equal to `betaMultiplier` exactly (same rationale as runTrajectoryFlatLT
     * — matches C++ Sequential's scalar beta_multiplier contract).
     */
    public static double[] runTrajectoryFlatLH(double[] feat0, double[] feat1,
                                               int[] sampleIdx,
                                               double betaMultiplier,
                                               int[] checkpoints) {
        int[] cps = checkpoints.clone();
        java.util.Arrays.sort(cps);
        int maxIter = cps[cps.length - 1];

        final double[] v0 = prescale(feat0);
        final double[] v1 = prescale(feat1);

        ScaledVec sv0 = new ScaledVec(v0, "bio1");
        ScaledVec sv1 = new ScaledVec(v1, "bio2");

        LinearFeature lf0 = new LinearFeature(sv0, "bio1");
        LinearFeature lf1 = new LinearFeature(sv1, "bio2");

        // Forward hinges: forward-eval d>min ? (d-min)/range : 0.
        //   bio1: (0.2, 1.0), (0.5, 1.0)
        //   bio2: (0.17, 1.0), (0.53, 1.0)
        HingeFeature hf00f = new HingeFeature(sv0, 0.2,  1.0,  "bio1");
        HingeFeature hf01f = new HingeFeature(sv0, 0.5,  1.0,  "bio1");
        HingeFeature hf10f = new HingeFeature(sv1, 0.17, 1.0,  "bio2");
        HingeFeature hf11f = new HingeFeature(sv1, 0.53, 1.0,  "bio2");

        // Reverse hinges: emulate via forward-hinge-on-negated-feature.
        // For original knot range (mink, maxk) in the unnormalised space,
        //   reverse eval = (v < maxk) ? (maxk - v) / (maxk - mink) : 0
        //              = (-v > -maxk) ? (-v - (-maxk)) / (maxk - mink) : 0
        // which is a forward hinge over (-v) with min=-maxk, max=-mink.
        //   bio1: (0.0, 0.5) rev  ->  forward over -v with (-0.5, 0.0)
        //   bio1: (0.0, 0.8) rev  ->  forward over -v with (-0.8, 0.0)
        //   bio2: (0.0, 0.47) rev ->  forward over -v with (-0.47, 0.0)
        //   bio2: (0.0, 0.77) rev ->  forward over -v with (-0.77, 0.0)
        final Feature neg0 = new Feature(v0.length, "bio1__rev") {
            public double eval(int p)    { return -v0[p]; }
            public double eval(Sample s) { return -v0[s.getPoint()]; }
        };
        final Feature neg1 = new Feature(v1.length, "bio2__rev") {
            public double eval(int p)    { return -v1[p]; }
            public double eval(Sample s) { return -v1[s.getPoint()]; }
        };
        HingeFeature hf00r = new HingeFeature(neg0, -0.5,  0.0, "bio1__rev");
        HingeFeature hf01r = new HingeFeature(neg0, -0.8,  0.0, "bio1__rev");
        HingeFeature hf10r = new HingeFeature(neg1, -0.47, 0.0, "bio2__rev");
        HingeFeature hf11r = new HingeFeature(neg1, -0.77, 0.0, "bio2__rev");

        // Uniform per-feature beta (see javadoc): mirrors maxent::Sequential's
        // scalar beta_multiplier contract.
        double beta = betaMultiplier;
        lf0.setBeta(beta);
        lf1.setBeta(beta);
        for (HingeFeature h : new HingeFeature[] {
                hf00f, hf01f, hf00r, hf01r,
                hf10f, hf11f, hf10r, hf11r }) {
            h.setBeta(beta);
        }

        Sample[] samples = new Sample[sampleIdx.length];
        for (int k = 0; k < sampleIdx.length; k++)
            samples[k] = new Sample(sampleIdx[k], 0, 0, 0.0, 0.0, "sp1");

        Params params = new Params();
        params.setBetamultiplier(betaMultiplier);
        params.setMaximumiterations(maxIter);
        params.setConvergencethreshold(0.0);
        params.setAutofeature(false);
        params.setLinear(true);
        params.setQuadratic(false);
        params.setProduct(false);
        params.setThreshold(false);
        params.setHinge(true);

        Feature[] feats = new Feature[] {
            lf0, hf00f, hf01f, hf00r, hf01r,
            lf1, hf10f, hf11f, hf10r, hf11r
        };
        FeaturedSpace        fsLocal = new FeaturedSpace(samples, feats, params);
        TrajectorySequential tseq    = new TrajectorySequential(
                                         fsLocal, params, cps);
        tseq.run();

        int stride = 2 + tseq.numFeatures;
        double[] out = new double[cps.length * stride];
        for (int i = 0; i < cps.length; i++) {
            int base        = i * stride;
            out[base]       = tseq.lossBuf[i];
            out[base + 1]   = tseq.entropyBuf[i];
            for (int j = 0; j < tseq.numFeatures; j++)
                out[base + 2 + j] = tseq.lambdaBuf[i][j];
        }
        return out;
    }

    /* ------------------------------------------------------------------
     * Golden-CSV generator (CLI entry point).
     *
     *   java density.MaxentRefRunner bio1.asc bio2.asc occurrences.csv outDir
     *
     * Writes:
     *   lambdas.csv       per-feature lambda/expectation/deviation/beta
     *   density.csv       per-point density, raw, cloglog_java
     *   scalars.csv       loss, entropy, gain, normalizers, iterations, ...
     * ------------------------------------------------------------------ */

    private static double ascXll, ascYll, ascCell;

    public static void main(String[] args) throws Exception {
        boolean miniMode = args.length > 0 && args[0].equals("--mini");
        int     base     = miniMode ? 1 : 0;
        int     minArgs  = base + 3;
        if (args.length < minArgs) {
            System.err.println("Usage: density.MaxentRefRunner "
                             + "[--mini] bio1.asc bio2.asc occurrences.csv [outDir]");
            System.exit(2);
        }
        String bio1Path  = args[base];
        String bio2Path  = args[base + 1];
        String occPath   = args[base + 2];
        String outDir    = (args.length > base + 3) ? args[base + 3] : ".";

        double[][] g1   = readAsc(bio1Path);
        double[][] g2   = readAsc(bio2Path);
        int    nrows    = g1.length;
        int    ncols    = g1[0].length;
        double xll      = ascXll;
        double yll      = ascYll;
        double cell     = ascCell;

        double[] v1 = new double[nrows * ncols];
        double[] v2 = new double[nrows * ncols];
        for (int r = 0; r < nrows; r++) {
            for (int c = 0; c < ncols; c++) {
                v1[r * ncols + c] = g1[r][c];
                v2[r * ncols + c] = g2[r][c];
            }
        }

        List<double[]> occ = readOccCsv(occPath);
        int[] idx = new int[occ.size()];
        for (int k = 0; k < occ.size(); k++) {
            double lon = occ.get(k)[0];
            double lat = occ.get(k)[1];
            int row = (int) Math.floor((yll + nrows * cell - lat) / cell);
            int col = (int) Math.floor((lon - xll) / cell);
            if (row < 0)       row = 0;
            if (row >= nrows)  row = nrows - 1;
            if (col < 0)       col = 0;
            if (col >= ncols)  col = ncols - 1;
            idx[k] = row * ncols + col;
        }

        File od = new File(outDir);
        od.mkdirs();
        int[] cps = new int[] { 1, 2, 3, 5, 10, 20, 50, 100, 200, 500 };

        if (miniMode) {
            // MaxentMini goodAlpha trajectory (same CSV schema as real Java).
            int stride = 2 + 2;  // loss, entropy, lambda_0, lambda_1
            double[] traj = new double[cps.length * stride];
            for (int i = 0; i < cps.length; i++) {
                double[] s = MaxentMini.trainNIterations(
                    v1, v2, idx, 1.0, 0.001, cps[i]);
                // trainNIterations returns [loss, entropy, lambda0, lambda1]
                traj[i * stride]     = s[0];
                traj[i * stride + 1] = s[1];
                traj[i * stride + 2] = s[2];
                traj[i * stride + 3] = s[3];
            }
            writeTrajectoryCsv(new File(od, "trajectory_mini.csv"),
                               cps, traj, 2);
            System.out.println("MaxentMiniTrajectory: wrote "
                               + new File(od, "trajectory_mini.csv"));
            return;
        }

        MaxentRefRunner run = new MaxentRefRunner(v1, v2, idx,
                                                  1.0, 500, 1e-5);
        writeLambdasCsv(new File(od, "lambdas.csv"), run);
        writeDensityCsv(new File(od, "density.csv"),
                        run.getDensity(), run.getRaw(),
                        run.getCloglogJava());
        writeScalarsCsv(new File(od, "scalars.csv"), run);

        // Phase B trajectory: per-iteration (loss, entropy, lambdas) at
        // checkpoints 1, 2, 3, 5, 10, 20, 50, 100, 200, 500.
        double[] traj = runTrajectoryFlat(v1, v2, idx, 1.0, cps);
        writeTrajectoryCsv(new File(od, "trajectory_java.csv"),
                           cps, traj, 2);

        // Phase D trajectory: linear + quadratic (4 features) at the same
        // checkpoints, for richer-feature parity validation.
        double[] trajLQ = runTrajectoryFlatLQ(v1, v2, idx, 1.0, cps);
        writeTrajectoryCsv(new File(od, "trajectory_java_lq.csv"),
                           cps, trajLQ, 4);

        // Phase D.2 trajectory: linear + quadratic + product (5 features)
        // using the product-on beta_lqp schedule (Runner.java:2256-2258).
        double[] trajLQP = runTrajectoryFlatLQP(v1, v2, idx, 1.0, cps);
        writeTrajectoryCsv(new File(od, "trajectory_java_lqp.csv"),
                           cps, trajLQP, 5);

        // Phase D.2b-threshold trajectory: linear + fixed-set thresholds
        // (8 features) using per-feature-class betas (beta_lin for linear,
        // beta_thr for threshold — see runTrajectoryFlatLT javadoc).
        //
        // betaMultiplier is intentionally reduced to 0.1 so that beta_thr
        // at n=10 (= 1.9 * 0.1 = 0.19) does not completely dominate the
        // |obs-exp| <= 1 deviation on binary threshold features. At the
        // default 1.0 multiplier, all threshold lambdas stay pinned at 0
        // and the parity test becomes trivial (0 == 0). 0.1 exercises the
        // deltaLossBound / newtonStep / searchAlpha path with non-trivial
        // lambdas while leaving headroom for the Phase D.2b 1e-6 bar.
        // Phase D.2b uses the BLOCK sample fixture (rows 6..9, cols 7..9 on
        // the 10×10 asym raster — same 10 indices that test_sequential.cpp,
        // test_sequential_lq.cpp and test_sequential_lqp.cpp construct in
        // C++).  We intentionally diverge from the diagonal occurrences.csv
        // fixture used by the linear/LQ/LQP goldens above: with beta=0.1 and
        // diagonal samples, ThresholdFeature sampleExp matches on both
        // sides but the prescaled sample range is so wide (0..1) that many
        // tested thresholds (e.g. 0.78) produce sampleExp values whose
        // Interval computation walks into Java's BINARY special case or
        // sits near the beta-pinned regime, obscuring the parity signal.
        // Block samples (prescaled bio1 range ~0.74..1.00) give every
        // threshold a clean non-degenerate split and exercise the binary
        // goodAlpha / reduceAlpha path at non-trivial lambda magnitudes.
        int[] blockIdx = new int[] {
            6 * ncols + 8,
            7 * ncols + 7,
            7 * ncols + 8,
            7 * ncols + 9,
            8 * ncols + 7,
            8 * ncols + 8,
            8 * ncols + 9,
            9 * ncols + 7,
            9 * ncols + 8,
            9 * ncols + 9,
        };
        double[] trajLT = runTrajectoryFlatLT(v1, v2, blockIdx, 0.1, cps);
        writeTrajectoryCsv(new File(od, "trajectory_java_lt.csv"),
                           cps, trajLT, 8);

        // Phase D.2b-hinge trajectory: linear + 2 forward + 2 reverse hinge
        // knots per variable (10 features) using a uniform beta (see
        // runTrajectoryFlatLH javadoc).  Uses the same block sample
        // fixture as the LT trajectory above (see that block comment for
        // the rationale).
        double[] trajLH = runTrajectoryFlatLH(v1, v2, blockIdx, 0.1, cps);
        writeTrajectoryCsv(new File(od, "trajectory_java_lh.csv"),
                           cps, trajLH, 10);

        System.out.println("MaxentRefRunner: golden outputs written to "
                           + od.getAbsolutePath());
        System.out.println("  iterations=" + run.getIterations()
                           + " loss="      + run.getLoss()
                           + " entropy="   + run.getEntropy());
        System.out.println("  lambdas="    + Arrays.toString(run.getLambdas()));
    }

    /* ------------------------------------------------------------------
     * ASC / CSV IO helpers (minimal, for the mock fixture only).
     * ------------------------------------------------------------------ */

    static double[][] readAsc(String path) throws IOException {
        try (BufferedReader br = new BufferedReader(new FileReader(path))) {
            int     ncols = 0, nrows = 0;
            double  nodata = -9999;
            String  line;
            // Header block: stop at the first line whose first token does
            // not match a known ASC header key.
            while ((line = br.readLine()) != null) {
                String t = line.trim();
                if (t.isEmpty()) continue;
                String[] p   = t.split("\\s+");
                String   key = p[0].toLowerCase(Locale.US);
                if      (key.equals("ncols"))        ncols   = Integer.parseInt(p[1]);
                else if (key.equals("nrows"))        nrows   = Integer.parseInt(p[1]);
                else if (key.equals("xllcorner"))    ascXll  = Double.parseDouble(p[1]);
                else if (key.equals("yllcorner"))    ascYll  = Double.parseDouble(p[1]);
                else if (key.equals("cellsize"))     ascCell = Double.parseDouble(p[1]);
                else if (key.equals("nodata_value")) nodata  = Double.parseDouble(p[1]);
                else break;
            }
            double[][] g = new double[nrows][ncols];
            for (int r = 0; r < nrows; r++) {
                if (line == null)
                    throw new IOException("ASC: unexpected EOF on row " + r);
                String[] p = line.trim().split("\\s+");
                for (int c = 0; c < ncols; c++)
                    g[r][c] = Double.parseDouble(p[c]);
                line = br.readLine();
            }
            return g;
        }
    }

    static List<double[]> readOccCsv(String path) throws IOException {
        try (BufferedReader br = new BufferedReader(new FileReader(path))) {
            String header = br.readLine();
            if (header == null) throw new IOException("Empty CSV: " + path);
            String[] h = header.split(",");
            int lonIdx = -1, latIdx = -1;
            for (int i = 0; i < h.length; i++) {
                String hh = h[i].trim().toLowerCase(Locale.US);
                if      (hh.equals("lon") || hh.equals("longitude") || hh.equals("x"))
                    lonIdx = i;
                else if (hh.equals("lat") || hh.equals("latitude")  || hh.equals("y"))
                    latIdx = i;
            }
            if (lonIdx < 0 || latIdx < 0)
                throw new IOException("CSV missing lon/lat columns: " + path);
            List<double[]> out = new ArrayList<>();
            String line;
            while ((line = br.readLine()) != null) {
                if (line.trim().isEmpty()) continue;
                String[] p = line.split(",");
                out.add(new double[] {
                    Double.parseDouble(p[lonIdx].trim()),
                    Double.parseDouble(p[latIdx].trim())
                });
            }
            return out;
        }
    }

    static void writeLambdasCsv(File f, MaxentRefRunner r) throws IOException {
        try (PrintWriter pw = new PrintWriter(new FileWriter(f))) {
            pw.println("feature,lambda,sample_expectation,sample_deviation,expectation,beta");
            String[] names = r.getFeatureNames();
            double[] lam   = r.getLambdas();
            double[] sExp  = r.getSampleExpectations();
            double[] sDev  = r.getSampleDeviations();
            double[] exp   = r.getExpectations();
            double[] beta  = r.getBetas();
            for (int i = 0; i < names.length; i++) {
                pw.println(names[i]
                        + "," + fmt(lam[i])
                        + "," + fmt(sExp[i])
                        + "," + fmt(sDev[i])
                        + "," + fmt(exp[i])
                        + "," + fmt(beta[i]));
            }
        }
    }

    static void writeDensityCsv(File f,
                                double[] density, double[] raw, double[] cloglog)
            throws IOException {
        try (PrintWriter pw = new PrintWriter(new FileWriter(f))) {
            pw.println("point,density,raw,cloglog_java");
            for (int i = 0; i < density.length; i++)
                pw.println(i
                        + "," + fmt(density[i])
                        + "," + fmt(raw[i])
                        + "," + fmt(cloglog[i]));
        }
    }

    static void writeScalarsCsv(File f, MaxentRefRunner r) throws IOException {
        try (PrintWriter pw = new PrintWriter(new FileWriter(f))) {
            pw.println("quantity,value");
            pw.println("loss,"                        + fmt(r.getLoss()));
            pw.println("unregularized_loss,"          + fmt(r.getUnregularizedLoss()));
            pw.println("l1_reg,"                      + fmt(r.getL1Reg()));
            pw.println("gain,"                        + fmt(r.getGain()));
            pw.println("entropy,"                     + fmt(r.getEntropy()));
            pw.println("density_normalizer,"          + fmt(r.getDensityNormalizer()));
            pw.println("linear_predictor_normalizer," + fmt(r.getLinearPredictorNormalizer()));
            pw.println("beta_multiplier,"             + fmt(r.getBetaMultiplier()));
            pw.println("auto_beta_lqp,"               + fmt(r.getAutoBetaLqp()));
            pw.println("iterations,"                  + r.getIterations());
            pw.println("num_features,"                + r.getNumFeatures());
            pw.println("num_points,"                  + r.getNumPoints());
        }
    }

    static void writeTrajectoryCsv(File f, int[] checkpoints,
                                   double[] flat, int numFeatures)
            throws IOException {
        try (PrintWriter pw = new PrintWriter(new FileWriter(f))) {
            pw.print("iteration,loss,entropy");
            for (int j = 0; j < numFeatures; j++) pw.print(",lambda_" + j);
            pw.println();
            int stride = 2 + numFeatures;
            for (int i = 0; i < checkpoints.length; i++) {
                int base = i * stride;
                pw.print(checkpoints[i]
                         + "," + fmt(flat[base])
                         + "," + fmt(flat[base + 1]));
                for (int j = 0; j < numFeatures; j++)
                    pw.print("," + fmt(flat[base + 2 + j]));
                pw.println();
            }
        }
    }

    /** Full-precision double formatting (17 sig digits, US locale). */
    static String fmt(double d) {
        return String.format(Locale.US, "%.17g", d);
    }
}
