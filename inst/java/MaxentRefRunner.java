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
