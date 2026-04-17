// trajectory_cpp.cpp
// --------------------------------------------------------------------------
// Phase B (maxentcpp #36 / #37): quantify the maxentcpp C++ vs real-Java gap.
//
// Standalone CLI that drives maxentcpp's C++ `FeaturedSpace::train()` on a
// caller-supplied ASC + occurrence fixture and dumps one row per iteration
// checkpoint with (loss, entropy, lambda_0, lambda_1).
//
// To keep the comparison with `density.MaxentRefRunner` fully black-box,
// each checkpoint is computed from a **fresh** FeaturedSpace and trained
// for exactly that many iterations with convergence_threshold=0 (so the
// 20-iter early-out never fires). Since maxentcpp's train() is fully
// deterministic this reproduces the per-iteration state without needing
// an instrumented hook inside the library.
//
// Build:
//   g++ -std=c++17 -O2 \
//       -I <maxentcpp>/src/cpp/include \
//       -o trajectory_cpp \
//       trajectory_cpp.cpp <maxentcpp>/src/cpp/src/*.cpp
//
// Run:
//   trajectory_cpp bio1.asc bio2.asc occurrences.csv trajectory_cpp.csv
// --------------------------------------------------------------------------

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "maxent/feature.hpp"
#include "maxent/featured_space.hpp"

namespace {

struct AscGrid {
    int    ncols  = 0;
    int    nrows  = 0;
    double xll    = 0.0;
    double yll    = 0.0;
    double cell   = 1.0;
    double nodata = -9999.0;
    std::vector<double> values;  // row-major, nrows * ncols
};

AscGrid read_asc(const std::string& path)
{
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open ASC: " + path);

    AscGrid g;
    std::string line;
    std::string leftover;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        std::string key;
        ss >> key;
        std::string lkey = key;
        for (auto& ch : lkey) ch = (char)std::tolower((unsigned char)ch);
        if      (lkey == "ncols")        ss >> g.ncols;
        else if (lkey == "nrows")        ss >> g.nrows;
        else if (lkey == "xllcorner")    ss >> g.xll;
        else if (lkey == "yllcorner")    ss >> g.yll;
        else if (lkey == "cellsize")     ss >> g.cell;
        else if (lkey == "nodata_value") ss >> g.nodata;
        else { leftover = line; break; }
    }
    g.values.reserve((size_t)g.ncols * (size_t)g.nrows);
    auto push_line = [&](const std::string& s) {
        std::istringstream ls(s);
        double v;
        while (ls >> v) g.values.push_back(v);
    };
    if (!leftover.empty()) push_line(leftover);
    double v;
    while (in >> v) g.values.push_back(v);
    if ((int)g.values.size() != g.ncols * g.nrows)
        throw std::runtime_error("ASC: expected " +
            std::to_string(g.ncols * g.nrows) + " values, got " +
            std::to_string(g.values.size()) + " in " + path);
    return g;
}

std::vector<std::pair<double,double>> read_occ(const std::string& path)
{
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open occurrences: " + path);
    std::string header;
    std::getline(in, header);
    std::vector<std::string> cols;
    {
        std::istringstream hs(header);
        std::string c;
        while (std::getline(hs, c, ',')) {
            for (auto& ch : c) ch = (char)std::tolower((unsigned char)ch);
            // strip CR / whitespace
            while (!c.empty() &&
                   (c.back() == '\r' || c.back() == ' ' || c.back() == '\t'))
                c.pop_back();
            cols.push_back(c);
        }
    }
    int lon_col = -1, lat_col = -1;
    for (size_t i = 0; i < cols.size(); ++i) {
        if (cols[i] == "lon" || cols[i] == "longitude" || cols[i] == "x")
            lon_col = (int)i;
        else if (cols[i] == "lat" || cols[i] == "latitude" || cols[i] == "y")
            lat_col = (int)i;
    }
    if (lon_col < 0 || lat_col < 0)
        throw std::runtime_error("occurrences.csv: missing lon/lat columns");

    std::vector<std::pair<double,double>> out;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::vector<std::string> fields;
        std::istringstream ls(line);
        std::string f;
        while (std::getline(ls, f, ',')) fields.push_back(f);
        if ((int)fields.size() <= std::max(lon_col, lat_col)) continue;
        out.emplace_back(std::stod(fields[lon_col]), std::stod(fields[lat_col]));
    }
    return out;
}

std::vector<int> occ_to_indices(const std::vector<std::pair<double,double>>& occ,
                                const AscGrid& g)
{
    std::vector<int> idx;
    idx.reserve(occ.size());
    for (auto [lon, lat] : occ) {
        int row = (int)std::floor((g.yll + g.nrows * g.cell - lat) / g.cell);
        int col = (int)std::floor((lon - g.xll) / g.cell);
        if (row < 0)         row = 0;
        if (row >= g.nrows)  row = g.nrows - 1;
        if (col < 0)         col = 0;
        if (col >= g.ncols)  col = g.ncols - 1;
        idx.push_back(row * g.ncols + col);
    }
    return idx;
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 5) {
        std::fprintf(stderr,
            "Usage: trajectory_cpp bio1.asc bio2.asc occurrences.csv out.csv\n");
        return 2;
    }
    try {
        AscGrid g1 = read_asc(argv[1]);
        AscGrid g2 = read_asc(argv[2]);
        if (g1.ncols != g2.ncols || g1.nrows != g2.nrows)
            throw std::runtime_error("bio1 and bio2 grid shapes differ");

        auto occ = read_occ(argv[3]);
        auto idx = occ_to_indices(occ, g1);

        const int n = g1.ncols * g1.nrows;
        auto v1 = std::make_shared<std::vector<double>>(g1.values);
        auto v2 = std::make_shared<std::vector<double>>(g2.values);
        double v1min = *std::min_element(v1->begin(), v1->end());
        double v1max = *std::max_element(v1->begin(), v1->end());
        double v2min = *std::min_element(v2->begin(), v2->end());
        double v2max = *std::max_element(v2->begin(), v2->end());

        const std::vector<int> checkpoints = {1, 2, 3, 5, 10, 20, 50, 100, 200, 500};

        std::ofstream out(argv[4]);
        if (!out) throw std::runtime_error(std::string("cannot write: ") + argv[4]);
        out.setf(std::ios::scientific);
        out.precision(16);
        // loss = regularized loss (= X.getLoss()+reg in Java) so Phase B
        // can diff C++ against Java trajectory_java.csv directly.
        out << "iteration,loss,entropy,lambda_0,lambda_1,"
               "loss_unreg,l1_reg\n";

        for (int k : checkpoints) {
            auto f1 = std::make_shared<maxent::LinearFeature>(v1, "bio1", v1min, v1max);
            auto f2 = std::make_shared<maxent::LinearFeature>(v2, "bio2", v2min, v2max);
            maxent::FeaturedSpace fs(
                n,
                std::vector<int>(idx),
                std::vector<std::shared_ptr<maxent::Feature>>{f1, f2});
            // convergence_threshold=0 ⇒ the 20-iter early-out never fires
            auto res = fs.train(k, /*conv=*/0.0, /*beta_mult=*/1.0,
                                /*min_dev=*/0.001);
            double reg        = fs.get_l1_reg();
            double loss_reg   = res.loss + reg;
            out << k
                << ',' << loss_reg
                << ',' << res.entropy
                << ',' << res.lambdas[0]
                << ',' << res.lambdas[1]
                << ',' << res.loss
                << ',' << reg
                << '\n';
        }
        std::cout << "trajectory_cpp: wrote " << argv[4] << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "trajectory_cpp: %s\n", e.what());
        return 1;
    }
}
