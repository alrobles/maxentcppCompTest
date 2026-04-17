#!/usr/bin/env python3
"""Compare per-iteration trajectories of three maxentcpp optimizers.

Reads trajectory_{java,mini,cpp}.csv (columns: iteration, loss, entropy,
lambda_0, lambda_1) from a golden directory and prints two tables:

  (A) Per-iteration L_inf(lambda_X - lambda_java) and |loss_X - loss_java|
      for X in {cpp, mini}.
  (B) Final-iteration summary: L_inf(lambda), |loss|, |entropy|.

Intended for Phase B of the maxentcpp fidelity plan (issues #36 / #37).
Does NOT require numpy / pandas — pure stdlib so it can run on any VM.

Usage:
    python3 tools/compare_trajectories.py inst/extdata/golden/
    python3 tools/compare_trajectories.py inst/extdata/golden/asym/
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path


def load_trajectory(path: Path) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    with path.open() as fh:
        rdr = csv.DictReader(fh)
        for row in rdr:
            it = int(row["iteration"])
            out[it] = {
                "loss":     float(row["loss"]),
                "entropy":  float(row["entropy"]),
                "lambda_0": float(row["lambda_0"]),
                "lambda_1": float(row["lambda_1"]),
            }
    return out


def linf_lambda(a: dict[str, float], b: dict[str, float]) -> float:
    return max(abs(a["lambda_0"] - b["lambda_0"]),
               abs(a["lambda_1"] - b["lambda_1"]))


def summarize(golden_dir: Path) -> None:
    j = load_trajectory(golden_dir / "trajectory_java.csv")
    m = load_trajectory(golden_dir / "trajectory_mini.csv")
    c = load_trajectory(golden_dir / "trajectory_cpp.csv")

    iters = sorted(set(j) & set(m) & set(c))

    print(f"\n=== Fixture: {golden_dir} ===")
    hdr = (f"{'iter':>5}  "
           f"{'||λ_cpp−λ_java||∞':>20}  "
           f"{'|loss_cpp−loss_java|':>22}  "
           f"{'||λ_mini−λ_java||∞':>22}  "
           f"{'|loss_mini−loss_java|':>24}")
    print(hdr)
    print("-" * len(hdr))
    for it in iters:
        dl_c  = linf_lambda(c[it], j[it])
        dloss_c = abs(c[it]["loss"] - j[it]["loss"])
        dl_m  = linf_lambda(m[it], j[it])
        dloss_m = abs(m[it]["loss"] - j[it]["loss"])
        print(f"{it:>5}  "
              f"{dl_c:>20.3e}  "
              f"{dloss_c:>22.3e}  "
              f"{dl_m:>22.3e}  "
              f"{dloss_m:>24.3e}")

    # Final-iteration summary
    last = iters[-1]
    print(f"\nFinal iteration ({last}):")
    for tag, tr in (("java", j), ("cpp", c), ("mini", m)):
        lam = (tr[last]["lambda_0"], tr[last]["lambda_1"])
        print(f"  {tag:>4}: loss={tr[last]['loss']:.16g}  "
              f"entropy={tr[last]['entropy']:.16g}  "
              f"lambda=[{lam[0]:.16g}, {lam[1]:.16g}]")
    print(f"  Δ(cpp,  java) L∞(λ) = {linf_lambda(c[last], j[last]):.3e}")
    print(f"  Δ(mini, java) L∞(λ) = {linf_lambda(m[last], j[last]):.3e}")
    print(f"  Δ(cpp,  mini) L∞(λ) = {linf_lambda(c[last], m[last]):.3e}")


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: compare_trajectories.py <golden-dir> [more-dirs...]",
              file=sys.stderr)
        return 2
    for arg in sys.argv[1:]:
        summarize(Path(arg))
    return 0


if __name__ == "__main__":
    sys.exit(main())
