#!/usr/bin/env bash
# build_ref.sh -- Compile MaxentRefRunner.java against the unmodified
# density/*.java source tree from alrobles/Maxent, and package everything
# (including parameters.csv) into maxent_ref.jar.
#
# Layout assumed by default:
#   ~/repos/Maxent/density/*.java         (the real Java Maxent 3.4.4 source)
#   ~/repos/maxentcppCompTest/inst/java/  (this directory)
#
# Override the Maxent source tree by setting MAXENT_SRC:
#   MAXENT_SRC=/path/to/Maxent/density ./build_ref.sh
#
# Phase A (maxentcpp issues #36 / #37): establishes a real-Java reference
# oracle so that maxentcpp C++ outputs can be validated against the real
# density.Sequential optimizer (not against the goodAlpha-only MaxentMini).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_FILE="$SCRIPT_DIR/MaxentRefRunner.java"
JAR_FILE="$SCRIPT_DIR/maxent_ref.jar"
CLASS_DIR="$SCRIPT_DIR/classes_ref"

# Default location: sibling of the maxentcppCompTest checkout
MAXENT_SRC="${MAXENT_SRC:-$(realpath "$SCRIPT_DIR/../../../Maxent/density" 2>/dev/null || true)}"

if [ ! -f "$SRC_FILE" ]; then
    echo "ERROR: MaxentRefRunner.java not found in $SCRIPT_DIR" >&2
    exit 1
fi
if [ -z "$MAXENT_SRC" ] || [ ! -d "$MAXENT_SRC" ]; then
    echo "ERROR: Maxent source tree not found." >&2
    echo "       Expected at: $SCRIPT_DIR/../../../Maxent/density" >&2
    echo "       Or set MAXENT_SRC=/path/to/alrobles/Maxent/density" >&2
    exit 1
fi
if ! command -v javac >/dev/null 2>&1; then
    echo "ERROR: javac not found (install a JDK >= 8)." >&2
    exit 1
fi

echo "Using Maxent source: $MAXENT_SRC"
rm -rf "$CLASS_DIR"
mkdir -p "$CLASS_DIR"

# Collect all density/*.java except files that fail to compile on modern
# JDKs (Extractor.java has pre-existing double[]/float[] errors; those
# classes are not reachable from MaxentRefRunner's code path).
EXCLUDE_REGEX='(Extractor|Explain|Explainold|Getval|NceasApply|AUC)\.java$'
SRC_LIST="$(mktemp)"
trap 'rm -f "$SRC_LIST"' EXIT
find "$MAXENT_SRC" -maxdepth 1 -name '*.java' \
    | grep -Ev "$EXCLUDE_REGEX" > "$SRC_LIST"
echo "$SRC_FILE" >> "$SRC_LIST"

echo "Compiling $(wc -l < "$SRC_LIST") Java files ..."
# -Xlint:none silences the many deprecation warnings in the original source.
# We deliberately do NOT pass -Werror; broken Extractor.java would otherwise
# terminate the build. MaxentRefRunner does not use Extractor.
# -sourcepath lets javac resolve subpackages such as density.tools.* that are
# referenced from Runner.java but not enumerated in $SRC_LIST.
MAXENT_ROOT="$(dirname "$MAXENT_SRC")"
javac -Xlint:none -sourcepath "$MAXENT_ROOT" \
      -d "$CLASS_DIR" "@$SRC_LIST" 2> "$CLASS_DIR/.javac.log" || true

# Pre-existing compile errors in Extractor.java (double[] vs float[]) are
# not on MaxentRefRunner's code path; accept them as long as
# MaxentRefRunner.class and its dependencies are produced.
if [ ! -f "$CLASS_DIR/density/MaxentRefRunner.class" ]; then
    echo "ERROR: MaxentRefRunner.class was not produced." >&2
    echo "---- javac log ----" >&2
    cat "$CLASS_DIR/.javac.log" >&2
    exit 2
fi

ONLY_EXTRACTOR_ERRORS="$(grep -E '\.java:[0-9]+: error:' "$CLASS_DIR/.javac.log" \
                       | grep -vE '/Extractor\.java:' || true)"
if [ -n "$ONLY_EXTRACTOR_ERRORS" ]; then
    echo "WARNING: unexpected javac errors in non-Extractor files:" >&2
    echo "$ONLY_EXTRACTOR_ERRORS" >&2
fi

# density/parameters.csv is loaded via getResourceAsStream at Params init.
cp "$MAXENT_SRC/parameters.csv" "$CLASS_DIR/density/parameters.csv"

echo "Creating $JAR_FILE ..."
jar cf "$JAR_FILE" -C "$CLASS_DIR" .

echo "Done. Reference JAR written to $JAR_FILE"
echo "Smoke test:"
java -cp "$JAR_FILE" density.MaxentRefRunner \
    "$SCRIPT_DIR/../extdata/bio1.asc" \
    "$SCRIPT_DIR/../extdata/bio2.asc" \
    "$SCRIPT_DIR/../extdata/occurrences.csv" \
    "$SCRIPT_DIR/../extdata/golden" 2>&1 | tail -5
