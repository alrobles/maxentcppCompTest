#!/bin/bash
# build.sh -- Compile MaxentMini.java and create maxent_mini.jar
# Place this script in inst/java/ and run it from there (or from any directory).
# The JAR is written to the same directory as this script.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_FILE="$SCRIPT_DIR/MaxentMini.java"
JAR_FILE="$SCRIPT_DIR/maxent_mini.jar"
CLASS_DIR="$SCRIPT_DIR/classes"

if [ ! -f "$SRC_FILE" ]; then
    echo "ERROR: MaxentMini.java not found in $SCRIPT_DIR" >&2
    exit 1
fi

echo "Compiling $SRC_FILE ..."
mkdir -p "$CLASS_DIR"

if ! command -v javac &>/dev/null; then
    echo "ERROR: javac not found. Please install a Java Development Kit (JDK >= 8)." >&2
    exit 1
fi

javac -d "$CLASS_DIR" "$SRC_FILE"

echo "Creating $JAR_FILE ..."
jar cf "$JAR_FILE" -C "$CLASS_DIR" .

echo "Done. JAR written to $JAR_FILE"
