#!/bin/sh

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

: "${TT_METAL_HOME:?TT_METAL_HOME is not set. Export it before running this project.}"
: "${TT_METAL_SIMULATOR:?TT_METAL_SIMULATOR is not set. Export it before running this project.}"
: "${TT_METAL_SLOW_DISPATCH_MODE:=1}"
: "${TT_METAL_RUNTIME_ROOT:=$TT_METAL_HOME}"

export TT_METAL_HOME
export TT_METAL_SIMULATOR
export TT_METAL_SLOW_DISPATCH_MODE
export TT_METAL_RUNTIME_ROOT
export TT_VISIBLE_DEVICES="$1"

# Possíveis valores para DPRINT_CORES
export TT_METAL_DPRINT_CORES="all"
# export TT_METAL_DPRINT_CORES="0,1"
# export TT_METAL_DPRINT_CORES="(0,0)-(1,1)"
# export TT_METAL_DPRINT_CORES="(0,0),(0,1)"

# Valid values are BR,NC,TR0,TR1,TR2,TR*,ER0,ER1,ER*.
# export TT_METAL_DPRINT_RISCVS=TR*
# export TT_METAL_DPRINT_RISCVS=BR,NC

# $TT_METAL_HOME/build_Release/tools/profiler/bin/capture-release -f -o /tmp/local-tracy-output.tracy &
BIN_PATH="${JACOBI_BINARY:-$SCRIPT_DIR/build/jacobi_deslocamento_sistolico}"
if [ ! -x "$BIN_PATH" ]; then
    BIN_PATH="${JACOBI_BINARY:-$SCRIPT_DIR/build_Debug/jacobi_deslocamento_sistolico}"
fi

if [ ! -x "$BIN_PATH" ]; then
    printf '%s\n' "Error: jacobi binary not found. Set JACOBI_BINARY or build the project in $SCRIPT_DIR/build." >&2
    exit 1
fi

distrobox enter ubuntu -- "$BIN_PATH" "$2" "$3" "$4" "$5" "$6"
