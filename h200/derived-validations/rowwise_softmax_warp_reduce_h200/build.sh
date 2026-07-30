#!/usr/bin/env bash
set -euo pipefail

# Build H200 FP16 Row-wise Softmax
# Usage: ./build.sh [so|bench|all]
#   so    — build shared library for Python ctypes
#   bench — build standalone benchmark executable
#   all   — build both (default)

MODE="${1:-all}"
SRC="softmax.cu"
ARCH="sm_90a"
NVCC_FLAGS="-arch=${ARCH} -O3 -std=c++17"

build_so() {
    echo "=== Building softmax.so (shared library) ==="
    nvcc ${NVCC_FLAGS} -Xcompiler -fPIC -shared "${SRC}" -o softmax.so
    echo "  -> softmax.so built"
}

build_bench() {
    echo "=== Building softmax_bench (standalone executable) ==="
    nvcc ${NVCC_FLAGS} -DSOFTMAX_BENCH "${SRC}" -o softmax_bench
    echo "  -> softmax_bench built"
}

case "$MODE" in
    so)    build_so ;;
    bench) build_bench ;;
    all)   build_so; build_bench ;;
    *)
        echo "Usage: $0 [so|bench|all]"
        exit 1
        ;;
esac
