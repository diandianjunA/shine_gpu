#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DEST_DIR="${1:-}"

if [[ -z "$DEST_DIR" ]]; then
    echo "Usage: $0 <destination-project-dir>" >&2
    exit 1
fi

RUNTIME_LIB="$(find "$PROJECT_DIR/build" -name libshine_runtime.a -print -quit)"
RDMA_LIB="$(find "$PROJECT_DIR/build" -name librdma_library.a -print -quit)"

if [[ -z "$RUNTIME_LIB" || -z "$RDMA_LIB" ]]; then
    echo "error: required libraries not found under $PROJECT_DIR/build" >&2
    echo "build the project first: cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j" >&2
    exit 1
fi

mkdir -p "$DEST_DIR/lib" "$DEST_DIR/include/shine"
rm -rf "$DEST_DIR/include/shine/src" \
       "$DEST_DIR/include/shine/rdma-library" \
       "$DEST_DIR/include/shine/thirdparty"

cp "$RUNTIME_LIB" "$DEST_DIR/lib/libshine_runtime.a"
cp "$RDMA_LIB" "$DEST_DIR/lib/librdma_library.a"
cp -r "$PROJECT_DIR/src" "$DEST_DIR/include/shine/src"
cp -r "$PROJECT_DIR/rdma-library" "$DEST_DIR/include/shine/rdma-library"
cp -r "$PROJECT_DIR/thirdparty" "$DEST_DIR/include/shine/thirdparty"
rm -rf "$DEST_DIR/include/shine/thirdparty/httplib"

echo "packaged SHINE main runtime into $DEST_DIR"
