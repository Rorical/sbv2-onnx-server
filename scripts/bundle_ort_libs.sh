#!/usr/bin/env bash
set -euo pipefail

if [ $# -ne 1 ]; then
    echo "usage: $0 <path-to-target-release-dir>" >&2
    exit 1
fi

BIN_DIR="$1"
if [ ! -d "$BIN_DIR" ]; then
    echo "error: directory '$BIN_DIR' does not exist" >&2
    exit 1
fi

BIN_DIR="$(cd "$BIN_DIR" && pwd)"
BUILD_ROOT="$BIN_DIR/build"

if [ ! -d "$BUILD_ROOT" ]; then
    echo "warning: no build artifacts found at '$BUILD_ROOT'; nothing to copy" >&2
    exit 0
fi

libs_found=0
copied_names=""
while IFS= read -r -d '' lib; do
    libs_found=1
    fname="$(basename "$lib")"
    case " $copied_names " in
        *" $fname "*) continue ;;
    esac
    cp -f "$lib" "$BIN_DIR/$fname"
    echo "copied $fname -> $BIN_DIR"
    copied_names="$copied_names $fname"
done < <(find "$BUILD_ROOT" -path "*/lib/*" \( -type f -o -type l \) \( -name "libonnxruntime*.so*" -o -name "libonnxruntime*.dylib" -o -name "onnxruntime.dll" \) -print0)

if [ "$libs_found" -eq 0 ]; then
    echo "error: no ONNX Runtime libraries found under '$BUILD_ROOT'" >&2
    exit 1
fi
