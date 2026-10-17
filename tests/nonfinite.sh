#!/usr/bin/env bash
set -euo pipefail
QUIDRA="$1"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PACKAGE_ROOT="$(dirname "$ROOT"):$ROOT"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
export QUIDRA_CACHE_DIR="${QUIDRA_CACHE_DIR:-$TMP/run-cache}"

cat > "$TMP/finite.qui" <<'QUI'
import vision

tensor<real32> | error reduced = vision.downsample_mean(
    tensor.ones<real32>([1, 2, 2]), 2
)
match reduced
    tensor<real32> value
        print(value[0, 0, 0].item() == real32(1.0))
    error
        print(false)
print(NL)
QUI
finite="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/finite.qui")"
if [[ "$finite" != "true" ]]; then
    printf 'Vision finite block mean mismatch: %s\n' "$finite" >&2
    exit 1
fi

# A finite input can overflow an intermediate block sum to infinity.
# Infinity is valid; an indeterminate block (+inf + -inf) is not.
for element in real32 real64; do
    if [[ "$element" == real32 ]]; then large=3.0e38; else large=1.0e308; fi
    cat > "$TMP/overflow.qui" <<QUI
import vision
tensor<$element> pixels = tensor.ones<$element>([1, 2, 2]) * $element($large)
tensor<$element> | error reduced = vision.downsample_mean(pixels, 2)
match reduced
    tensor<$element> value
        print(value[0, 0, 0].item() == $element(inf))
    error
        print(false)
print(NL)
QUI
    result="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/overflow.qui")"
    if [[ "$result" != true ]]; then
        echo "Vision infinity block mean mismatch for $element: $result" >&2
        exit 1
    fi

    cat > "$TMP/opposed.qui" <<QUI
import vision
$element infinity = $element(1) / $element(0)
tensor<$element> pixels = tensor.ones<$element>([1, 2, 2])
pixels[0, 0, 0] = infinity
pixels[0, 0, 1] = -infinity
print("before")
print(NL)
tensor<$element> | error reduced = vision.downsample_mean(pixels, 2)
print("after")
print(NL)
QUI
    set +e
    QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/opposed.qui" \
        > "$TMP/opposed.out" 2> "$TMP/opposed.err"
    status=$?
    set -e
    if [[ "$status" -ne 101 || "$(cat "$TMP/opposed.out")" != before ]] ||
       ! grep -Fq 'Quidra runtime error[NONFINITE_RESULT]' "$TMP/opposed.err" ||
       ! grep -Fq "package operation vision.downsample_mean produced a NaN $element value" "$TMP/opposed.err"; then
        echo "Vision opposite infinity was not rejected for $element (status $status):" >&2
        cat "$TMP/opposed.out" "$TMP/opposed.err" >&2
        exit 1
    fi
done

# Write tiny uncompressed float TIFFs with Python's standard library only.
# No PIL/tifffile dependency is introduced into CI.
python3 - "$TMP" <<'PY_TIFF'
import math
from pathlib import Path
import struct
import sys
base = Path(sys.argv[1])
for element, bits, kind in (("real32", 32, "f"), ("real64", 64, "d")):
    for name, values in (("inf", (math.inf, -math.inf, 1.0, 2.0)),
                         ("nan", (math.nan, 2.0, 3.0, 4.0))):
        samples = struct.pack("<4" + kind, *values)
        tags = [(256, 4, 1, 2), (257, 4, 1, 2), (258, 3, 1, bits),
                (259, 3, 1, 1), (262, 3, 1, 1), (273, 4, 1, 0),
                (277, 3, 1, 1), (278, 4, 1, 2),
                (279, 4, 1, len(samples)), (284, 3, 1, 1), (339, 3, 1, 3)]
        offset = 8 + 2 + 12 * len(tags) + 4
        tags = [(tag, typ, count, offset if tag == 273 else value)
                for tag, typ, count, value in tags]
        header = b"II" + struct.pack("<HIH", 42, 8, len(tags))
        ifd = b"".join(struct.pack("<HHII", *tag) for tag in tags) + struct.pack("<I", 0)
        (base / f"{element}-{name}.tiff").write_bytes(header + ifd + samples)
PY_TIFF

for element in real32 real64; do
    cat > "$TMP/read-infinity.qui" <<QUI
import vision
tensor<$element> | error result = vision.read<$element>("$TMP/$element-inf.tiff")
match result
    tensor<$element> data
        print(data[0, 0, 0].item() == $element(inf) and data[0, 0, 1].item() == $element(-inf))
    error
        print(false)
print(NL)
QUI
    got="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/read-infinity.qui")"
    if [[ "$got" != true ]]; then
        echo "Vision TIFF $element infinity read mismatch: $got" >&2
        exit 1
    fi

    cat > "$TMP/read-nan.qui" <<QUI
import vision
print("before")
print(NL)
tensor<$element> | error result = vision.read<$element>("$TMP/$element-nan.tiff")
print("after")
print(NL)
QUI
    set +e
    QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/read-nan.qui" \
        > "$TMP/read-nan.out" 2> "$TMP/read-nan.err"
    status=$?
    set -e
    if [[ "$status" -ne 101 || "$(cat "$TMP/read-nan.out")" != before ]] ||
       ! grep -Fq 'Quidra runtime error[NONFINITE_RESULT]' "$TMP/read-nan.err" ||
       ! grep -Fq "package operation vision.read produced a NaN $element value" "$TMP/read-nan.err"; then
        echo "Vision TIFF $element NaN was not rejected (status $status):" >&2
        cat "$TMP/read-nan.out" "$TMP/read-nan.err" >&2
        exit 1
    fi
done

echo 'vision NaN/infinity contracts: ok'
