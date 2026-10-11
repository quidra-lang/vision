#!/usr/bin/env bash
set -euo pipefail
QUIDRA="$1"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PACKAGE_ROOT="$(dirname "$ROOT"):$ROOT"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
export QUIDRA_CACHE_DIR="$TMP/run-cache"

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

for element in real32 real64; do
    if [[ "$element" == "real32" ]]; then
        large=3.0e38
    else
        large=1.0e308
    fi
    cat > "$TMP/overflow.qui" <<QUI
import vision

tensor<$element> pixels = tensor.ones<$element>([1, 2, 2]) * $element($large)
tensor<$element> | error reduced = vision.downsample_mean(pixels, 2)
print("returned")
QUI
    set +e
    QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/overflow.qui" > "$TMP/overflow.out" 2>&1
    status=$?
    set -e
    if [[ "$status" -ne 101 ]] ||
       ! grep -Fq 'error[NONFINITE_RESULT]' "$TMP/overflow.out" ||
       ! grep -Fq "package operation vision.downsample_mean produced a non-finite $element value" "$TMP/overflow.out" ||
       grep -Fq 'returned' "$TMP/overflow.out"; then
        cat "$TMP/overflow.out" >&2
        exit 1
    fi
done

echo 'vision non-finite: ok'
