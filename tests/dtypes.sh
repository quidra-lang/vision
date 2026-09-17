#!/usr/bin/env bash
set -euo pipefail

QUIDRA="$1"
REPOSITORY_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PACKAGE_ROOT="$(dirname "$REPOSITORY_ROOT")"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

cat > "$TMP/vision-dtypes.qui" <<'QUI'
import vision

// The documented grayscale coefficients must be used directly, not the old
// 77/150/29 over 256 integer approximation.
tensor<uint8> rgb = tensor.zeros<uint8>([3, 1, 2])
rgb[1, 0, 0] = uint8(255)
rgb[2, 0, 1] = uint8(255)
tensor<uint8> gray = vision.grayscale(rgb)
print(gray[0, 0, 0].item())
print(gray[0, 0, 1].item())

// Geometry only relocates samples, so it must preserve the source dtype.
tensor<uint16> pixels = tensor.zeros<uint16>([1, 2, 2])
pixels[0, 0, 0] = uint16(1000)
pixels[0, 0, 1] = uint16(2000)
pixels[0, 1, 0] = uint16(3000)
pixels[0, 1, 1] = uint16(4000)

tensor<uint16> cropped = vision.crop(pixels, top = 0, left = 1, height = 2, width = 1)
tensor<uint16> resized = vision.resize(pixels, height = 4, width = 4)
tensor<uint16> flipped = vision.flip_horizontal(pixels)
tensor<uint16> turned = vision.rotate90(pixels)

print(cropped[0, 1, 0].item())
print(resized[0, 3, 3].item())
print(flipped[0, 0, 0].item())
print(turned[0, 0, 1].item())
QUI

output="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/vision-dtypes.qui")"
expected="$(printf '150\n29\n4000\n4000\n2000\n1000')"
if [[ "$output" != "$expected" ]]; then
    echo "unexpected vision dtype output:" >&2
    printf '%s\n' "$output" >&2
    exit 1
fi

echo "vision dtype integration: ok"
