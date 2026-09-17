#!/usr/bin/env bash
set -euo pipefail

QUIDRA="$1"
REPOSITORY_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PACKAGE_ROOT="$(dirname "$REPOSITORY_ROOT")"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

cat > "$TMP/use-vision.qui" <<'QUI'
import vision

tensor<uint8> pixels = tensor.zeros<uint8>([3, 2, 3])
int value = 0
for channel in range(3)
    for y in range(2)
        for x in range(3)
            pixels[channel, y, x] = uint8(value)
            value += 1

tensor<uint8> cropped = vision.crop(pixels, top = 0, left = 1, height = 2, width = 2)
tensor<uint8> resized = vision.resize(pixels, height = 4, width = 6)
tensor<uint8> flipped = vision.flip_horizontal(pixels)
tensor<uint8> rotated = vision.rotate90(pixels)
tensor<uint8> gray = vision.grayscale(pixels)
tensor<uint8> binary = vision.threshold(pixels, cutoff = uint8(8))

print(cropped.shape()[2])
print(cropped[0, 0, 0].item())
print(resized.shape()[1])
print(resized.shape()[2])
print(flipped[0, 0, 0].item())
print(rotated.shape()[1])
print(rotated.shape()[2])
print(gray.shape()[0])
print(binary[0, 0, 0].item())
print(binary[2, 1, 2].item())
QUI

output="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/use-vision.qui")"
expected="$(printf '2\n1\n4\n6\n2\n3\n2\n1\n0\n255')"
if [[ "$output" != "$expected" ]]; then
    echo "unexpected vision output: $output" >&2
    exit 1
fi

cat > "$TMP/filters.qui" <<'QUI'
import vision

tensor<uint8> impulse = tensor.zeros<uint8>([1, 3, 3])
impulse[0, 1, 1] = uint8(255)
tensor<uint8> blurred = vision.blur(impulse, radius = 1)
tensor<uint8> expanded = vision.dilate(impulse, radius = 1)
tensor<uint8> contracted = vision.erode(impulse, radius = 1)
tensor<int> kernel = tensor.zeros<int>([3, 3])
kernel[1, 1] = 1
tensor<uint8> filtered = vision.filter(impulse, kernel)

print(blurred[0, 1, 1].item())
print(expanded[0, 0, 0].item())
print(contracted[0, 1, 1].item())
print(filtered[0, 1, 1].item())
QUI

filters_output="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/filters.qui")"
filters_expected="$(printf '28\n255\n0\n255')"
if [[ "$filters_output" != "$filters_expected" ]]; then
    echo "unexpected vision filter output: $filters_output" >&2
    exit 1
fi

cat > "$TMP/rotations.qui" <<'QUI'
import vision

tensor<uint8> pixels = tensor.zeros<uint8>([1, 2, 3])
int value = 1
for y in range(2)
    for x in range(3)
        pixels[0, y, x] = uint8(value)
        value += 1

tensor<uint8> turned90 = vision.rotate90(pixels)
tensor<uint8> turned180 = vision.rotate180(pixels)
tensor<uint8> turned270 = vision.rotate270(pixels)
tensor<uint8> mirrored = vision.flip_vertical(pixels)

print(turned90[0, 0, 0].item())
print(turned90[0, 2, 1].item())
print(turned180[0, 0, 0].item())
print(turned180[0, 1, 2].item())
print(turned270[0, 0, 0].item())
print(turned270[0, 2, 1].item())
print(mirrored[0, 0, 0].item())

tensor<uint8> full_turn = vision.rotate90(vision.rotate270(pixels))
tensor<uint8> half_turn = vision.rotate90(vision.rotate90(pixels))
int differences = 0
for y in range(2)
    for x in range(3)
        if full_turn[0, y, x].item() != pixels[0, y, x].item()
            differences += 1
        if half_turn[0, y, x].item() != turned180[0, y, x].item()
            differences += 1
print(differences)
QUI

rotations_output="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/rotations.qui")"
rotations_expected="$(printf '4\n3\n6\n1\n3\n4\n4\n0')"
if [[ "$rotations_output" != "$rotations_expected" ]]; then
    echo "unexpected vision rotation output: $rotations_output" >&2
    exit 1
fi

echo "vision integration: ok"
