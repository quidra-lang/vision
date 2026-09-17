#!/usr/bin/env bash
set -euo pipefail

QUIDRA="$1"
REPOSITORY_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PACKAGE_ROOT="$(dirname "$REPOSITORY_ROOT")"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

cat > "$TMP/boundary.qui" <<'QUI'
import vision

tensor<uint8> pixels = tensor.zeros<uint8>([3, 2, 3])

bool resize_rejected = false
tensor<uint8> | error bad_resize = vision.resize(pixels, height = 0, width = 2)
match bad_resize
    tensor<uint8>
        resize_rejected = false
    error
        resize_rejected = true
print(resize_rejected)

bool dilate_rejected = false
tensor<uint8> | error bad_dilate = vision.dilate(pixels, radius = -1)
match bad_dilate
    tensor<uint8>
        dilate_rejected = false
    error
        dilate_rejected = true
print(dilate_rejected)

bool erode_rejected = false
tensor<uint8> | error bad_erode = vision.erode(pixels, radius = -1)
match bad_erode
    tensor<uint8>
        erode_rejected = false
    error
        erode_rejected = true
print(erode_rejected)
QUI

output="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/boundary.qui")"
expected="$(printf 'true\ntrue\ntrue')"
if [[ "$output" != "$expected" ]]; then
    echo "unexpected vision boundary output: $output" >&2
    exit 1
fi

cat > "$TMP/properties.qui" <<'QUI'
import vision

tensor<uint8> pixels = tensor.zeros<uint8>([1, 2, 3])
int value = 1
for y in range(2)
    for x in range(3)
        pixels[0, y, x] = uint8(value)
        value += 1

tensor<uint8> flipped_twice = vision.flip_horizontal(vision.flip_horizontal(pixels))
tensor<uint8> rotated_four = vision.rotate90(
    vision.rotate90(vision.rotate90(vision.rotate90(pixels)))
)
tensor<uint8> resized_same = vision.resize(pixels, height = 2, width = 3)
tensor<uint8> dilated_zero = vision.dilate(pixels, radius = 0)
tensor<uint8> eroded_zero = vision.erode(pixels, radius = 0)

int differences = 0
for y in range(2)
    for x in range(3)
        uint8 expected = pixels[0, y, x].item()
        if flipped_twice[0, y, x].item() != expected
            differences += 1
        if rotated_four[0, y, x].item() != expected
            differences += 1
        if resized_same[0, y, x].item() != expected
            differences += 1
        if dilated_zero[0, y, x].item() != expected
            differences += 1
        if eroded_zero[0, y, x].item() != expected
            differences += 1
print(differences)
QUI

properties_output="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/properties.qui")"
if [[ "$properties_output" != "0" ]]; then
    echo "unexpected vision property output: $properties_output" >&2
    exit 1
fi

echo "vision boundary tests: ok"
