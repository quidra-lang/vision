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

bool crop_rejected = false
tensor<uint8> | error bad_crop = vision.crop(pixels, top = 1, left = 2, height = 2, width = 2)
match bad_crop
    tensor<uint8>
        crop_rejected = false
    error
        crop_rejected = true
print(crop_rejected)

bool grayscale_rejected = false
tensor<uint8> two_channels = tensor.zeros<uint8>([2, 2, 2])
tensor<uint8> | error bad_grayscale = vision.grayscale(two_channels)
match bad_grayscale
    tensor<uint8>
        grayscale_rejected = false
    error
        grayscale_rejected = true
print(grayscale_rejected)

bool blur_rejected = false
tensor<uint8> | error bad_blur = vision.blur(pixels, radius = -1)
match bad_blur
    tensor<uint8>
        blur_rejected = false
    error
        blur_rejected = true
print(blur_rejected)

bool filter_rejected = false
tensor<int> kernel = tensor.ones<int>([3, 3])
tensor<uint8> | error bad_filter = vision.filter(pixels, kernel, divisor = 0)
match bad_filter
    tensor<uint8>
        filter_rejected = false
    error
        filter_rejected = true
print(filter_rejected)

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

bool rank_rejected = false
tensor<uint8> rank_two = tensor.zeros<uint8>([2, 3])
tensor<uint8> | error bad_flip = vision.flip_horizontal(rank_two)
match bad_flip
    tensor<uint8>
        rank_rejected = false
    error
        rank_rejected = true
print(rank_rejected)
QUI

output="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/boundary.qui")"
expected="$(printf 'true\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue')"
if [[ "$output" != "$expected" ]]; then
    echo "unexpected vision boundary output: $output" >&2
    exit 1
fi

cat > "$TMP/properties.qui" <<'QUI'
import vision

int | error run()
    tensor<uint8> pixels = tensor.zeros<uint8>([1, 2, 3])
    int value = 1
    for y in range(2)
        for x in range(3)
            pixels[0, y, x] = uint8(value)
            value += 1

    tensor<uint8> flipped_once = try vision.flip_horizontal(pixels)
    tensor<uint8> flipped_twice = try vision.flip_horizontal(flipped_once)
    tensor<uint8> r1 = try vision.rotate90(pixels)
    tensor<uint8> r2 = try vision.rotate90(r1)
    tensor<uint8> r3 = try vision.rotate90(r2)
    tensor<uint8> rotated_four = try vision.rotate90(r3)
    tensor<uint8> resized_same = try vision.resize(pixels, height = 2, width = 3)
    tensor<uint8> dilated_zero = try vision.dilate(pixels, radius = 0)
    tensor<uint8> eroded_zero = try vision.erode(pixels, radius = 0)
    tensor<uint8> full_crop = try vision.crop(pixels, top = 0, left = 0, height = 2, width = 3)

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
            if full_crop[0, y, x].item() != expected
                differences += 1
    print(differences)
    return 0

auto result = run()
match result
    int
        int ignored = result
    error problem
        print(problem)
QUI

properties_output="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/properties.qui")"
if [[ "$properties_output" != "0" ]]; then
    echo "unexpected vision property output: $properties_output" >&2
    exit 1
fi

echo "vision boundary tests: ok"
