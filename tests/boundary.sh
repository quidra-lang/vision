#!/usr/bin/env bash
set -euo pipefail

QUIDRA="$1"
REPOSITORY_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PACKAGE_ROOT="$(dirname "$REPOSITORY_ROOT"):$REPOSITORY_ROOT"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
export QUIDRA_CACHE_DIR="$TMP/run-cache"

cat > "$TMP/boundary.qui" <<'QUI'
import vision

tensor<nat8> pixels = tensor.zeros<nat8>([3, 2, 3])

bool resize_rejected = false
tensor<nat8> | error bad_resize = vision.resize(pixels, height = 0, width = 2)
match bad_resize
    tensor<nat8>
        resize_rejected = false
    error
        resize_rejected = true
print(resize_rejected)
print(NL)

bool crop_rejected = false
tensor<nat8> | error bad_crop = vision.crop(pixels, top = 1, left = 2, height = 2, width = 2)
match bad_crop
    tensor<nat8>
        crop_rejected = false
    error
        crop_rejected = true
print(crop_rejected)
print(NL)

bool grayscale_rejected = false
tensor<nat8> two_channels = tensor.zeros<nat8>([2, 2, 2])
tensor<nat8> | error bad_grayscale = vision.grayscale(two_channels)
match bad_grayscale
    tensor<nat8>
        grayscale_rejected = false
    error
        grayscale_rejected = true
print(grayscale_rejected)
print(NL)

bool blur_rejected = false
tensor<nat8> | error bad_blur = vision.blur(pixels, radius = -1)
match bad_blur
    tensor<nat8>
        blur_rejected = false
    error
        blur_rejected = true
print(blur_rejected)
print(NL)

bool filter_rejected = false
tensor<int64> kernel = tensor.ones<int>([3, 3])
tensor<nat8> | error bad_filter = vision.filter(pixels, kernel, divisor = 0)
match bad_filter
    tensor<nat8>
        filter_rejected = false
    error
        filter_rejected = true
print(filter_rejected)
print(NL)

bool dilate_rejected = false
tensor<nat8> | error bad_dilate = vision.dilate(pixels, radius = -1)
match bad_dilate
    tensor<nat8>
        dilate_rejected = false
    error
        dilate_rejected = true
print(dilate_rejected)
print(NL)

bool erode_rejected = false
tensor<nat8> | error bad_erode = vision.erode(pixels, radius = -1)
match bad_erode
    tensor<nat8>
        erode_rejected = false
    error
        erode_rejected = true
print(erode_rejected)
print(NL)

bool rank_rejected = false
tensor<nat8> rank_two = tensor.zeros<nat8>([2, 3])
tensor<nat8> | error bad_flip = vision.flip_horizontal(rank_two)
match bad_flip
    tensor<nat8>
        rank_rejected = false
    error
        rank_rejected = true
print(rank_rejected)
print(NL)
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
    tensor<nat8> pixels = tensor.zeros<nat8>([1, 2, 3])
    int value = 1
    for y in range(2)
        for x in range(3)
            pixels[0, y, x] = nat8(value)
            value += 1

    tensor<nat8> flipped_once = try vision.flip_horizontal(pixels)
    tensor<nat8> flipped_twice = try vision.flip_horizontal(flipped_once)
    tensor<nat8> r1 = try vision.rotate90(pixels)
    tensor<nat8> r2 = try vision.rotate90(r1)
    tensor<nat8> r3 = try vision.rotate90(r2)
    tensor<nat8> rotated_four = try vision.rotate90(r3)
    tensor<nat8> resized_same = try vision.resize(pixels, height = 2, width = 3)
    tensor<nat8> dilated_zero = try vision.dilate(pixels, radius = 0)
    tensor<nat8> eroded_zero = try vision.erode(pixels, radius = 0)
    tensor<nat8> full_crop = try vision.crop(pixels, top = 0, left = 0, height = 2, width = 3)

    int differences = 0
    for y in range(2)
        for x in range(3)
            nat8 expected = pixels[0, y, x].item()
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
    print(NL)
    return 0

auto | error result = run()
match result
    int
        int ignored = result
    error problem
        print(problem)
        print(NL)
QUI

properties_output="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/properties.qui")"
if [[ "$properties_output" != "0" ]]; then
    echo "unexpected vision property output: $properties_output" >&2
    exit 1
fi

cat > "$TMP/downsample-boundary.qui" <<'QUI'
import vision

bool rejects_float(tensor<real32> pixels, int factor)
    tensor<real32> | error result = vision.downsample_mean(pixels, factor)
    match result
        tensor<real32>
            return false
        error
            return true

bool rejects_bytes(tensor<nat8> pixels, int factor)
    tensor<nat8> | error result = vision.downsample_mean(pixels, factor)
    match result
        tensor<nat8>
            return false
        error
            return true

tensor<real32> image = tensor.zeros<real32>([1, 4, 6])
tensor<nat8> bytes = tensor.zeros<nat8>([1, 4, 6])
print(rejects_float(image, 0))
print(NL)
print(rejects_float(image, -2))
print(NL)
print(rejects_float(image, 5))
print(NL)
print(rejects_float(tensor.zeros<real32>([1, 6, 4]), 5))
print(NL)
print(rejects_float(tensor.zeros<real32>([4, 6]), 2))
print(NL)
print(rejects_float(tensor.zeros<real32>([1, 0, 6]), 1))
print(NL)
print(rejects_bytes(bytes, 0))
print(NL)
print(rejects_bytes(bytes, 7))
print(NL)
print(rejects_bytes(tensor.zeros<nat8>([4, 6]), 2))
print(NL)
print(rejects_float(image, 4))
print(NL)
print(rejects_bytes(bytes, 1))
print(NL)
QUI

downsample_output="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/downsample-boundary.qui")"
downsample_expected="$(printf 'true\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\nfalse\nfalse')"
if [[ "$downsample_output" != "$downsample_expected" ]]; then
    echo "unexpected vision downsample_mean boundary output:" >&2
    printf '%s\n' "$downsample_output" >&2
    exit 1
fi

# vision.downsample_mean falls back to the portable composition only where no
# Vision kernel can run; every other native status becomes an error. The policy
# lives in the package's internal module, reached through a copy.
cp "$REPOSITORY_ROOT/internal.qui" "$TMP/vision_internal.qui"
cat > "$TMP/downsample-status.qui" <<'QUI'
import composition = "./vision_internal.qui"

int[] statuses = [1, 2, 3, 4, 5, 6, 7, 8, 9]
for status in statuses
    if composition.block_native_unavailable(int32(status))
        print("{status} portable")
    else
        print("{status} error: {composition.block_native_problem(int32(status))}")
    print(NL)
QUI

status_output="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/downsample-status.qui")"
status_expected="1 error: vision.downsample_mean native kernel rejected its arguments (status 1)
2 portable
3 error: vision.downsample_mean native kernel rejected its arguments (status 3)
4 error: vision.downsample_mean native kernel rejected its arguments (status 4)
5 portable
6 portable
7 error: vision.downsample_mean device kernel failed
8 error: vision.downsample_mean could not attach its autograd node
9 error: vision.downsample_mean does not support float64 tensors on Metal, which has no float64 arithmetic; convert to real32 or move the tensor to the CPU explicitly"
if [[ "$status_output" != "$status_expected" ]]; then
    echo "unexpected vision downsample_mean native status policy:" >&2
    printf '%s\n' "$status_output" >&2
    exit 1
fi

# Package-defined errors expose stable codes independently of their text.
cat > "$TMP/vision-codes.qui" <<'QUI'
import vision

tensor<nat8> pixels = tensor.zeros<nat8>([3, 2, 2])
tensor<nat8> | error bad_shape = vision.resize(pixels, height = 0, width = 2)
match bad_shape
    tensor<nat8>
        print(false)
    error problem
        print(problem.code == "VISION_ARGUMENT")
print(NL)

tensor<nat8> | error bad_radius = vision.dilate(pixels, radius = -1)
match bad_radius
    tensor<nat8>
        print(false)
    error problem
        print(problem.code == "VISION_ARGUMENT")
print(NL)

tensor<real32> floating = tensor.ones<real32>([3, 2, 2]).track()
tensor<real32> | error bad_tracked = vision.threshold(floating, cutoff = real32(0.5))
match bad_tracked
    tensor<real32>
        print(false)
    error problem
        print(problem.code == "VISION_TRACKED")
print(NL)
QUI
codes_output="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/vision-codes.qui")"
if [[ "$codes_output" != "$(printf 'true\ntrue\ntrue')" ]]; then
    printf 'vision code mismatch: %s\n' "$codes_output" >&2
    exit 1
fi

echo "vision boundary tests: ok"
