#!/usr/bin/env bash
set -euo pipefail

QUIDRA="$1"
REPOSITORY_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PACKAGE_ROOT="$(dirname "$REPOSITORY_ROOT"):$REPOSITORY_ROOT"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT


cat > "$TMP/image-io.qui" <<QUI
import vision

int | error run()
    tensor<uint8> pixels = tensor.zeros<uint8>([3, 2, 2])
    pixels[0, 0, 0] = uint8(10)
    pixels[1, 0, 0] = uint8(20)
    pixels[2, 0, 0] = uint8(30)
    pixels[0, 1, 1] = uint8(200)
    try vision.write("$TMP/roundtrip.bmp", pixels)
    tensor<uint8> decoded = try vision.read<uint8>("$TMP/roundtrip.bmp", channels = 3)
    print(decoded.shape()[0] == 3)
    print(NL)
    print(decoded.shape()[1] == 2)
    print(NL)
    print(decoded.shape()[2] == 2)
    print(NL)
    print(decoded[0, 0, 0].item() == uint8(10))
    print(NL)
    print(decoded[1, 0, 0].item() == uint8(20))
    print(NL)
    print(decoded[2, 0, 0].item() == uint8(30))
    print(NL)
    print(decoded[0, 1, 1].item() == uint8(200))
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

image_io_output="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/image-io.qui")"
image_io_expected="$(printf 'true\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue')"
if [[ "$image_io_output" != "$image_io_expected" ]]; then
    echo "unexpected Vision image I/O output:" >&2
    printf '%s\n' "$image_io_output" >&2
    exit 1
fi

cat > "$TMP/use-vision.qui" <<'QUI'
import vision

int | error run()
    tensor<uint8> pixels = tensor.zeros<uint8>([3, 2, 3])
    int value = 0
    for channel in range(3)
        for y in range(2)
            for x in range(3)
                pixels[channel, y, x] = uint8(value)
                value += 1

    tensor<uint8> cropped = try vision.crop(pixels, top = 0, left = 1, height = 2, width = 2)
    tensor<uint8> resized = try vision.resize(pixels, height = 4, width = 6)
    tensor<uint8> flipped = try vision.flip_horizontal(pixels)
    tensor<uint8> rotated = try vision.rotate90(pixels)
    tensor<uint8> gray = try vision.grayscale(pixels)
    tensor<uint8> binary = try vision.threshold(pixels, cutoff = uint8(8))

    print(cropped.shape()[2])
    print(NL)
    print(cropped[0, 0, 0].item())
    print(NL)
    print(resized.shape()[1])
    print(NL)
    print(resized.shape()[2])
    print(NL)
    print(flipped[0, 0, 0].item())
    print(NL)
    print(rotated.shape()[1])
    print(NL)
    print(rotated.shape()[2])
    print(NL)
    print(gray.shape()[0])
    print(NL)
    print(binary[0, 0, 0].item())
    print(NL)
    print(binary[2, 1, 2].item())
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

output="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/use-vision.qui")"
expected="$(printf '2\n1\n4\n6\n2\n3\n2\n1\n0\n255')"
if [[ "$output" != "$expected" ]]; then
    echo "unexpected vision output: $output" >&2
    exit 1
fi

cat > "$TMP/filters.qui" <<'QUI'
import vision

int | error run()
    tensor<uint8> impulse = tensor.zeros<uint8>([1, 3, 3])
    impulse[0, 1, 1] = uint8(255)
    tensor<uint8> blurred = try vision.blur(impulse, radius = 1)
    tensor<uint8> expanded = try vision.dilate(impulse, radius = 1)
    tensor<uint8> contracted = try vision.erode(impulse, radius = 1)
    tensor<int> kernel = tensor.zeros<int>([3, 3])
    kernel[1, 1] = 1
    tensor<uint8> filtered = try vision.filter(impulse, kernel)

    tensor<uint8> line = tensor.zeros<uint8>([1, 1, 3])
    line[0, 0, 0] = uint8(10)
    line[0, 0, 1] = uint8(20)
    line[0, 0, 2] = uint8(30)
    tensor<int> directional_kernel = tensor.zeros<int>([1, 2])
    directional_kernel[0, 0] = 1
    tensor<uint8> directional = try vision.filter(line, directional_kernel)

    print(blurred[0, 1, 1].item())
    print(NL)
    print(blurred[0, 0, 0].item())
    print(NL)
    print(expanded[0, 0, 0].item())
    print(NL)
    print(contracted[0, 1, 1].item())
    print(NL)
    print(filtered[0, 1, 1].item())
    print(NL)
    print(directional[0, 0, 0].item())
    print(NL)
    print(directional[0, 0, 2].item())
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

filters_output="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/filters.qui")"
filters_expected="$(printf '28\n63\n255\n0\n255\n0\n20')"
if [[ "$filters_output" != "$filters_expected" ]]; then
    echo "unexpected vision filter output: $filters_output" >&2
    exit 1
fi

cat > "$TMP/rotations.qui" <<'QUI'
import vision

int | error run()
    tensor<uint8> pixels = tensor.zeros<uint8>([1, 2, 3])
    int value = 1
    for y in range(2)
        for x in range(3)
            pixels[0, y, x] = uint8(value)
            value += 1

    tensor<uint8> turned90 = try vision.rotate90(pixels)
    tensor<uint8> turned180 = try vision.rotate180(pixels)
    tensor<uint8> turned270 = try vision.rotate270(pixels)
    tensor<uint8> mirrored = try vision.flip_vertical(pixels)

    print(turned90[0, 0, 0].item())
    print(NL)
    print(turned90[0, 2, 1].item())
    print(NL)
    print(turned180[0, 0, 0].item())
    print(NL)
    print(turned180[0, 1, 2].item())
    print(NL)
    print(turned270[0, 0, 0].item())
    print(NL)
    print(turned270[0, 2, 1].item())
    print(NL)
    print(mirrored[0, 0, 0].item())
    print(NL)

    tensor<uint8> after270 = try vision.rotate270(pixels)
    tensor<uint8> full_turn = try vision.rotate90(after270)
    tensor<uint8> first90 = try vision.rotate90(pixels)
    tensor<uint8> half_turn = try vision.rotate90(first90)
    int differences = 0
    for y in range(2)
        for x in range(3)
            if full_turn[0, y, x].item() != pixels[0, y, x].item()
                differences += 1
            if half_turn[0, y, x].item() != turned180[0, y, x].item()
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

rotations_output="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/rotations.qui")"
rotations_expected="$(printf '4\n3\n6\n1\n3\n4\n4\n0')"
if [[ "$rotations_output" != "$rotations_expected" ]]; then
    echo "unexpected vision rotation output: $rotations_output" >&2
    exit 1
fi

cat > "$TMP/leading-dimensions.qui" <<'QUI'
import vision

int | error run()
    tensor<uint8> rgb = tensor.zeros<uint8>([2, 3, 1, 1])
    rgb[0, 0, 0, 0] = uint8(100)
    rgb[0, 1, 0, 0] = uint8(150)
    rgb[0, 2, 0, 0] = uint8(200)
    rgb[1, 0, 0, 0] = uint8(255)
    tensor<uint8> gray = try vision.grayscale(rgb)
    int[] gray_shape = gray.shape()
    print(len(gray_shape) == 4 and gray_shape[0] == 2 and gray_shape[1] == 1)
    print(NL)
    print(gray[0, 0, 0, 0].item() == uint8(141))
    print(NL)
    print(gray[1, 0, 0, 0].item() == uint8(76))
    print(NL)

    tensor<uint8> stack = tensor.zeros<uint8>([1, 2, 1, 2, 2])
    stack[0, 0, 0, 0, 0] = uint8(10)
    stack[0, 0, 0, 0, 1] = uint8(200)
    stack[0, 1, 0, 1, 1] = uint8(250)
    tensor<uint8> thresholded = try vision.threshold(
        stack, uint8(128), low = uint8(3), high = uint8(9)
    )
    print(thresholded.shape()[1] == 2)
    print(NL)
    print(thresholded[0, 0, 0, 0, 0].item() == uint8(3))
    print(NL)
    print(thresholded[0, 0, 0, 0, 1].item() == uint8(9))
    print(NL)
    print(thresholded[0, 1, 0, 1, 1].item() == uint8(9))
    print(NL)

    tensor<float32> floating = tensor.zeros<float32>([1, 1, 1, 3])
    floating[0, 0, 0, 0] = float32(0.25)
    floating[0, 0, 0, 1] = float32(0.5)
    floating[0, 0, 0, 2] = float32(0.75)
    tensor<float32> floating_thresholded = try vision.threshold(
        floating, float32(0.5), low = float32(-1.0), high = float32(2.0)
    )
    print(floating_thresholded[0, 0, 0, 0].item() == float32(-1.0))
    print(NL)
    print(floating_thresholded[0, 0, 0, 1].item() == float32(2.0))
    print(NL)
    print(floating_thresholded[0, 0, 0, 2].item() == float32(2.0))
    print(NL)

    tensor<uint8> images = tensor.zeros<uint8>([2, 1, 3, 3])
    images[0, 0, 1, 1] = uint8(255)
    images[1, 0, 0, 0] = uint8(90)
    tensor<uint8> blurred = try vision.blur(images, radius = 1)
    print(blurred.shape()[0] == 2 and blurred.shape()[1] == 1)
    print(NL)
    print(blurred[0, 0, 1, 1].item() == uint8(28))
    print(NL)
    print(blurred[1, 0, 2, 2].item() == uint8(0))
    print(NL)

    tensor<int> identity = tensor.zeros<int>([1, 1])
    identity[0, 0] = 1
    tensor<uint8> filtered = try vision.filter(images, identity)
    print(filtered[0, 0, 1, 1].item() == uint8(255))
    print(NL)
    print(filtered[1, 0, 0, 0].item() == uint8(90))
    print(NL)

    tensor<uint8> expanded = try vision.dilate(stack, radius = 1)
    tensor<uint8> contracted = try vision.erode(stack, radius = 1)
    print(expanded.shape()[0] == 1 and expanded.shape()[1] == 2)
    print(NL)
    print(expanded[0, 0, 0, 1, 0].item() == uint8(200))
    print(NL)
    print(contracted[0, 0, 0, 0, 1].item() == uint8(0))
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

leading_output="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/leading-dimensions.qui")"
leading_expected="$(printf 'true\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue')"
if [[ "$leading_output" != "$leading_expected" ]]; then
    echo "unexpected Vision leading-dimension output:" >&2
    printf '%s\n' "$leading_output" >&2
    exit 1
fi

echo "vision integration: ok"
