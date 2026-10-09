#!/usr/bin/env bash
set -euo pipefail

QUIDRA="$1"
REPOSITORY_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PACKAGE_ROOT="$(dirname "$REPOSITORY_ROOT"):$REPOSITORY_ROOT"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

cat > "$TMP/vision-dtypes.qui" <<'QUI'
import vision

int | error run()
    // The documented grayscale coefficients must be used directly, not the old
    // 77/150/29 over 256 integer approximation.
    tensor<nat8> rgb = tensor.zeros<nat8>([3, 1, 2])
    rgb[1, 0, 0] = nat8(255)
    rgb[2, 0, 1] = nat8(255)
    tensor<nat8> gray = try vision.grayscale(rgb)
    print(gray[0, 0, 0].item())
    print(NL)
    print(gray[0, 0, 1].item())
    print(NL)

    // Geometry only relocates samples, so it must preserve the source dtype.
    tensor<nat16> pixels = tensor.zeros<nat16>([1, 2, 2])
    pixels[0, 0, 0] = nat16(1000)
    pixels[0, 0, 1] = nat16(2000)
    pixels[0, 1, 0] = nat16(3000)
    pixels[0, 1, 1] = nat16(4000)

    tensor<nat16> cropped = try vision.crop(pixels, top = 0, left = 1, height = 2, width = 1)
    tensor<nat16> resized = try vision.resize(pixels, height = 4, width = 4)
    tensor<nat16> flipped = try vision.flip_horizontal(pixels)
    tensor<nat16> turned = try vision.rotate90(pixels)

    print(cropped[0, 1, 0].item())
    print(NL)
    print(resized[0, 3, 3].item())
    print(NL)
    print(flipped[0, 0, 0].item())
    print(NL)
    print(turned[0, 0, 1].item())
    print(NL)

    // Morphology must not retain nat8-specific sentinels such as 0 or 255.
    tensor<int16> signed_pixels = tensor.zeros<int16>([1, 1, 3])
    signed_pixels[0, 0, 0] = int16(-10)
    signed_pixels[0, 0, 1] = int16(-5)
    signed_pixels[0, 0, 2] = int16(-20)
    tensor<int16> signed_dilated = try vision.dilate(signed_pixels, radius = 1)
    tensor<int16> signed_eroded = try vision.erode(signed_pixels, radius = 1)
    print(signed_dilated[0, 0, 0].item())
    print(NL)
    print(signed_eroded[0, 0, 1].item())
    print(NL)

    tensor<nat16> bright = tensor.zeros<nat16>([1, 1, 2])
    bright[0, 0, 0] = nat16(1000)
    bright[0, 0, 1] = nat16(2000)
    tensor<nat16> bright_dilated = try vision.dilate(bright, radius = 1)
    tensor<nat16> bright_eroded = try vision.erode(bright, radius = 1)
    print(bright_dilated[0, 0, 0].item())
    print(NL)
    print(bright_eroded[0, 0, 1].item())
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

output="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/vision-dtypes.qui")"
expected="$(printf '150\n29\n4000\n4000\n2000\n1000\n-5\n-20\n2000\n1000')"
if [[ "$output" != "$expected" ]]; then
    echo "unexpected vision dtype output:" >&2
    printf '%s\n' "$output" >&2
    exit 1
fi

cat > "$TMP/downsample-dtypes.qui" <<'QUI'
import vision

int | error run()
    // Block means keep the element type of the source.
    tensor<nat8> bytes = tensor.zeros<nat8>([1, 2, 2])
    bytes[0, 0, 0] = nat8(255)
    bytes[0, 0, 1] = nat8(255)
    bytes[0, 1, 0] = nat8(255)
    bytes[0, 1, 1] = nat8(254)
    tensor<nat8> small_bytes = try vision.downsample_mean(bytes, 2)
    print(small_bytes[0, 0, 0].item())
    print(NL)

    tensor<real32> single = tensor.zeros<real32>([1, 2, 2])
    single[0, 0, 0] = real32(1)
    single[0, 0, 1] = real32(2)
    single[0, 1, 0] = real32(3)
    single[0, 1, 1] = real32(4.5)
    tensor<real32> small_single = try vision.downsample_mean(single, 2)
    print(small_single[0, 0, 0].item() == real32(2.625))
    print(NL)

    tensor<real64> double = tensor.zeros<real64>([1, 1, 3, 3])
    double[0, 0, 0, 0] = 0.5
    double[0, 0, 0, 1] = 1.5
    double[0, 0, 1, 0] = 2.5
    double[0, 0, 1, 1] = 4.0
    double[0, 0, 2, 2] = 100.0
    tensor<real64> small_double = try vision.downsample_mean(double, 2)
    print(small_double[0, 0, 0, 0].item() == 2.125)
    print(NL)

    // The CPU kernel keeps subnormal real32 values: four equal subnormals
    // average to themselves exactly, and 1, 1, 2, 3 times the smallest
    // subnormal sum to 7 of them, which rounds to 2 after the division by 4.
    real32 tiny = real32(1.0e-39)
    real32 smallest = real32(1.0e-45)
    tensor<real32> subnormal = tensor.ones<real32>([1, 2, 4]) * tiny
    subnormal[0, 0, 2] = smallest
    subnormal[0, 0, 3] = smallest
    subnormal[0, 1, 2] = smallest + smallest
    subnormal[0, 1, 3] = smallest + smallest + smallest
    tensor<real32> small_subnormal = try vision.downsample_mean(subnormal, 2)
    print(
        tiny > real32(0.0) and small_subnormal[0, 0, 0].item() == tiny
        and small_subnormal[0, 0, 1].item() == smallest + smallest
    )
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

downsample_output="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/downsample-dtypes.qui")"
downsample_expected="$(printf '254\ntrue\ntrue\ntrue')"
if [[ "$downsample_output" != "$downsample_expected" ]]; then
    echo "unexpected vision downsample_mean dtype output:" >&2
    printf '%s\n' "$downsample_output" >&2
    exit 1
fi

echo "vision dtype integration: ok"
