#!/usr/bin/env bash
set -euo pipefail

QUIDRA="$1"
REPOSITORY_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PACKAGE_ROOT="$(dirname "$REPOSITORY_ROOT"):$REPOSITORY_ROOT"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
export QUIDRA_CACHE_DIR="$TMP/run-cache"


cat > "$TMP/image-io.qui" <<QUI
import vision

int | error run()
    tensor<nat8> pixels = tensor.zeros<nat8>([3, 2, 2])
    pixels[0, 0, 0] = nat8(10)
    pixels[1, 0, 0] = nat8(20)
    pixels[2, 0, 0] = nat8(30)
    pixels[0, 1, 1] = nat8(200)
    try vision.write("$TMP/roundtrip.bmp", pixels)
    tensor<nat8> decoded = try vision.read<nat8>("$TMP/roundtrip.bmp", channels = 3)
    print(decoded.shape()[0] == 3)
    print(NL)
    print(decoded.shape()[1] == 2)
    print(NL)
    print(decoded.shape()[2] == 2)
    print(NL)
    print(decoded[0, 0, 0].item() == nat8(10))
    print(NL)
    print(decoded[1, 0, 0].item() == nat8(20))
    print(NL)
    print(decoded[2, 0, 0].item() == nat8(30))
    print(NL)
    print(decoded[0, 1, 1].item() == nat8(200))
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
    tensor<nat8> pixels = tensor.zeros<nat8>([3, 2, 3])
    int value = 0
    for channel in range(3)
        for y in range(2)
            for x in range(3)
                pixels[channel, y, x] = nat8(value)
                value += 1

    tensor<nat8> cropped = try vision.crop(pixels, top = 0, left = 1, height = 2, width = 2)
    tensor<nat8> resized = try vision.resize(pixels, height = 4, width = 6)
    tensor<nat8> flipped = try vision.flip_horizontal(pixels)
    tensor<nat8> rotated = try vision.rotate90(pixels)
    tensor<nat8> gray = try vision.grayscale(pixels)
    tensor<nat8> binary = try vision.threshold(pixels, cutoff = nat8(8))

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
    tensor<nat8> impulse = tensor.zeros<nat8>([1, 3, 3])
    impulse[0, 1, 1] = nat8(255)
    tensor<nat8> blurred = try vision.blur(impulse, radius = 1)
    tensor<nat8> expanded = try vision.dilate(impulse, radius = 1)
    tensor<nat8> contracted = try vision.erode(impulse, radius = 1)
    tensor<int64> kernel = tensor.zeros<int>([3, 3])
    kernel[1, 1] = 1
    tensor<nat8> filtered = try vision.filter(impulse, kernel)

    tensor<nat8> line = tensor.zeros<nat8>([1, 1, 3])
    line[0, 0, 0] = nat8(10)
    line[0, 0, 1] = nat8(20)
    line[0, 0, 2] = nat8(30)
    tensor<int64> directional_kernel = tensor.zeros<int>([1, 2])
    directional_kernel[0, 0] = 1
    tensor<nat8> directional = try vision.filter(line, directional_kernel)

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
    tensor<nat8> pixels = tensor.zeros<nat8>([1, 2, 3])
    int value = 1
    for y in range(2)
        for x in range(3)
            pixels[0, y, x] = nat8(value)
            value += 1

    tensor<nat8> turned90 = try vision.rotate90(pixels)
    tensor<nat8> turned180 = try vision.rotate180(pixels)
    tensor<nat8> turned270 = try vision.rotate270(pixels)
    tensor<nat8> mirrored = try vision.flip_vertical(pixels)

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

    tensor<nat8> after270 = try vision.rotate270(pixels)
    tensor<nat8> full_turn = try vision.rotate90(after270)
    tensor<nat8> first90 = try vision.rotate90(pixels)
    tensor<nat8> half_turn = try vision.rotate90(first90)
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
    tensor<nat8> rgb = tensor.zeros<nat8>([2, 3, 1, 1])
    rgb[0, 0, 0, 0] = nat8(100)
    rgb[0, 1, 0, 0] = nat8(150)
    rgb[0, 2, 0, 0] = nat8(200)
    rgb[1, 0, 0, 0] = nat8(255)
    tensor<nat8> gray = try vision.grayscale(rgb)
    int[] gray_shape = gray.shape()
    print(len(gray_shape) == 4 and gray_shape[0] == 2 and gray_shape[1] == 1)
    print(NL)
    print(gray[0, 0, 0, 0].item() == nat8(141))
    print(NL)
    print(gray[1, 0, 0, 0].item() == nat8(76))
    print(NL)

    tensor<nat8> stack = tensor.zeros<nat8>([1, 2, 1, 2, 2])
    stack[0, 0, 0, 0, 0] = nat8(10)
    stack[0, 0, 0, 0, 1] = nat8(200)
    stack[0, 1, 0, 1, 1] = nat8(250)
    tensor<nat8> thresholded = try vision.threshold(
        stack, nat8(128), low = nat8(3), high = nat8(9)
    )
    print(thresholded.shape()[1] == 2)
    print(NL)
    print(thresholded[0, 0, 0, 0, 0].item() == nat8(3))
    print(NL)
    print(thresholded[0, 0, 0, 0, 1].item() == nat8(9))
    print(NL)
    print(thresholded[0, 1, 0, 1, 1].item() == nat8(9))
    print(NL)

    tensor<real32> floating = tensor.zeros<real32>([1, 1, 1, 3])
    floating[0, 0, 0, 0] = real32(0.25)
    floating[0, 0, 0, 1] = real32(0.5)
    floating[0, 0, 0, 2] = real32(0.75)
    tensor<real32> floating_thresholded = try vision.threshold(
        floating, real32(0.5), low = real32(-1.0), high = real32(2.0)
    )
    print(floating_thresholded[0, 0, 0, 0].item() == real32(-1.0))
    print(NL)
    print(floating_thresholded[0, 0, 0, 1].item() == real32(2.0))
    print(NL)
    print(floating_thresholded[0, 0, 0, 2].item() == real32(2.0))
    print(NL)

    tensor<nat8> images = tensor.zeros<nat8>([2, 1, 3, 3])
    images[0, 0, 1, 1] = nat8(255)
    images[1, 0, 0, 0] = nat8(90)
    tensor<nat8> blurred = try vision.blur(images, radius = 1)
    print(blurred.shape()[0] == 2 and blurred.shape()[1] == 1)
    print(NL)
    print(blurred[0, 0, 1, 1].item() == nat8(28))
    print(NL)
    print(blurred[1, 0, 2, 2].item() == nat8(0))
    print(NL)

    tensor<int64> identity = tensor.zeros<int>([1, 1])
    identity[0, 0] = 1
    tensor<nat8> filtered = try vision.filter(images, identity)
    print(filtered[0, 0, 1, 1].item() == nat8(255))
    print(NL)
    print(filtered[1, 0, 0, 0].item() == nat8(90))
    print(NL)

    tensor<nat8> expanded = try vision.dilate(stack, radius = 1)
    tensor<nat8> contracted = try vision.erode(stack, radius = 1)
    print(expanded.shape()[0] == 1 and expanded.shape()[1] == 2)
    print(NL)
    print(expanded[0, 0, 0, 1, 0].item() == nat8(200))
    print(NL)
    print(contracted[0, 0, 0, 0, 1].item() == nat8(0))
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

cat > "$TMP/downsample.qui" <<'QUI'
import vision

int len_all(int[] shape)
    int count = 1
    for extent in shape
        count = count * extent
    return count

// Reference block mean in the documented order: each block row left to right,
// then the row sums top to bottom, divided once by factor * factor.
real32 reference_mean(tensor<real32> pixels, int plane, int oy, int ox, int factor)
    int[] shape = pixels.shape()
    int height = shape[len(shape) - 2]
    int width = shape[len(shape) - 1]
    tensor<real32> flat = pixels.reshape([len_all(shape)])
    real32 total = 0.0
    for dy in range(factor)
        int base = (plane * height + oy * factor + dy) * width + ox * factor
        real32 row_sum = flat[base].item()
        for dx in range(1, factor)
            row_sum = row_sum + flat[base + dx].item()
        if dy == 0
            total = row_sum
        else
            total = total + row_sum
    return total / real32(factor * factor)

// Counts output samples that differ from the reference; `pixels` has any
// rank >= 3 and the result keeps every leading dimension.
int mismatches(tensor<real32> pixels, tensor<real32> reduced, int factor)
    int[] shape = pixels.shape()
    int[] reduced_shape = reduced.shape()
    int rank = len(shape)
    int planes = 1
    for axis in range(rank - 2)
        if reduced_shape[axis] != shape[axis]
            return -1
        planes = planes * shape[axis]
    int output_height = reduced_shape[rank - 2]
    int output_width = reduced_shape[rank - 1]
    if output_height != shape[rank - 2] / factor or output_width != shape[rank - 1] / factor
        return -1
    tensor<real32> flat = reduced.reshape([planes * output_height * output_width])
    int count = 0
    for plane in range(planes)
        for oy in range(output_height)
            for ox in range(output_width)
                real32 actual = flat[(plane * output_height + oy) * output_width + ox].item()
                if actual != reference_mean(pixels, plane, oy, ox, factor)
                    count += 1
    return count

tensor<real32> pattern(int[] shape)
    int count = len_all(shape)
    tensor<real32> flat = tensor.zeros<real32>([count])
    for index in range(count)
        flat[index] = real32((index * 37 + index / 7) % 23) * real32(0.37) - real32(1.1)
    return flat.reshape(shape)

int | error run()
    // Multiple channels, a leading batch axis, and sizes that leave trailing
    // rows and columns outside every block.
    tensor<real32> image = pattern([3, 11, 14])
    tensor<real32> batch = pattern([2, 2, 9, 10])
    int[] factors = [2, 3, 4]
    int total_mismatches = 0
    for factor in factors
        total_mismatches += mismatches(image, try vision.downsample_mean(image, factor), factor)
        total_mismatches += mismatches(batch, try vision.downsample_mean(batch, factor), factor)
    print(total_mismatches)
    print(NL)

    tensor<real32> reduced = try vision.downsample_mean(image, 3)
    int[] reduced_shape = reduced.shape()
    print(len(reduced_shape) == 3 and reduced_shape[0] == 3 and reduced_shape[1] == 3 and reduced_shape[2] == 4)
    print(NL)

    // Samples outside every complete block never contribute.
    tensor<real32> edited = image.reshape([3 * 11 * 14])
    for c in range(3)
        for y in range(11)
            edited[(c * 11 + y) * 14 + 13] = real32(1000)
        for x in range(14)
            edited[(c * 11 + 10) * 14 + x] = real32(-1000)
    tensor<real32> edited_reduced = try vision.downsample_mean(edited.reshape([3, 11, 14]), 3)
    print(mismatches(image, edited_reduced, 3))
    print(NL)

    // factor 1 keeps every sample; a factor equal to the extent averages the
    // whole axis.
    tensor<real32> same = try vision.downsample_mean(image, 1)
    print(mismatches(image, same, 1))
    print(NL)
    tensor<real32> column = try vision.downsample_mean(pattern([1, 4, 9]), 4)
    print(column.shape()[1] == 1 and column.shape()[2] == 2)
    print(NL)

    // Untracked non-contiguous views give the same result as their
    // contiguous copy.
    tensor<real32> transposed = pattern([3, 14, 11]).transpose(1, 2)
    tensor<real32> from_view = try vision.downsample_mean(transposed, 3)
    tensor<real32> from_copy = try vision.downsample_mean(transposed.contiguous(), 3)
    print(transposed.is_contiguous() == false and mismatches(transposed.contiguous(), from_view, 3) == 0 and mismatches(transposed.contiguous(), from_copy, 3) == 0)
    print(NL)

    // A shrink built from gathers: the mean of factor x factor gathers over
    // the top-left blocks of an 8-bit image equals downsample_mean followed by
    // a crop to those blocks.
    int factor = 4
    int height = 8
    int width = 12
    tensor<real32> sem = tensor.zeros<real32>([1, 35, 50])
    for y in range(35)
        for x in range(50)
            sem[0, y, x] = real32((y * 50 + x) * 97 % 256)
    tensor<real32> gathered = tensor.zeros([1, height, width])
    int[] indices = array(height * width, fill = 0)
    for dy in range(factor)
        for dx in range(factor)
            for index in range(height * width)
                indices[index] = ((index / width) * factor + dy) * 50 + (index % width) * factor + dx
            gathered = gathered + sem.gather(indices, [1, height, width])
    gathered = gathered / real32(factor * factor)
    tensor<real32> blocks = try vision.downsample_mean(sem, factor)
    tensor<real32> shrunk = try vision.crop(blocks, 0, 0, height, width)
    int shrink_differences = 0
    for y in range(height)
        for x in range(width)
            if shrunk[0, y, x].item() != gathered[0, y, x].item()
                shrink_differences += 1
    print(shrink_differences)
    print(NL)

    // nat8 sums exactly and truncates the quotient, like the nat8 blur.
    tensor<nat8> bytes = tensor.zeros<nat8>([1, 2, 5])
    bytes[0, 0, 0] = nat8(1)
    bytes[0, 0, 1] = nat8(2)
    bytes[0, 1, 0] = nat8(2)
    bytes[0, 1, 1] = nat8(2)
    bytes[0, 0, 2] = nat8(255)
    bytes[0, 0, 3] = nat8(255)
    bytes[0, 1, 2] = nat8(255)
    bytes[0, 1, 3] = nat8(254)
    bytes[0, 0, 4] = nat8(200)
    tensor<nat8> small = try vision.downsample_mean(bytes, 2)
    print(small.shape()[1] == 1 and small.shape()[2] == 2)
    print(NL)
    print(small[0, 0, 0].item())
    print(NL)
    print(small[0, 0, 1].item())
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

downsample_output="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/downsample.qui")"
downsample_expected="$(printf '0\ntrue\n0\n0\ntrue\ntrue\n0\ntrue\n1\n254')"
if [[ "$downsample_output" != "$downsample_expected" ]]; then
    echo "unexpected Vision downsample_mean output:" >&2
    printf '%s\n' "$downsample_output" >&2
    exit 1
fi

echo "vision integration: ok"
