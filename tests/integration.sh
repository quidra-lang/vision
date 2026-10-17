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
    for 0 <= nat channel < 3
        for 0 <= nat y < 2
            for 0 <= nat x < 3
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
    tensor<int64> kernel = tensor.zeros<int64>([3, 3])
    kernel[1, 1] = 1
    tensor<nat8> filtered = try vision.filter(impulse, kernel)

    tensor<nat8> line = tensor.zeros<nat8>([1, 1, 3])
    line[0, 0, 0] = nat8(10)
    line[0, 0, 1] = nat8(20)
    line[0, 0, 2] = nat8(30)
    tensor<int64> directional_kernel = tensor.zeros<int64>([1, 2])
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
    for 0 <= nat y < 2
        for 0 <= nat x < 3
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
    for 0 <= nat y < 2
        for 0 <= nat x < 3
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
    int[] gray_shape = int(gray.shape())
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

    tensor<int64> identity = tensor.zeros<int64>([1, 1])
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
    for auto extent = shape[]
        count = count * extent
    return count

// Reference block mean in the documented order: each block row left to right,
// then the row sums top to bottom, divided once by factor * factor.
real32 reference_mean(tensor<real32> pixels, int plane, int oy, int ox, int factor)
    int[] shape = int(pixels.shape())
    int height = shape[int(len(shape)) - 2]
    int width = shape[int(len(shape)) - 1]
    tensor<real32> flat = pixels.reshape([nat(len_all(shape))])
    real32 total = 0.0
    for 0 <= int dy < factor
        int base = (plane * height + oy * factor + dy) * width + ox * factor
        real32 row_sum = flat[base].item()
        for 1 <= int dx < factor
            row_sum = row_sum + flat[base + dx].item()
        if dy == 0
            total = row_sum
        else
            total = total + row_sum
    real32 divisor = real32(factor * factor)
    return total / divisor

// Counts output samples that differ from the reference; `pixels` has any
// rank >= 3 and the result keeps every leading dimension.
int mismatches(tensor<real32> pixels, tensor<real32> reduced, int factor)
    int[] shape = int(pixels.shape())
    int[] reduced_shape = int(reduced.shape())
    int rank = int(len(shape))
    int planes = 1
    for 0 <= int axis < rank - 2
        if reduced_shape[axis] != shape[axis]
            return -1
        planes = planes * shape[axis]
    int output_height = reduced_shape[rank - 2]
    int output_width = reduced_shape[rank - 1]
    if output_height != shape[rank - 2] / factor or output_width != shape[rank - 1] / factor
        return -1
    tensor<real32> flat = reduced.reshape([nat(planes * output_height * output_width)])
    int count = 0
    for 0 <= int plane < planes
        for 0 <= int oy < output_height
            for 0 <= int ox < output_width
                real32 actual = flat[(plane * output_height + oy) * output_width + ox].item()
                if actual != reference_mean(pixels, plane, oy, ox, factor)
                    count += 1
    return count

tensor<real32> pattern(int[] shape)
    int count = len_all(shape)
    tensor<real32> flat = tensor.zeros<real32>([nat(count)])
    for 0 <= int index < count
        real32 sample = real32((index * 37 + index / 7) % 23)
        flat[index] = sample * real32(0.37) - real32(1.1)
    return flat.reshape(nat(shape))

int | error run()
    // Multiple channels, a leading batch axis, and sizes that leave trailing
    // rows and columns outside every block.
    tensor<real32> image = pattern([3, 11, 14])
    tensor<real32> batch = pattern([2, 2, 9, 10])
    int[] factors = [2, 3, 4]
    int total_mismatches = 0
    for auto factor = factors[]
        total_mismatches += mismatches(image, try vision.downsample_mean(image, factor), factor)
        total_mismatches += mismatches(batch, try vision.downsample_mean(batch, factor), factor)
    print(total_mismatches)
    print(NL)

    tensor<real32> reduced = try vision.downsample_mean(image, 3)
    int[] reduced_shape = int(reduced.shape())
    print(len(reduced_shape) == 3 and reduced_shape[0] == 3 and reduced_shape[1] == 3 and reduced_shape[2] == 4)
    print(NL)

    // Samples outside every complete block never contribute.
    tensor<real32> edited = image.reshape([3 * 11 * 14])
    for 0 <= nat c < 3
        for 0 <= nat y < 11
            edited[(c * 11 + y) * 14 + 13] = real32(1000)
        for 0 <= nat x < 14
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
    for 0 <= nat y < 35
        for 0 <= nat x < 50
            real32 sample = real32((y * 50 + x) * 97 % 256)
            sem[0, y, x] = sample
    tensor<real32> gathered = tensor.zeros([1, nat(height), nat(width)])
    int[] indices = array(nat(height * width), fill = 0)
    for 0 <= int dy < factor
        for 0 <= int dx < factor
            for 0 <= int index < height * width
                indices[index] = ((index / width) * factor + dy) * 50 + (index % width) * factor + dx
            gathered = gathered + sem.gather(indices, [1, height, width])
    real32 divisor = real32(factor * factor)
    gathered = gathered / divisor
    tensor<real32> blocks = try vision.downsample_mean(sem, factor)
    tensor<real32> shrunk = try vision.crop(blocks, 0, 0, height, width)
    int shrink_differences = 0
    for 0 <= int y < height
        for 0 <= int x < width
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


# Vision's host kernels split their work with Core's qcore_parallel_for over
# independent rows of output samples, never over the terms of one block sum,
# so results do not depend on the thread count. A program whose block means
# (float32, float64, uint8), gradient spreads and second-order adjoints each
# span several parallel chunks prints the same values with one thread as
# with several.
cat > "$TMP/thread-count.qui" <<QUI
import vision
import math

tensor<real32> pattern32(int channels, int height, int width, int seed)
    tensor<real32> value = tensor.zeros<real32>([nat(channels), nat(height), nat(width)])
    for 0 <= int c < channels
        for 0 <= int y < height
            for 0 <= int x < width
                real32 sample = real32((c * 53 + y * 131 + x * 71 + seed * 17) % 97)
                value[c, y, x] = sample / real32(48) - real32(1)
    return value

tensor<real64> pattern64(int channels, int height, int width, int seed)
    tensor<real64> value = tensor.zeros<real64>([nat(channels), nat(height), nat(width)])
    for 0 <= int c < channels
        for 0 <= int y < height
            for 0 <= int x < width
                real64 sample = real64((c * 53 + y * 131 + x * 71 + seed * 17) % 97)
                value[c, y, x] = sample / 48.0 - 1.0
    return value

void sample32(string label, tensor<real32> value)
    tensor<real32> flat = value.untrack().cpu().contiguous()
    int count = 1
    for auto extent = flat.shape()[]
        count = count * int(extent)
    flat = flat.reshape([nat(count)])
    for 0 <= int index < count step 7
        print("{label} {index} {flat[index].item()}{NL}")

void sample64(string label, tensor<real64> value)
    tensor<real64> flat = value.untrack().cpu().contiguous()
    int count = 1
    for auto extent = flat.shape()[]
        count = count * int(extent)
    flat = flat.reshape([nat(count)])
    for 0 <= int index < count step 7
        print("{label} {index} {flat[index].item()}{NL}")

void sample8(string label, tensor<nat8> value)
    tensor<nat8> flat = value.cpu().contiguous()
    int count = 1
    for auto extent = flat.shape()[]
        count = count * int(extent)
    flat = flat.reshape([nat(count)])
    for 0 <= int index < count step 7
        print("{label} {index} {flat[index].item()}{NL}")

int | error run()
    // Factor 3 leaves a row and two columns outside every block, which the
    // spread fills with zeros.
    tensor<real32> source = pattern32(2, 64, 401, 1).track()
    tensor<real32> reduced = try vision.downsample_mean(source, 3)
    tensor<real32> weights = pattern32(2, 21, 133, 2)
    math.mean(reduced * weights * reduced).backward(&source, track = true)
    tensor<real32> first = source.grad
    sample32("mean32", reduced)
    sample32("grad32", first)
    math.mean(first * first).backward(&source)
    sample32("second32", source.grad)

    tensor<nat8> bytes = tensor.zeros<nat8>([2, 64, 401])
    for 0 <= nat c < 2
        for 0 <= nat y < 64
            for 0 <= nat x < 401
                bytes[c, y, x] = nat8((c * 89 + y * 37 + x * 11) % 256)
    sample8("mean8", try vision.downsample_mean(bytes, 3))

    // The uint8 filter kernel (CPU only) over rows of several chunks.
    tensor<int64> taps = tensor.zeros<int64>([3, 3])
    for 0 <= nat ky < 3
        for 0 <= nat kx < 3
            taps[ky, kx] = int64((int(ky) * 3 + int(kx)) % 4 + 1)
    tensor<nat8> filtered = try vision.filter(bytes, taps, divisor = 9, offset = 3)
    sample8("filter", filtered)
    return 0

int | error run64()
    tensor<real64> source = pattern64(2, 64, 401, 3).track()
    tensor<real64> reduced = try vision.downsample_mean(source, 3)
    math.mean(reduced * reduced).backward(&source)
    sample64("mean64", reduced)
    sample64("grad64", source.grad)
    return 0

auto | error result = run()
match result
    int
        int ignored = result
    error problem
        print(problem)
        print(NL)
auto | error result64 = run64()
match result64
    int
        int ignored = result64
    error problem
        print(problem)
        print(NL)
QUI
thread_outputs=()
for threads in 1 3 ""; do
    set +e
    thread_output="$(QUIDRA_CPU_THREADS="$threads" QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/thread-count.qui" 2>&1)"
    thread_status=$?
    set -e
    if [[ "$thread_status" -ne 0 || -z "$thread_output" ]] || grep -q "error" <<<"$thread_output"; then
        echo "Vision CPU thread-count program failed (QUIDRA_CPU_THREADS='$threads', status $thread_status):" >&2
        printf '%s\n' "$thread_output" | head -20 >&2
        exit 1
    fi
    thread_outputs+=("$thread_output")
done
if [[ "${thread_outputs[0]}" != "${thread_outputs[1]}" || "${thread_outputs[0]}" != "${thread_outputs[2]}" ]]; then
    echo "Vision CPU results depend on QUIDRA_CPU_THREADS" >&2
    diff <(printf '%s\n' "${thread_outputs[0]}") <(printf '%s\n' "${thread_outputs[2]}") | head -20 >&2
    exit 1
fi

bash "$REPOSITORY_ROOT/tests/nonfinite.sh" "$QUIDRA"
echo "vision integration: ok"
