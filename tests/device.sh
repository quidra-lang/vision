#!/usr/bin/env bash
set -euo pipefail

QUIDRA="$1"
REPOSITORY_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PACKAGE_ROOT="$(dirname "$REPOSITORY_ROOT"):$REPOSITORY_ROOT"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
export QUIDRA_CACHE_DIR="$TMP/run-cache"

# Effective only with the core's test-only fake GPU backend.
export QUIDRA_TEST_FAKE_GPU_COUNT=2

cat > "$TMP/device-check.qui" <<'QUI'
import vision

int | error compile_device_surface()
    tensor<nat8> direct = tensor.zeros<nat8>([3, 2, 2], gpu = 0)
    tensor<nat8> transferred = tensor.ones<nat8>([3, 2, 2]).gpu(0)
    tensor<nat8> roundtrip = transferred.cpu()
    tensor<nat8> transformed = try vision.flip_horizontal(direct)
    print(roundtrip.shape()[0])
    print(NL)
    print(transformed.shape()[0])
    print(NL)
    return 0
QUI

QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" check "$TMP/device-check.qui" >/dev/null

cat > "$TMP/no-fallback-create.qui" <<'QUI'
tensor<nat8> value = tensor.zeros<nat8>([1, 1, 1], gpu = 2147483647)
print(value.shape()[0])
print(NL)
QUI

set +e
QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/no-fallback-create.qui" >"$TMP/create.out" 2>"$TMP/create.err"
status=$?
set -e
if [[ $status -eq 0 ]]; then
    echo "GPU creation unexpectedly fell back to CPU" >&2
    exit 1
fi
if ! grep -Fq "gpu(2147483647) is not available" "$TMP/create.err"; then
    echo "missing explicit unavailable-GPU diagnostic" >&2
    cat "$TMP/create.err" >&2
    exit 1
fi

cat > "$TMP/no-fallback-transfer.qui" <<'QUI'
tensor<nat8> cpu = tensor.ones<nat8>([1, 1, 1])
tensor<nat8> value = cpu.gpu(2147483647)
print(value.shape()[0])
print(NL)
QUI

set +e
QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/no-fallback-transfer.qui" >"$TMP/transfer.out" 2>"$TMP/transfer.err"
status=$?
set -e
if [[ $status -eq 0 ]]; then
    echo "GPU transfer unexpectedly fell back to CPU" >&2
    exit 1
fi
if ! grep -Fq "gpu(2147483647) is not available" "$TMP/transfer.err"; then
    echo "missing explicit unavailable-GPU transfer diagnostic" >&2
    cat "$TMP/transfer.err" >&2
    exit 1
fi

expect_device_failure() {
    local source="$1"
    local expected="$2"
    set +e
    QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$source" >"$source.out" 2>"$source.err"
    local status=$?
    set -e
    if [[ $status -ne 101 ]]; then
        echo "expected runtime status 101 for $source, got $status" >&2
        cat "$source.out" >&2 || true
        cat "$source.err" >&2 || true
        exit 1
    fi
    if ! grep -Fq "$expected" "$source.err"; then
        echo "missing device diagnostic '$expected'" >&2
        cat "$source.err" >&2
        exit 1
    fi
    if [[ -s "$source.out" ]]; then
        echo "Vision produced CPU output before GPU failure" >&2
        cat "$source.out" >&2
        exit 1
    fi
}


cat > "$TMP/vision-gpu-compute.qui" <<'QUI'
import vision

int | error run()
    tensor<nat8> pixels = tensor.zeros<nat8>([1, 2, 3], gpu = 0)
    pixels[0, 0, 0] = nat8(1)
    pixels[0, 0, 1] = nat8(2)
    pixels[0, 0, 2] = nat8(3)
    pixels[0, 1, 0] = nat8(4)
    pixels[0, 1, 1] = nat8(5)
    pixels[0, 1, 2] = nat8(6)

    tensor<nat8> cropped = try vision.crop(
        pixels, top = 0, left = 1, height = 2, width = 2
    )
    tensor<nat8> resized = try vision.resize(pixels, height = 4, width = 6)
    tensor<nat8> horizontal = try vision.flip_horizontal(pixels)
    tensor<nat8> vertical = try vision.flip_vertical(pixels)
    tensor<nat8> turned90 = try vision.rotate90(pixels)
    tensor<nat8> turned180 = try vision.rotate180(pixels)
    tensor<nat8> turned270 = try vision.rotate270(pixels)

    print(cropped[0, 0, 0].item())
    print(NL)
    print(resized.shape()[1])
    print(NL)
    print(resized.shape()[2])
    print(NL)
    print(horizontal[0, 0, 0].item())
    print(NL)
    print(vertical[0, 0, 0].item())
    print(NL)
    print(turned90[0, 0, 0].item())
    print(NL)
    print(turned90[0, 2, 1].item())
    print(NL)
    print(turned180[0, 0, 0].item())
    print(NL)
    print(turned270[0, 0, 0].item())
    print(NL)

    tensor<nat8> rgb = tensor.ones<nat8>([3, 2, 2], gpu = 0)
    tensor<nat8> gray = try vision.grayscale(rgb)
    tensor<nat8> binary = try vision.threshold(pixels, cutoff = nat8(4))
    print(gray[0, 0, 0].item())
    print(NL)
    print(binary[0, 0, 0].item())
    print(NL)
    print(binary[0, 1, 2].item())
    print(NL)

    tensor<nat8> impulse = tensor.zeros<nat8>([1, 3, 3], gpu = 0)
    impulse[0, 1, 1] = nat8(255)
    tensor<nat8> blurred = try vision.blur(impulse, radius = 1)
    tensor<nat8> expanded = try vision.dilate(impulse, radius = 1)
    tensor<nat8> contracted = try vision.erode(impulse, radius = 1)
    tensor<nat8> huge_blurred = try vision.blur(impulse, radius = 2147483647)
    tensor<nat8> huge_expanded = try vision.dilate(impulse, radius = 2147483647)
    tensor<nat8> huge_contracted = try vision.erode(impulse, radius = 2147483647)
    tensor<int64> kernel = tensor.zeros<int64>([3, 3], gpu = 0)
    kernel[1, 1] = 1
    tensor<nat8> filtered = try vision.filter(impulse, kernel)
    print(blurred[0, 1, 1].item())
    print(NL)
    print(expanded[0, 0, 0].item())
    print(NL)
    print(contracted[0, 1, 1].item())
    print(NL)
    print(huge_blurred[0, 1, 1].item())
    print(NL)
    print(huge_expanded[0, 0, 0].item())
    print(NL)
    print(huge_contracted[0, 1, 1].item())
    print(NL)
    print(filtered[0, 1, 1].item())
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

vision_output="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/vision-gpu-compute.qui")"
vision_expected="$(printf '2\n4\n6\n3\n4\n4\n3\n6\n3\n1\n0\n255\n28\n255\n0\n28\n255\n0\n255')"
if [[ "$vision_output" != "$vision_expected" ]]; then
    echo "unexpected GPU Vision output:" >&2
    printf '%s\n' "$vision_output" >&2
    exit 1
fi

cat > "$TMP/filter-device-mismatch.qui" <<'QUI'
import vision

tensor<nat8> pixels = tensor.ones<nat8>([1, 2, 2], gpu = 0)
tensor<int64> kernel = tensor.ones<int64>([1, 1])
tensor<nat8> | error output = vision.filter(pixels, kernel)
match output
    tensor<nat8> value
        print(value.shape()[0])
        print(NL)
    error problem
        print(problem)
        print(NL)
QUI
expect_device_failure "$TMP/filter-device-mismatch.qui" "tensor operands are on different devices"

cat > "$TMP/image-write-gpu.qui" <<'QUI'
import vision

tensor<nat8> image_gpu = tensor.ones<nat8>([1, 1, 1], gpu = 0)
auto | error written = vision.write("should-not-exist.png", image_gpu)
match written
    void
        print("unexpected success")
        print(NL)
    error problem
        print(problem)
        print(NL)
QUI

write_output="$(cd "$TMP" && QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" run "$TMP/image-write-gpu.qui")"
if [[ "$write_output" != "vision.write failed; image codecs require a CPU tensor and a supported file extension" ]]; then
    echo "unexpected GPU vision.write result: $write_output" >&2
    exit 1
fi
if [[ -e "$TMP/should-not-exist.png" ]]; then
    echo "GPU vision.write unexpectedly wrote a CPU-fallback file" >&2
    exit 1
fi

# The portable composition (the CUDA/HIP path) is reached through a copy of the
# package's internal module and compared with the native kernels.
cp "$REPOSITORY_ROOT/internal.qui" "$TMP/vision_internal.qui"

cat > "$TMP/downsample-device.qui" <<'QUI'
import vision
import math
import composition = "./vision_internal.qui"

int differences(tensor<real32> left, tensor<real32> right)
    int[] shape = int(left.shape())
    int count = 0
    for 0 <= int c < shape[0]
        for 0 <= int y < shape[1]
            for 0 <= int x < shape[2]
                if left[c, y, x].item() != right[c, y, x].item()
                    count += 1
    return count

int | error run()
    tensor<real32> host = tensor.zeros<real32>([2, 7, 9])
    tensor<nat8> host_bytes = tensor.zeros<nat8>([2, 7, 9])
    int index = 0
    for 0 <= nat c < 2
        for 0 <= nat y < 7
            for 0 <= nat x < 9
                real32 sample = real32((index * 13) % 11)
                host[c, y, x] = sample * real32(0.3) - real32(1)
                int byte_value = (index * 57) % 256
                host_bytes[c, y, x] = nat8(byte_value)
                index += 1

    // Results stay on the input's device and match the CPU result.
    tensor<real32> on_device = try vision.downsample_mean(host.gpu(1), 3)
    tensor<real32> on_host = try vision.downsample_mean(host, 3)
    print(on_device.device())
    print(NL)
    print(differences(on_device.cpu(), on_host))
    print(NL)

    tensor<nat8> bytes_on_device = try vision.downsample_mean(host_bytes.gpu(0), 2)
    tensor<nat8> bytes_on_host = try vision.downsample_mean(host_bytes, 2)
    print(bytes_on_device.device())
    print(NL)
    tensor<nat8> bytes_back = bytes_on_device.cpu()
    int byte_differences = 0
    for 0 <= nat c < 2
        for 0 <= nat y < 3
            for 0 <= nat x < 4
                if bytes_back[c, y, x].item() != bytes_on_host[c, y, x].item()
                    byte_differences += 1
    print(byte_differences)
    print(NL)

    // Tracked device input keeps its graph and gradients on that device.
    tensor<real32> source = host.gpu(0).track()
    tensor<real32> reduced = try vision.downsample_mean(source, 3)
    print(reduced.is_tracked())
    print(NL)
    math.mean(reduced).backward(&source)
    print(source.grad.device())
    print(NL)
    tensor<real32> host_source = host.track()
    tensor<real32> host_reduced = try vision.downsample_mean(host_source, 3)
    math.mean(host_reduced).backward(&host_source)
    print(differences(source.grad.cpu(), host_source.grad))
    print(NL)

    // A tracked non-contiguous view stays on the device, and its values and
    // gradients match the CPU.
    tensor<real32> view = host.gpu(0).transpose(1, 2).track()
    tensor<real32> view_reduced = try vision.downsample_mean(view, 2)
    print(view_reduced.device())
    print(NL)
    tensor<real32> host_view = host.transpose(1, 2).track()
    tensor<real32> host_view_reduced = try vision.downsample_mean(host_view, 2)
    math.mean(view_reduced * view_reduced).backward(&view)
    math.mean(host_view_reduced * host_view_reduced).backward(&host_view)
    print(view.grad.device())
    print(NL)
    print(differences(view.grad.cpu(), host_view.grad) + differences(view_reduced.untrack().cpu(), host_view_reduced.untrack()))
    print(NL)

    // A strided upstream gradient (transpose after the block mean) reaches
    // the device gradient kernel densely and matches the CPU.
    tensor<real32> weights = tensor.zeros<real32>([2, 3, 2])
    for 0 <= nat c < 2
        for 0 <= nat y < 3
            for 0 <= nat x < 2
                nat weight_index = c * 6 + y * 2 + x + 1
                real32 weight_sample = real32(weight_index)
                weights[c, y, x] = weight_sample * real32(0.25)
    tensor<real32> strided_source = host.gpu(1).track()
    tensor<real32> strided_reduced = try vision.downsample_mean(strided_source, 3)
    math.mean(strided_reduced.transpose(1, 2) * weights.gpu(1)).backward(&strided_source)
    tensor<real32> host_strided_source = host.track()
    tensor<real32> host_strided_reduced = try vision.downsample_mean(host_strided_source, 3)
    math.mean(host_strided_reduced.transpose(1, 2) * weights).backward(&host_strided_source)
    print(strided_source.grad.device())
    print(NL)
    print(differences(strided_source.grad.cpu(), host_strided_source.grad))
    print(NL)

    // The portable composition gives the native results on CPU and on the
    // device, for floating and nat8 input.
    tensor<real32> portable_host = composition.block_mean<real32>(host, 3)
    tensor<real32> portable_device = composition.block_mean<real32>(host.gpu(0), 3)
    print(portable_device.device())
    print(NL)
    print(differences(portable_host, on_host) + differences(portable_device.cpu(), on_host))
    print(NL)
    tensor<nat8> portable_bytes_host = try composition.block_mean_u8(host_bytes, 2)
    tensor<nat8> portable_bytes_device = try composition.block_mean_u8(host_bytes.gpu(1), 2)
    print(portable_bytes_device.device())
    print(NL)
    tensor<nat8> portable_bytes_back = portable_bytes_device.cpu()
    int portable_byte_differences = 0
    for 0 <= nat c < 2
        for 0 <= nat y < 3
            for 0 <= nat x < 4
                if portable_bytes_host[c, y, x].item() != bytes_on_host[c, y, x].item()
                    portable_byte_differences += 1
                if portable_bytes_back[c, y, x].item() != bytes_on_host[c, y, x].item()
                    portable_byte_differences += 1
    print(portable_byte_differences)
    print(NL)
    tensor<real32> portable_source = host.gpu(0).track()
    tensor<real32> portable_reduced = composition.block_mean<real32>(portable_source, 3)
    math.mean(portable_reduced).backward(&portable_source)
    print(differences(portable_source.grad.cpu(), host_source.grad))
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

downsample_output="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/downsample-device.qui")"
downsample_expected="$(printf '1\n0\n0\n0\ntrue\n0\n0\n0\n0\n0\n1\n0\n0\n0\n1\n0\n0')"
if [[ "$downsample_output" != "$downsample_expected" ]]; then
    echo "unexpected Vision downsample_mean device output:" >&2
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
    tensor<real32> source = pattern32(2, 64, 401, 1).gpu(0).track()
    tensor<real32> reduced = try vision.downsample_mean(source, 3)
    tensor<real32> weights = pattern32(2, 21, 133, 2).gpu(0)
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
    sample8("mean8", try vision.downsample_mean(bytes.gpu(0), 3))
    return 0

int | error run64()
    tensor<real64> source = pattern64(2, 64, 401, 3).gpu(0).track()
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
        echo "Vision fake-GPU thread-count program failed (QUIDRA_CPU_THREADS='$threads', status $thread_status):" >&2
        printf '%s\n' "$thread_output" | head -20 >&2
        exit 1
    fi
    thread_outputs+=("$thread_output")
done
if [[ "${thread_outputs[0]}" != "${thread_outputs[1]}" || "${thread_outputs[0]}" != "${thread_outputs[2]}" ]]; then
    echo "Vision fake-GPU results depend on QUIDRA_CPU_THREADS" >&2
    diff <(printf '%s\n' "${thread_outputs[0]}") <(printf '%s\n' "${thread_outputs[2]}") | head -20 >&2
    exit 1
fi

# Core lends no storage of a view that contains an uninitialized element, so
# an uninitialized device input of downsample_mean takes the portable
# composition, whose read through Core reports UNINITIALIZED, as on the CPU.
for element in real32 nat8; do
    cat > "$TMP/uninitialized-$element.qui" <<QUI
import vision

int | error run()
    tensor<$element> pixels = tensor<$element>([1, 4, 4], gpu = 0)
    pixels[0, 0, 0] = $element(1)
    tensor<$element> reduced = try vision.downsample_mean(pixels, 2)
    print(reduced.cpu()[0, 0, 0].item())
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
    expect_device_failure "$TMP/uninitialized-$element.qui" "UNINITIALIZED"
done

echo "vision device contracts: ok"
