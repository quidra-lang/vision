#!/usr/bin/env bash
set -euo pipefail

QUIDRA="${1:-}"
if [[ -z "$QUIDRA" ]]; then
    echo "usage: $0 /path/to/quidra" >&2
    exit 2
fi

REPOSITORY_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PACKAGE_ROOT="$(dirname "$REPOSITORY_ROOT"):$REPOSITORY_ROOT"
GPU_INDEX="${QUIDRA_REAL_GPU_INDEX:-0}"
REQUIRE_REAL="${QUIDRA_REQUIRE_REAL_GPU:-0}"
REQUIRE_BACKEND="${QUIDRA_REQUIRE_GPU_BACKEND:-}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
export QUIDRA_CACHE_DIR="$TMP/run-cache"

set +e
gpu_info="$("$QUIDRA" gpu 2>&1)"
gpu_status=$?
set -e
skip_or_fail() {
    local reason="$1"
    if [[ "$REQUIRE_REAL" == "1" ]]; then
        echo "real Vision GPU integration required but unavailable: $reason" >&2
        printf '%s\n' "$gpu_info" >&2
        exit 1
    fi
    echo "vision real GPU integration: skipped ($reason)"
    exit 0
}
if [[ $gpu_status -ne 0 ]]; then skip_or_fail "quidra gpu failed"; fi
if grep -Fiq "backend: TEST" <<<"$gpu_info"; then skip_or_fail "fake GPU backend is active"; fi
if ! grep -Fq "GPU $GPU_INDEX" <<<"$gpu_info"; then skip_or_fail "gpu($GPU_INDEX) is not present"; fi
gpu_block="$(awk -v target="GPU $GPU_INDEX" '
    $0 == target { found = 1; print; next }
    found && /^GPU [0-9]+$/ { exit }
    found { print }
' <<<"$gpu_info")"
if [[ -n "$REQUIRE_BACKEND" ]] && ! grep -Fiq "backend: $REQUIRE_BACKEND" <<<"$gpu_block"; then
    skip_or_fail "gpu($GPU_INDEX) is not backend $REQUIRE_BACKEND"
fi

# Metal has no float64 arithmetic; vision.downsample_mean returns a VISION_NATIVE
# error there instead of aborting inside Core.
IS_METAL=false
if grep -Fq "backend: Metal" <<<"$gpu_block"; then IS_METAL=true; fi

# The portable composition (the CUDA/HIP path) is reached through a copy of the
# package's internal module.
cp "$REPOSITORY_ROOT/internal.qui" "$TMP/vision_internal.qui"

cat > "$TMP/vision-real-gpu.qui" <<QUI
import vision
import math
import composition = "./vision_internal.qui"

int | error run()
    tensor<nat8> cpu = tensor.zeros<nat8>([3, 3, 3])
    cpu[0, 0, 0] = nat8(100)
    cpu[1, 0, 0] = nat8(50)
    cpu[2, 0, 0] = nat8(10)
    cpu[0, 1, 1] = nat8(200)
    cpu[1, 1, 1] = nat8(100)
    cpu[2, 1, 1] = nat8(20)
    tensor<nat8> device_pixels = cpu.gpu($GPU_INDEX)

    tensor<nat8> cpu_crop = try vision.crop(cpu, 0, 0, 2, 2)
    tensor<nat8> gpu_crop = (try vision.crop(device_pixels, 0, 0, 2, 2)).cpu()
    print(cpu_crop[0, 1, 1].item() == gpu_crop[0, 1, 1].item())
    print(NL)

    tensor<nat8> cpu_resize = try vision.resize(cpu, 2, 2)
    tensor<nat8> gpu_resize = (try vision.resize(device_pixels, 2, 2)).cpu()
    print(cpu_resize[0, 0, 0].item() == gpu_resize[0, 0, 0].item())
    print(NL)

    tensor<nat8> cpu_flip = try vision.flip_horizontal(cpu)
    tensor<nat8> gpu_flip = (try vision.flip_horizontal(device_pixels)).cpu()
    print(cpu_flip[0, 1, 1].item() == gpu_flip[0, 1, 1].item())
    print(NL)

    tensor<nat8> cpu_rotate = try vision.rotate90(cpu)
    tensor<nat8> gpu_rotate = (try vision.rotate90(device_pixels)).cpu()
    print(cpu_rotate[0, 0, 1].item() == gpu_rotate[0, 0, 1].item())
    print(NL)

    tensor<nat8> cpu_rotate180 = try vision.rotate180(cpu)
    tensor<nat8> gpu_rotate180 = (try vision.rotate180(device_pixels)).cpu()
    print(cpu_rotate180[0, 0, 0].item() == gpu_rotate180[0, 0, 0].item())
    print(NL)

    tensor<nat8> cpu_rotate270 = try vision.rotate270(cpu)
    tensor<nat8> gpu_rotate270 = (try vision.rotate270(device_pixels)).cpu()
    print(cpu_rotate270[0, 1, 0].item() == gpu_rotate270[0, 1, 0].item())
    print(NL)

    tensor<nat8> cpu_gray = try vision.grayscale(cpu)
    tensor<nat8> gpu_gray = (try vision.grayscale(device_pixels)).cpu()
    print(cpu_gray[0, 0, 0].item() == gpu_gray[0, 0, 0].item())
    print(NL)

    tensor<nat8> cpu_threshold = try vision.threshold(cpu, nat8(60))
    tensor<nat8> gpu_threshold = (try vision.threshold(device_pixels, nat8(60))).cpu()
    print(cpu_threshold[0, 0, 0].item() == gpu_threshold[0, 0, 0].item())
    print(NL)

    tensor<nat8> cpu_blur = try vision.blur(cpu, 1)
    tensor<nat8> gpu_blur = (try vision.blur(device_pixels, 1)).cpu()
    print(cpu_blur[0, 1, 1].item() == gpu_blur[0, 1, 1].item())
    print(NL)

    tensor<int64> kernel = tensor.zeros<int64>([3, 3])
    kernel[1, 1] = 1
    tensor<nat8> cpu_filter = try vision.filter(cpu, kernel)
    tensor<nat8> gpu_filter = (try vision.filter(device_pixels, kernel.gpu($GPU_INDEX))).cpu()
    print(cpu_filter[0, 1, 1].item() == gpu_filter[0, 1, 1].item())
    print(NL)

    tensor<nat8> cpu_dilate = try vision.dilate(cpu, 1)
    tensor<nat8> gpu_dilate = (try vision.dilate(device_pixels, 1)).cpu()
    print(cpu_dilate[0, 1, 1].item() == gpu_dilate[0, 1, 1].item())
    print(NL)

    tensor<nat8> cpu_erode = try vision.erode(cpu, 1)
    tensor<nat8> gpu_erode = (try vision.erode(device_pixels, 1)).cpu()
    print(cpu_erode[0, 1, 1].item() == gpu_erode[0, 1, 1].item())
    print(NL)

    tensor<nat8> cpu_huge_blur = try vision.blur(cpu, 2147483647)
    tensor<nat8> gpu_huge_blur = (try vision.blur(device_pixels, 2147483647)).cpu()
    print(cpu_huge_blur[0, 1, 1].item() == gpu_huge_blur[0, 1, 1].item())
    print(NL)

    tensor<nat8> cpu_huge_dilate = try vision.dilate(cpu, 2147483647)
    tensor<nat8> gpu_huge_dilate = (try vision.dilate(device_pixels, 2147483647)).cpu()
    print(cpu_huge_dilate[0, 1, 1].item() == gpu_huge_dilate[0, 1, 1].item())
    print(NL)

    tensor<nat8> cpu_huge_erode = try vision.erode(cpu, 2147483647)
    tensor<nat8> gpu_huge_erode = (try vision.erode(device_pixels, 2147483647)).cpu()
    print(cpu_huge_erode[0, 1, 1].item() == gpu_huge_erode[0, 1, 1].item())
    print(NL)

    tensor<real32> cpu_float = tensor.ones<real32>([1, 2, 3])
    tensor<real32> gpu_float = cpu_float.gpu($GPU_INDEX)
    tensor<real32> cpu_float_flip = try vision.flip_vertical(cpu_float)
    tensor<real32> gpu_float_flip = (try vision.flip_vertical(gpu_float)).cpu()
    print(cpu_float_flip[0, 1, 2].item() == gpu_float_flip[0, 1, 2].item())
    print(NL)

    tensor<real32> cpu_float_blur = try vision.blur(cpu_float, radius = 1)
    tensor<real32> gpu_float_blur_source = gpu_float.track()
    tensor<real32> gpu_float_blur_tracked = try vision.blur(
        gpu_float_blur_source, radius = 1
    )
    tensor<real32> gpu_float_blur = gpu_float_blur_tracked.untrack().cpu()
    print(cpu_float_blur[0, 1, 2].item() == gpu_float_blur[0, 1, 2].item())
    print(NL)
    math.mean(gpu_float_blur_tracked).backward(&gpu_float_blur_source)
    print(gpu_float_blur_source.has_grad())
    print(NL)

    // Block means: Vision's device kernels use the CPU accumulation order, so
    // results match exactly, including a divisor that is not a power of two.
    tensor<real32> cpu_blocks = tensor.zeros<real32>([2, 10, 14])
    tensor<nat8> cpu_block_bytes = tensor.zeros<nat8>([2, 10, 14])
    int block_index = 0
    for 0 <= nat c < 2
        for 0 <= nat y < 10
            for 0 <= nat x < 14
                int float_value = (block_index * 31) % 17
                real32 float_sample = real32(float_value)
                cpu_blocks[c, y, x] = float_sample * real32(0.13) - real32(0.9)
                int byte_value = (block_index * 67) % 256
                cpu_block_bytes[c, y, x] = nat8(byte_value)
                block_index += 1
    tensor<real32> cpu_block_mean = try vision.downsample_mean(cpu_blocks, 3)
    tensor<real32> gpu_block_mean_device = try vision.downsample_mean(cpu_blocks.gpu($GPU_INDEX), 3)
    print(gpu_block_mean_device.device() == $GPU_INDEX)
    print(NL)
    tensor<real32> gpu_block_mean = gpu_block_mean_device.cpu()
    tensor<nat8> cpu_block_bytes_mean = try vision.downsample_mean(cpu_block_bytes, 3)
    tensor<nat8> gpu_block_bytes_mean = (try vision.downsample_mean(cpu_block_bytes.gpu($GPU_INDEX), 3)).cpu()
    int block_differences = 0
    int byte_block_differences = 0
    for 0 <= nat c < 2
        for 0 <= nat y < 3
            for 0 <= nat x < 4
                if cpu_block_mean[c, y, x].item() != gpu_block_mean[c, y, x].item()
                    block_differences += 1
                if cpu_block_bytes_mean[c, y, x].item() != gpu_block_bytes_mean[c, y, x].item()
                    byte_block_differences += 1
    print(block_differences == 0)
    print(NL)
    print(byte_block_differences == 0)
    print(NL)

    tensor<real32> cpu_block_source = cpu_blocks.track()
    tensor<real32> cpu_block_tracked = try vision.downsample_mean(cpu_block_source, 3)
    math.mean(cpu_block_tracked).backward(&cpu_block_source)
    tensor<real32> gpu_block_source = cpu_blocks.gpu($GPU_INDEX).track()
    tensor<real32> gpu_block_tracked = try vision.downsample_mean(gpu_block_source, 3)
    math.mean(gpu_block_tracked).backward(&gpu_block_source)
    tensor<real32> gpu_block_grad = gpu_block_source.grad.cpu()
    int grad_differences = 0
    for 0 <= nat c < 2
        for 0 <= nat y < 10
            for 0 <= nat x < 14
                if cpu_block_source.grad[c, y, x].item() != gpu_block_grad[c, y, x].item()
                    grad_differences += 1
    print(gpu_block_source.grad.device() == $GPU_INDEX and grad_differences == 0)
    print(NL)

    // A tracked strided view and a strided upstream gradient (transpose after
    // the block mean) both use the Vision kernels and match the CPU exactly.
    tensor<real32> cpu_view_source = cpu_blocks.transpose(1, 2).track()
    tensor<real32> cpu_view_blocks = try vision.downsample_mean(cpu_view_source, 3)
    math.mean(cpu_view_blocks.transpose(1, 2) * cpu_view_blocks.transpose(1, 2)).backward(&cpu_view_source)
    tensor<real32> gpu_view_source = cpu_blocks.gpu($GPU_INDEX).transpose(1, 2).track()
    tensor<real32> gpu_view_blocks = try vision.downsample_mean(gpu_view_source, 3)
    math.mean(gpu_view_blocks.transpose(1, 2) * gpu_view_blocks.transpose(1, 2)).backward(&gpu_view_source)
    tensor<real32> gpu_view_values = gpu_view_blocks.untrack().cpu()
    tensor<real32> gpu_view_grad = gpu_view_source.grad.cpu()
    int view_differences = 0
    for 0 <= nat c < 2
        for 0 <= nat y < 4
            for 0 <= nat x < 3
                if cpu_view_blocks.untrack()[c, y, x].item() != gpu_view_values[c, y, x].item()
                    view_differences += 1
        for 0 <= nat y < 14
            for 0 <= nat x < 10
                if cpu_view_source.grad[c, y, x].item() != gpu_view_grad[c, y, x].item()
                    view_differences += 1
    print(gpu_view_source.grad.device() == $GPU_INDEX and view_differences == 0)
    print(NL)

    // The portable composition stays on the device; nat8 and power-of-two
    // floating block means match the Vision kernels exactly.
    tensor<nat8> portable_bytes = (try composition.block_mean_u8(cpu_block_bytes.gpu($GPU_INDEX), 3)).cpu()
    tensor<real32> native_quarter = try vision.downsample_mean(cpu_blocks, 2)
    tensor<real32> portable_quarter_device = composition.block_mean<real32>(cpu_blocks.gpu($GPU_INDEX), 2)
    tensor<real32> portable_quarter = portable_quarter_device.cpu()
    int portable_differences = 0
    for 0 <= nat c < 2
        for 0 <= nat y < 3
            for 0 <= nat x < 4
                if portable_bytes[c, y, x].item() != cpu_block_bytes_mean[c, y, x].item()
                    portable_differences += 1
        for 0 <= nat y < 5
            for 0 <= nat x < 7
                if portable_quarter[c, y, x].item() != native_quarter[c, y, x].item()
                    portable_differences += 1
    print(portable_quarter_device.device() == $GPU_INDEX and portable_differences == 0)
    print(NL)

    // backward(track = true) on the device: the first gradient is a tracked
    // spread whose own backward is the block mean, and higher orders keep
    // alternating between the two kernels. The values are exact dyadic
    // fractions (see develop_autograd.sh), so every backend gives them. The
    // second-order losses read the results through transposes, so the tracked
    // callbacks receive gradients that reach them strided.
    tensor<real32> high_source = tensor.ones<real32>([1, 1, 4, 4]).gpu($GPU_INDEX).track()
    tensor<real32> high = try vision.downsample_mean(high_source, 2)
    math.mean(high.transpose(2, 3) * high.transpose(2, 3)).backward(&high_source, track = true)
    tensor<real32> high_first = high_source.grad
    bool high_first_ok = high_first.is_tracked() and high_first.device() == $GPU_INDEX
    high_first_ok = high_first_ok and high_first.untrack().cpu()[0, 0, 2, 1].item() == real32(0.125)
    math.mean(high_first.transpose(2, 3)).backward(&high_source)
    print(high_first_ok and high_source.grad.untrack().cpu()[0, 0, 3, 3].item() == real32(0.1328125))
    print(NL)
    tensor<real32> third_source = tensor.ones<real32>([1, 1, 4, 4]).gpu($GPU_INDEX).track()
    tensor<real32> third = try vision.downsample_mean(third_source, 2)
    math.mean(third * third * third).backward(&third_source, track = true)
    tensor<real32> third_first = third_source.grad
    math.mean(third_first * third_first).backward(&third_source, track = true)
    tensor<real32> third_second = third_source.grad
    bool third_ok = third_second.is_tracked() and third_second.untrack().cpu()[0, 0, 0, 0].item() == real32(0.1962890625)
    math.mean(third_second).backward(&third_source)
    print(third_ok and third_source.grad.untrack().cpu()[0, 0, 1, 3].item() == real32(0.22137451171875))
    print(NL)

    // Normal-range values match the CPU exactly (above). A GPU may flush
    // subnormal real32 values to zero (Metal does); it must never produce
    // anything other than the CPU value or zero.
    tensor<real32> subnormal = tensor.ones<real32>([1, 2, 4]) * real32(1.0e-39)
    subnormal[0, 0, 2] = real32(1.0e-45)
    subnormal[0, 0, 3] = real32(1.0e-45)
    tensor<real32> cpu_subnormal = try vision.downsample_mean(subnormal, 2)
    tensor<real32> gpu_subnormal = (try vision.downsample_mean(subnormal.gpu($GPU_INDEX), 2)).cpu()
    bool subnormal_ok = true
    for 0 <= nat x < 2
        real32 gpu_value = gpu_subnormal[0, 0, x].item()
        if gpu_value != cpu_subnormal[0, 0, x].item() and gpu_value != real32(0.0)
            subnormal_ok = false
    print(subnormal_ok)
    print(NL)

    if $IS_METAL
        tensor<real64> double_pixels = tensor.ones<real64>([1, 4, 6]).gpu($GPU_INDEX)
        tensor<real64> | error double_result = vision.downsample_mean(double_pixels, 2)
        bool double_rejected = false
        match double_result
            tensor<real64>
                double_rejected = false
            error problem
                double_rejected = problem.code == "VISION_NATIVE"
        print(double_rejected)
    else
        print(true)
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

expected="$(printf 'true\n%.0s' {1..28})"
expect_equivalence() {
    local label="$1"
    shift
    local output
    output="$(env "$@" QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/vision-real-gpu.qui")"
    if [[ "$output" != "$expected" ]]; then
        echo "Vision real GPU numerical equivalence failed on gpu($GPU_INDEX) ($label)" >&2
        printf '%s\n' "$output" >&2
        exit 1
    fi
}
expect_equivalence "default"

if [[ "$IS_METAL" == true ]]; then
    # Without Core's command stream Vision fills a command buffer of its own on
    # Core's queue and commits it without waiting; the results are the same.
    expect_equivalence "without Core's command stream" QUIDRA_METAL_STREAM=0

    # Block-mean outputs are write-only on Metal: Core lends them without a
    # fill, and every Vision kernel stores each element. The gradients Vision's
    # first-order callbacks receive are declared full writes, so they arrive
    # without a fill too. With every Metal buffer poisoned when Core hands it
    # out, the value checks still hold.
    expect_equivalence "over poisoned buffers" QUIDRA_TEST_POOL_POISON=0x7f

    # Vision encodes into Core's Metal command stream: every block-mean or
    # spread call holds the device stream once and opens one encoder scope,
    # and Vision never borrows Core's queue (which would commit Core's open
    # work) or waits on the host. Each step below makes two forward calls
    # (float32 tracked, uint8) and one backward callback; the loss is a Core
    # gather, so no other package encodes. A step ends when backward() starts,
    # so every step after the first also holds the previous backward. Neither
    # the block-mean outputs (write-only) nor the gradient Vision's callback
    # receives (a declared full write) is zero-filled, and the Core work of a
    # step needs no fill either, so a step fills nothing.
    cat > "$TMP/stream-counters-real-gpu.qui" <<QUI
import vision

int | error run()
    tensor<real32> pixels = tensor.zeros<real32>([2, 9, 12])
    tensor<nat8> bytes = tensor.zeros<nat8>([2, 9, 12])
    for 0 <= nat c < 2
        for 0 <= nat y < 9
            for 0 <= nat x < 12
                nat pixel_value = (c * 108 + y * 12 + x) % 23
                real32 pixel_sample = real32(pixel_value)
                pixels[c, y, x] = pixel_sample * real32(0.25)
                nat byte_value = (c * 108 + y * 12 + x) % 256
                bytes[c, y, x] = nat8(byte_value)
    tensor<real32> source = pixels.gpu($GPU_INDEX).track()
    tensor<nat8> device_bytes = bytes.gpu($GPU_INDEX)
    for 0 <= nat step < 4
        tensor<real32> reduced = try vision.downsample_mean(source, 3)
        tensor<nat8> reduced_bytes = try vision.downsample_mean(device_bytes, 3)
        tensor<real32> picked = reduced.gather([5], [])
        picked.backward(&source)
    print(source.grad.device() == $GPU_INDEX)
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
    counter_output="$(QUIDRA_METAL_STREAM=1 QUIDRA_COUNTERS="$TMP/stream-counters.jsonl" QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" run "$TMP/stream-counters-real-gpu.qui")"
    if [[ "$counter_output" != "true" ]]; then
        echo "unexpected Vision stream counter program output:" >&2
        printf '%s\n' "$counter_output" >&2
        exit 1
    fi
    python3 - "$TMP/stream-counters.jsonl" <<'PY'
import json
import sys

records = [json.loads(line) for line in open(sys.argv[1])]
steps = [r for r in records if r["record"] == "step" and r.get("event") == "backward"]
total = [r for r in records if r["record"] == "total"][0]
assert len(steps) == 4, records
for step in steps[1:]:
    assert step["package_encode_holds"] == 3, step
    assert step["package_encode_scopes"] == 3, step
    assert step["fills"] == 0, step
for key in ("package_queue_borrows", "package_borrow_flushes",
            "host_waits_status", "package_scope_violations"):
    assert total[key] == 0, (key, total)
PY

    # Core lends no Metal handle of a view that contains an uninitialized
    # element, so an uninitialized device input of downsample_mean takes the
    # portable composition, whose read through Core reports UNINITIALIZED, as
    # on the CPU.
    for element in real32 nat8; do
        cat > "$TMP/uninitialized-$element.qui" <<QUI
import vision

int | error run()
    tensor<$element> pixels = tensor<$element>([1, 4, 4], gpu = $GPU_INDEX)
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
        set +e
        QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/uninitialized-$element.qui" >"$TMP/uninitialized.out" 2>"$TMP/uninitialized.err"
        uninitialized_status=$?
        set -e
        if [[ "$uninitialized_status" -ne 101 || -s "$TMP/uninitialized.out" ]] ||
           ! grep -Fq "UNINITIALIZED" "$TMP/uninitialized.err"; then
            echo "uninitialized $element Metal input of downsample_mean was not reported (status $uninitialized_status):" >&2
            cat "$TMP/uninitialized.out" "$TMP/uninitialized.err" >&2
            exit 1
        fi
    done
fi

echo "vision real GPU integration: ok on gpu($GPU_INDEX)"
