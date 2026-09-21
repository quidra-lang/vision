#!/usr/bin/env bash
set -euo pipefail

QUIDRA="${1:-}"
if [[ -z "$QUIDRA" ]]; then
    echo "usage: $0 /path/to/quidra" >&2
    exit 2
fi

REPOSITORY_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PACKAGE_ROOT="$(dirname "$REPOSITORY_ROOT")"
GPU_INDEX="${QUIDRA_REAL_GPU_INDEX:-0}"
REQUIRE_REAL="${QUIDRA_REQUIRE_REAL_GPU:-0}"
REQUIRE_BACKEND="${QUIDRA_REQUIRE_GPU_BACKEND:-}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

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
if grep -Fq "backend: TEST" <<<"$gpu_info"; then skip_or_fail "fake GPU backend is active"; fi
if ! grep -Fq "GPU $GPU_INDEX" <<<"$gpu_info"; then skip_or_fail "gpu($GPU_INDEX) is not present"; fi
gpu_block="$(awk -v target="GPU $GPU_INDEX" '
    $0 == target { found = 1; print; next }
    found && /^GPU [0-9]+$/ { exit }
    found { print }
' <<<"$gpu_info")"
if [[ -n "$REQUIRE_BACKEND" ]] && ! grep -Fq "backend: $REQUIRE_BACKEND" <<<"$gpu_block"; then
    skip_or_fail "gpu($GPU_INDEX) is not backend $REQUIRE_BACKEND"
fi

cat > "$TMP/vision-real-gpu.qui" <<QUI
import vision

int | error run()
    tensor<uint8> cpu = tensor.zeros<uint8>([3, 3, 3])
    cpu[0, 0, 0] = uint8(100)
    cpu[1, 0, 0] = uint8(50)
    cpu[2, 0, 0] = uint8(10)
    cpu[0, 1, 1] = uint8(200)
    cpu[1, 1, 1] = uint8(100)
    cpu[2, 1, 1] = uint8(20)
    tensor<uint8> device_pixels = cpu.gpu($GPU_INDEX)

    tensor<uint8> cpu_crop = try vision.crop(cpu, 0, 0, 2, 2)
    tensor<uint8> gpu_crop = (try vision.crop(device_pixels, 0, 0, 2, 2)).cpu()
    print(cpu_crop[0, 1, 1].item() == gpu_crop[0, 1, 1].item())

    tensor<uint8> cpu_resize = try vision.resize(cpu, 2, 2)
    tensor<uint8> gpu_resize = (try vision.resize(device_pixels, 2, 2)).cpu()
    print(cpu_resize[0, 0, 0].item() == gpu_resize[0, 0, 0].item())

    tensor<uint8> cpu_flip = try vision.flip_horizontal(cpu)
    tensor<uint8> gpu_flip = (try vision.flip_horizontal(device_pixels)).cpu()
    print(cpu_flip[0, 1, 1].item() == gpu_flip[0, 1, 1].item())

    tensor<uint8> cpu_rotate = try vision.rotate90(cpu)
    tensor<uint8> gpu_rotate = (try vision.rotate90(device_pixels)).cpu()
    print(cpu_rotate[0, 0, 1].item() == gpu_rotate[0, 0, 1].item())

    tensor<uint8> cpu_rotate180 = try vision.rotate180(cpu)
    tensor<uint8> gpu_rotate180 = (try vision.rotate180(device_pixels)).cpu()
    print(cpu_rotate180[0, 0, 0].item() == gpu_rotate180[0, 0, 0].item())

    tensor<uint8> cpu_rotate270 = try vision.rotate270(cpu)
    tensor<uint8> gpu_rotate270 = (try vision.rotate270(device_pixels)).cpu()
    print(cpu_rotate270[0, 1, 0].item() == gpu_rotate270[0, 1, 0].item())

    tensor<uint8> cpu_gray = try vision.grayscale(cpu)
    tensor<uint8> gpu_gray = (try vision.grayscale(device_pixels)).cpu()
    print(cpu_gray[0, 0, 0].item() == gpu_gray[0, 0, 0].item())

    tensor<uint8> cpu_threshold = try vision.threshold(cpu, uint8(60))
    tensor<uint8> gpu_threshold = (try vision.threshold(device_pixels, uint8(60))).cpu()
    print(cpu_threshold[0, 0, 0].item() == gpu_threshold[0, 0, 0].item())

    tensor<uint8> cpu_blur = try vision.blur(cpu, 1)
    tensor<uint8> gpu_blur = (try vision.blur(device_pixels, 1)).cpu()
    print(cpu_blur[0, 1, 1].item() == gpu_blur[0, 1, 1].item())

    tensor<int> kernel = tensor.zeros<int>([3, 3])
    kernel[1, 1] = 1
    tensor<uint8> cpu_filter = try vision.filter(cpu, kernel)
    tensor<uint8> gpu_filter = (try vision.filter(device_pixels, kernel.gpu($GPU_INDEX))).cpu()
    print(cpu_filter[0, 1, 1].item() == gpu_filter[0, 1, 1].item())

    tensor<uint8> cpu_dilate = try vision.dilate(cpu, 1)
    tensor<uint8> gpu_dilate = (try vision.dilate(device_pixels, 1)).cpu()
    print(cpu_dilate[0, 1, 1].item() == gpu_dilate[0, 1, 1].item())

    tensor<uint8> cpu_erode = try vision.erode(cpu, 1)
    tensor<uint8> gpu_erode = (try vision.erode(device_pixels, 1)).cpu()
    print(cpu_erode[0, 1, 1].item() == gpu_erode[0, 1, 1].item())

    tensor<uint8> cpu_huge_blur = try vision.blur(cpu, 2147483647)
    tensor<uint8> gpu_huge_blur = (try vision.blur(device_pixels, 2147483647)).cpu()
    print(cpu_huge_blur[0, 1, 1].item() == gpu_huge_blur[0, 1, 1].item())

    tensor<uint8> cpu_huge_dilate = try vision.dilate(cpu, 2147483647)
    tensor<uint8> gpu_huge_dilate = (try vision.dilate(device_pixels, 2147483647)).cpu()
    print(cpu_huge_dilate[0, 1, 1].item() == gpu_huge_dilate[0, 1, 1].item())

    tensor<uint8> cpu_huge_erode = try vision.erode(cpu, 2147483647)
    tensor<uint8> gpu_huge_erode = (try vision.erode(device_pixels, 2147483647)).cpu()
    print(cpu_huge_erode[0, 1, 1].item() == gpu_huge_erode[0, 1, 1].item())

    tensor<float32> cpu_float = tensor.ones<float32>([1, 2, 3])
    tensor<float32> gpu_float = cpu_float.gpu($GPU_INDEX)
    tensor<float32> cpu_float_flip = try vision.flip_vertical(cpu_float)
    tensor<float32> gpu_float_flip = (try vision.flip_vertical(gpu_float)).cpu()
    print(cpu_float_flip[0, 1, 2].item() == gpu_float_flip[0, 1, 2].item())
    return 0

auto result = run()
match result
    int
        int ignored = result
    error problem
        print(problem)
QUI

output="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/vision-real-gpu.qui")"
expected="$(printf 'true\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue')"
if [[ "$output" != "$expected" ]]; then
    echo "Vision real GPU numerical equivalence failed on gpu($GPU_INDEX)" >&2
    printf '%s\n' "$output" >&2
    exit 1
fi

echo "vision real GPU integration: ok on gpu($GPU_INDEX)"
