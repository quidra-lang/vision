#!/usr/bin/env bash
set -euo pipefail

QUIDRA="$1"
REPOSITORY_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PACKAGE_ROOT="$(dirname "$REPOSITORY_ROOT")"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# Effective only with the core's test-only fake GPU backend.
export QUIDRA_TEST_FAKE_GPU_COUNT=2

cat > "$TMP/device-check.qui" <<'QUI'
import vision

int | error compile_device_surface()
    tensor<uint8> direct = tensor.zeros<uint8>([3, 2, 2], gpu = 0)
    tensor<uint8> transferred = tensor.ones<uint8>([3, 2, 2]).gpu(0)
    tensor<uint8> roundtrip = transferred.cpu()
    tensor<uint8> transformed = try vision.flip_horizontal(direct)
    print(roundtrip.shape()[0])
    print(transformed.shape()[0])
    return 0
QUI

QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" check "$TMP/device-check.qui" >/dev/null

cat > "$TMP/no-fallback-create.qui" <<'QUI'
tensor<uint8> value = tensor.zeros<uint8>([1, 1, 1], gpu = 2147483647)
print(value.shape()[0])
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
tensor<uint8> cpu = tensor.ones<uint8>([1, 1, 1])
tensor<uint8> value = cpu.gpu(2147483647)
print(value.shape()[0])
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
    tensor<uint8> pixels = tensor.zeros<uint8>([1, 2, 3], gpu = 0)
    pixels[0, 0, 0] = uint8(1)
    pixels[0, 0, 1] = uint8(2)
    pixels[0, 0, 2] = uint8(3)
    pixels[0, 1, 0] = uint8(4)
    pixels[0, 1, 1] = uint8(5)
    pixels[0, 1, 2] = uint8(6)

    tensor<uint8> cropped = try vision.crop(
        pixels, top = 0, left = 1, height = 2, width = 2
    )
    tensor<uint8> resized = try vision.resize(pixels, height = 4, width = 6)
    tensor<uint8> horizontal = try vision.flip_horizontal(pixels)
    tensor<uint8> vertical = try vision.flip_vertical(pixels)
    tensor<uint8> turned90 = try vision.rotate90(pixels)
    tensor<uint8> turned180 = try vision.rotate180(pixels)
    tensor<uint8> turned270 = try vision.rotate270(pixels)

    print(cropped[0, 0, 0].item())
    print(resized.shape()[1])
    print(resized.shape()[2])
    print(horizontal[0, 0, 0].item())
    print(vertical[0, 0, 0].item())
    print(turned90[0, 0, 0].item())
    print(turned90[0, 2, 1].item())
    print(turned180[0, 0, 0].item())
    print(turned270[0, 0, 0].item())

    tensor<uint8> rgb = tensor.ones<uint8>([3, 2, 2], gpu = 0)
    tensor<uint8> gray = try vision.grayscale(rgb)
    tensor<uint8> binary = try vision.threshold(pixels, cutoff = uint8(4))
    print(gray[0, 0, 0].item())
    print(binary[0, 0, 0].item())
    print(binary[0, 1, 2].item())

    tensor<uint8> impulse = tensor.zeros<uint8>([1, 3, 3], gpu = 0)
    impulse[0, 1, 1] = uint8(255)
    tensor<uint8> blurred = try vision.blur(impulse, radius = 1)
    tensor<uint8> expanded = try vision.dilate(impulse, radius = 1)
    tensor<uint8> contracted = try vision.erode(impulse, radius = 1)
    tensor<uint8> huge_blurred = try vision.blur(impulse, radius = 2147483647)
    tensor<uint8> huge_expanded = try vision.dilate(impulse, radius = 2147483647)
    tensor<uint8> huge_contracted = try vision.erode(impulse, radius = 2147483647)
    tensor<int> kernel = tensor.zeros<int>([3, 3], gpu = 0)
    kernel[1, 1] = 1
    tensor<uint8> filtered = try vision.filter(impulse, kernel)
    print(blurred[0, 1, 1].item())
    print(expanded[0, 0, 0].item())
    print(contracted[0, 1, 1].item())
    print(huge_blurred[0, 1, 1].item())
    print(huge_expanded[0, 0, 0].item())
    print(huge_contracted[0, 1, 1].item())
    print(filtered[0, 1, 1].item())
    return 0

auto result = run()
match result
    int
        int ignored = result
    error problem
        print(problem)
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

tensor<uint8> pixels = tensor.ones<uint8>([1, 2, 2], gpu = 0)
tensor<int> kernel = tensor.ones<int>([1, 1])
tensor<uint8> | error output = vision.filter(pixels, kernel)
match output
    tensor<uint8> value
        print(value.shape()[0])
    error problem
        print(problem)
QUI
expect_device_failure "$TMP/filter-device-mismatch.qui" "image filter tensors must be on the same device"

cat > "$TMP/image-write-gpu.qui" <<'QUI'
tensor<uint8> image_gpu = tensor.ones<uint8>([1, 1, 1], gpu = 0)
auto written = image.write("should-not-exist.png", image_gpu)
match written
    void
        print("unexpected success")
    error problem
        print(problem)
QUI

write_output="$(cd "$TMP" && "$QUIDRA" run "$TMP/image-write-gpu.qui")"
if [[ "$write_output" != "image.write is not supported on gpu(0)" ]]; then
    echo "unexpected GPU image.write result: $write_output" >&2
    exit 1
fi
if [[ -e "$TMP/should-not-exist.png" ]]; then
    echo "GPU image.write unexpectedly wrote a CPU-fallback file" >&2
    exit 1
fi

echo "vision device contracts: ok"
