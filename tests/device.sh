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

cat > "$TMP/vision-no-fallback.qui" <<'QUI'
import vision

tensor<uint8> input = tensor.ones<uint8>([1, 2, 2], gpu = 0)
tensor<uint8> | error transformed = vision.flip_horizontal(input)
match transformed
    tensor<uint8> output
        print(output.shape()[0])
    error problem
        print(problem)
QUI

set +e
QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/vision-no-fallback.qui" >"$TMP/vision.out" 2>"$TMP/vision.err"
status=$?
set -e
if [[ $status -ne 101 ]]; then
    echo "expected unsupported GPU Vision operation to fail with status 101, got $status" >&2
    cat "$TMP/vision.out" >&2 || true
    cat "$TMP/vision.err" >&2 || true
    exit 1
fi
if ! grep -Fq "tensor.item is not supported on gpu(0)" "$TMP/vision.err"; then
    echo "Vision GPU operation did not fail at the explicit unsupported boundary" >&2
    cat "$TMP/vision.err" >&2
    exit 1
fi
if [[ -s "$TMP/vision.out" ]]; then
    echo "Vision unexpectedly produced a CPU result for a GPU input" >&2
    cat "$TMP/vision.out" >&2
    exit 1
fi

cat > "$TMP/image-write-gpu.qui" <<'QUI'
tensor<uint8> input = tensor.ones<uint8>([1, 1, 1], gpu = 0)
auto written = image.write("should-not-exist.png", input)
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
