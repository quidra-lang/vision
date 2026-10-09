#!/usr/bin/env bash
set -euo pipefail

QUIDRA="${1:?usage: aot.sh /path/to/quidra}"
REPOSITORY_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PACKAGE_ROOT="$(dirname "$REPOSITORY_ROOT"):$REPOSITORY_ROOT"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

cat > "$TMP/vision-aot.qui" <<QUI
import vision

int | error run()
    tensor<nat8> pixels = tensor.zeros<nat8>([3, 1, 1])
    pixels[0, 0, 0] = nat8(11)
    pixels[1, 0, 0] = nat8(22)
    pixels[2, 0, 0] = nat8(33)
    try vision.write("$TMP/roundtrip.png", pixels)
    tensor<nat8> decoded = try vision.read<nat8>(
        "$TMP/roundtrip.png",
        channels = 3
    )
    print(decoded.shape()[0] == 3)
    print(NL)
    print(decoded.shape()[1] == 1)
    print(NL)
    print(decoded.shape()[2] == 1)
    print(NL)
    print(decoded[0, 0, 0].item() == nat8(11))
    print(NL)
    print(decoded[1, 0, 0].item() == nat8(22))
    print(NL)
    print(decoded[2, 0, 0].item() == nat8(33))
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

OUTPUT="$TMP/vision-aot"
QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" build "$TMP/vision-aot.qui" -o "$OUTPUT"
actual="$("$OUTPUT")"
expected="$(printf 'true\n%.0s' {1..6})"
if [[ "$actual" != "$expected" ]]; then
    echo "unexpected Vision AOT output:" >&2
    printf '%s\n' "$actual" >&2
    exit 1
fi

repl_output="$(
    QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" repl < "$TMP/vision-aot.qui"
)"
true_lines="$(grep -Fxc "true" <<< "$repl_output" || true)"
if [[ "$true_lines" -lt 6 ]]; then
    echo "Vision package-native REPL/JIT load did not execute expected codec operations:" >&2
    printf '%s\n' "$repl_output" >&2
    exit 1
fi

echo "vision AOT and REPL/JIT integration: ok"
