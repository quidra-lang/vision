#!/usr/bin/env bash
set -euo pipefail

QUIDRA="$1"
REPOSITORY_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PACKAGE_ROOT="$(dirname "$REPOSITORY_ROOT"):$REPOSITORY_ROOT"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

cat > "$TMP/geometry-autograd.qui" <<'QUI'
import vision
import math

int | error run()
    tensor<real32> crop_source = tensor.ones<real32>([2, 1, 2, 3]).track()
    tensor<real32> cropped = try vision.crop(
        crop_source, top = 0, left = 1, height = 2, width = 2
    )
    math.mean(cropped).backward(&crop_source)
    print(cropped.shape()[0] == 2)
    print(NL)
    print(cropped.shape()[1] == 1)
    print(NL)
    print(cropped.shape()[2] == 2)
    print(NL)
    print(cropped.shape()[3] == 2)
    print(NL)
    print(crop_source.grad.shape()[0] == 2)
    print(NL)
    print(crop_source.grad[0, 0, 0, 0].item() == real32(0))
    print(NL)
    real32 crop_grad = crop_source.grad[0, 0, 0, 1].item()
    print(crop_grad > real32(0.1249) and crop_grad < real32(0.1251))
    print(NL)

    tensor<real32> resize_source = tensor.ones<real32>([1, 1, 2, 2]).track()
    tensor<real32> resized = try vision.resize(resize_source, height = 4, width = 4)
    math.mean(resized).backward(&resize_source)
    print(resized.shape()[2] == 4 and resized.shape()[3] == 4)
    print(NL)
    real32 resize_grad = resize_source.grad[0, 0, 0, 0].item()
    print(resize_grad > real32(0.2499) and resize_grad < real32(0.2501))
    print(NL)

    tensor<real32> horizontal_source = tensor.ones<real32>([1, 1, 2, 3]).track()
    tensor<real32> horizontal = try vision.flip_horizontal(horizontal_source)
    math.mean(horizontal).backward(&horizontal_source)
    print(horizontal_source.grad[0, 0, 0, 0].item() > real32(0))
    print(NL)

    tensor<real32> vertical_source = tensor.ones<real32>([1, 1, 2, 3]).track()
    tensor<real32> vertical = try vision.flip_vertical(vertical_source)
    math.mean(vertical).backward(&vertical_source)
    print(vertical_source.grad[0, 0, 0, 0].item() > real32(0))
    print(NL)

    tensor<real32> rotate90_source = tensor.ones<real32>([1, 1, 2, 3]).track()
    tensor<real32> turned90 = try vision.rotate90(rotate90_source)
    math.mean(turned90).backward(&rotate90_source)
    print(turned90.shape()[2] == 3 and turned90.shape()[3] == 2)
    print(NL)
    print(rotate90_source.grad[0, 0, 0, 0].item() > real32(0))
    print(NL)

    tensor<real32> rotate180_source = tensor.ones<real32>([1, 1, 2, 3]).track()
    tensor<real32> turned180 = try vision.rotate180(rotate180_source)
    math.mean(turned180).backward(&rotate180_source)
    print(rotate180_source.grad[0, 0, 0, 0].item() > real32(0))
    print(NL)

    tensor<real32> rotate270_source = tensor.ones<real32>([1, 1, 2, 3]).track()
    tensor<real32> turned270 = try vision.rotate270(rotate270_source)
    math.mean(turned270).backward(&rotate270_source)
    print(turned270.shape()[2] == 3 and turned270.shape()[3] == 2)
    print(NL)
    print(rotate270_source.grad[0, 0, 0, 0].item() > real32(0))
    print(NL)
    tensor<real32> high_crop_source = tensor.ones<real32>([1, 1, 1, 1]).track()
    tensor<real32> high_crop = try vision.crop(
        high_crop_source, top = 0, left = 0, height = 1, width = 1
    )
    math.mean((high_crop * high_crop)).backward(&high_crop_source, track = true)
    tensor<real32> high_crop_first = high_crop_source.grad
    print(high_crop_first.is_tracked())
    print(NL)
    math.mean(high_crop_first).backward(&high_crop_source)
    real32 high_crop_total = high_crop_source.grad.untrack()[0, 0, 0, 0].item()
    print(high_crop_total > real32(3.99) and high_crop_total < real32(4.01))
    print(NL)

    tensor<real32> high_resize_source = tensor.ones<real32>([1, 1, 1, 1]).track()
    tensor<real32> high_resize = try vision.resize(high_resize_source, height = 2, width = 2)
    math.mean((high_resize * high_resize)).backward(&high_resize_source, track = true)
    tensor<real32> high_resize_first = high_resize_source.grad
    print(high_resize_first.is_tracked())
    print(NL)
    math.mean(high_resize_first).backward(&high_resize_source)
    real32 high_resize_total = high_resize_source.grad.untrack()[0, 0, 0, 0].item()
    print(high_resize_total > real32(3.99) and high_resize_total < real32(4.01))
    print(NL)

    tensor<real32> high_horizontal_source = tensor.ones<real32>([1, 1, 1, 1]).track()
    tensor<real32> high_horizontal = try vision.flip_horizontal(high_horizontal_source)
    math.mean((high_horizontal * high_horizontal)).backward(&high_horizontal_source, track = true)
    tensor<real32> high_horizontal_first = high_horizontal_source.grad
    print(high_horizontal_first.is_tracked())
    print(NL)
    math.mean(high_horizontal_first).backward(&high_horizontal_source)
    real32 high_horizontal_total = high_horizontal_source.grad.untrack()[0, 0, 0, 0].item()
    print(high_horizontal_total > real32(3.99) and high_horizontal_total < real32(4.01))
    print(NL)

    tensor<real32> high_vertical_source = tensor.ones<real32>([1, 1, 1, 1]).track()
    tensor<real32> high_vertical = try vision.flip_vertical(high_vertical_source)
    math.mean((high_vertical * high_vertical)).backward(&high_vertical_source, track = true)
    tensor<real32> high_vertical_first = high_vertical_source.grad
    print(high_vertical_first.is_tracked())
    print(NL)
    math.mean(high_vertical_first).backward(&high_vertical_source)
    real32 high_vertical_total = high_vertical_source.grad.untrack()[0, 0, 0, 0].item()
    print(high_vertical_total > real32(3.99) and high_vertical_total < real32(4.01))
    print(NL)

    tensor<real32> high_rotate90_source = tensor.ones<real32>([1, 1, 1, 1]).track()
    tensor<real32> high_rotate90 = try vision.rotate90(high_rotate90_source)
    math.mean((high_rotate90 * high_rotate90)).backward(&high_rotate90_source, track = true)
    tensor<real32> high_rotate90_first = high_rotate90_source.grad
    print(high_rotate90_first.is_tracked())
    print(NL)
    math.mean(high_rotate90_first).backward(&high_rotate90_source)
    real32 high_rotate90_total = high_rotate90_source.grad.untrack()[0, 0, 0, 0].item()
    print(high_rotate90_total > real32(3.99) and high_rotate90_total < real32(4.01))
    print(NL)

    tensor<real32> high_rotate180_source = tensor.ones<real32>([1, 1, 1, 1]).track()
    tensor<real32> high_rotate180 = try vision.rotate180(high_rotate180_source)
    math.mean((high_rotate180 * high_rotate180)).backward(&high_rotate180_source, track = true)
    tensor<real32> high_rotate180_first = high_rotate180_source.grad
    print(high_rotate180_first.is_tracked())
    print(NL)
    math.mean(high_rotate180_first).backward(&high_rotate180_source)
    real32 high_rotate180_total = high_rotate180_source.grad.untrack()[0, 0, 0, 0].item()
    print(high_rotate180_total > real32(3.99) and high_rotate180_total < real32(4.01))
    print(NL)

    tensor<real32> high_rotate270_source = tensor.ones<real32>([1, 1, 1, 1]).track()
    tensor<real32> high_rotate270 = try vision.rotate270(high_rotate270_source)
    math.mean((high_rotate270 * high_rotate270)).backward(&high_rotate270_source, track = true)
    tensor<real32> high_rotate270_first = high_rotate270_source.grad
    print(high_rotate270_first.is_tracked())
    print(NL)
    math.mean(high_rotate270_first).backward(&high_rotate270_source)
    real32 high_rotate270_total = high_rotate270_source.grad.untrack()[0, 0, 0, 0].item()
    print(high_rotate270_total > real32(3.99) and high_rotate270_total < real32(4.01))
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

output="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/geometry-autograd.qui")"
expected="$(printf 'true\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue')"
if [[ "$output" != "$expected" ]]; then
    echo "unexpected Vision geometry autograd output:" >&2
    printf '%s\n' "$output" >&2
    exit 1
fi

cat > "$TMP/floating-vision-autograd.qui" <<'QUI'
import vision
import math

int | error run()
    tensor<real32> gray_source = tensor.ones<real32>([1, 3, 2, 2]).track()
    tensor<real32> gray = try vision.grayscale(gray_source)
    math.mean(gray).backward(&gray_source)
    print(gray.is_tracked())
    print(NL)
    print(gray_source.has_grad())
    print(NL)

    tensor<real32> blur_source = tensor.ones<real32>([1, 1, 3, 3]).track()
    tensor<real32> blurred = try vision.blur(blur_source, radius = 1)
    math.mean(blurred).backward(&blur_source)
    print(blurred.is_tracked())
    print(NL)
    print(blur_source.has_grad())
    print(NL)

    tensor<real32> filter_source = tensor.ones<real32>([1, 1, 3, 3]).track()
    tensor<real32> kernel = tensor.ones<real32>([3, 3])
    tensor<real32> filtered = try vision.filter(
        filter_source, kernel, real32(1), real32(0)
    )
    math.mean(filtered).backward(&filter_source)
    print(filtered.is_tracked())
    print(NL)
    print(filter_source.has_grad())
    print(NL)

    tensor<real32> high_gray_source = tensor.ones<real32>([1, 3, 1, 1]).track()
    tensor<real32> high_gray = try vision.grayscale(high_gray_source)
    math.mean((high_gray * high_gray)).backward(&high_gray_source, track = true)
    tensor<real32> high_gray_first = high_gray_source.grad
    print(high_gray_first.is_tracked())
    print(NL)
    math.mean(high_gray_first).backward(&high_gray_source)
    print(high_gray_source.grad.untrack()[0, 0, 0, 0].item() > real32(0))
    print(NL)

    tensor<real32> high_blur_source = tensor.ones<real32>([1, 1, 2, 2]).track()
    tensor<real32> high_blur = try vision.blur(high_blur_source, radius = 1)
    math.mean((high_blur * high_blur)).backward(&high_blur_source, track = true)
    tensor<real32> high_blur_first = high_blur_source.grad
    print(high_blur_first.is_tracked())
    print(NL)
    math.mean(high_blur_first).backward(&high_blur_source)
    print(high_blur_source.grad.untrack()[0, 0, 0, 0].item() > real32(0))
    print(NL)

    tensor<real32> high_filter_source = tensor.ones<real32>([1, 1, 2, 2]).track()
    tensor<real32> high_kernel = tensor.ones<real32>([1, 1])
    tensor<real32> high_filter = try vision.filter(high_filter_source, high_kernel)
    math.mean((high_filter * high_filter)).backward(&high_filter_source, track = true)
    tensor<real32> high_filter_first = high_filter_source.grad
    print(high_filter_first.is_tracked())
    print(NL)
    math.mean(high_filter_first).backward(&high_filter_source)
    print(high_filter_source.grad.untrack()[0, 0, 0, 0].item() > real32(0))
    print(NL)

    tensor<real32> dilate_source = tensor.ones<real32>([1, 1, 2, 2]).track()
    tensor<real32> dilated = try vision.dilate(dilate_source, radius = 1)
    math.mean(dilated).backward(&dilate_source)
    print(dilated.is_tracked())
    print(NL)
    print(dilate_source.has_grad())
    print(NL)

    tensor<real32> erode_source = tensor.ones<real32>([1, 1, 2, 2]).track()
    tensor<real32> eroded = try vision.erode(erode_source, radius = 1)
    math.mean(eroded).backward(&erode_source)
    print(eroded.is_tracked())
    print(NL)
    print(erode_source.has_grad())
    print(NL)

    tensor<real32> high_dilate_values = tensor.zeros<real32>([1, 1, 2, 2])
    high_dilate_values[0, 0, 0, 0] = real32(1)
    high_dilate_values[0, 0, 0, 1] = real32(2)
    high_dilate_values[0, 0, 1, 0] = real32(3)
    high_dilate_values[0, 0, 1, 1] = real32(4)
    tensor<real32> high_dilate_source = high_dilate_values.track()
    tensor<real32> high_dilate = try vision.dilate(high_dilate_source, radius = 1)
    math.mean((high_dilate * high_dilate)).backward(&high_dilate_source, track = true)
    tensor<real32> high_dilate_first = high_dilate_source.grad
    print(high_dilate_first.is_tracked())
    print(NL)
    math.mean(high_dilate_first).backward(&high_dilate_source)
    print(high_dilate_source.grad.untrack()[0, 0, 1, 1].item() > real32(0))
    print(NL)

    tensor<real32> high_erode_values = tensor.zeros<real32>([1, 1, 2, 2])
    high_erode_values[0, 0, 0, 0] = real32(1)
    high_erode_values[0, 0, 0, 1] = real32(2)
    high_erode_values[0, 0, 1, 0] = real32(3)
    high_erode_values[0, 0, 1, 1] = real32(4)
    tensor<real32> high_erode_source = high_erode_values.track()
    tensor<real32> high_erode = try vision.erode(high_erode_source, radius = 1)
    math.mean((high_erode * high_erode)).backward(&high_erode_source, track = true)
    tensor<real32> high_erode_first = high_erode_source.grad
    print(high_erode_first.is_tracked())
    print(NL)
    math.mean(high_erode_first).backward(&high_erode_source)
    print(high_erode_source.grad.untrack()[0, 0, 0, 0].item() > real32(0))
    print(NL)

    tensor<real32> threshold_source = tensor.ones<real32>([1, 1, 2, 2]).track()
    tensor<real32> | error thresholded = vision.threshold(
        threshold_source, real32(0.5), real32(0), real32(1)
    )
    bool threshold_rejected = false
    match thresholded
        tensor<real32>
            threshold_rejected = false
        error
            threshold_rejected = true
    print(threshold_rejected)
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

floating_output="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/floating-vision-autograd.qui")"
floating_expected="$(printf 'true\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue')"
if [[ "$floating_output" != "$floating_expected" ]]; then
    echo "unexpected floating Vision autograd output:" >&2
    printf '%s\n' "$floating_output" >&2
    exit 1
fi

# The portable composition is vision.downsample_mean's path on backends without
# a Vision kernel (CUDA, HIP). Tests reach it through a copy of the package's
# internal module so it is checked against the native kernel here.
cp "$REPOSITORY_ROOT/internal.qui" "$TMP/vision_internal.qui"

cat > "$TMP/downsample-autograd.qui" <<'QUI'
import vision
import math
import composition = "./vision_internal.qui"

int differences(tensor<real32> left, tensor<real32> right)
    int[] shape = left.shape()
    int count = 0
    for n in range(shape[0])
        for c in range(shape[1])
            for y in range(shape[2])
                for x in range(shape[3])
                    if left[n, c, y, x].item() != right[n, c, y, x].item()
                        count += 1
    return count

tensor<real32> pattern(int batch, int channels, int height, int width)
    tensor<real32> values = tensor.zeros<real32>([batch, channels, height, width])
    int index = 0
    for n in range(batch)
        for c in range(channels)
            for y in range(height)
                for x in range(width)
                    values[n, c, y, x] = real32((index * 29 + 3) % 19) * real32(0.21) - real32(1.7)
                    index += 1
    return values

int | error run()
    // The operation is linear: each covered sample receives the upstream
    // gradient divided by factor * factor; uncovered samples receive zero.
    tensor<real32> source = pattern(2, 1, 5, 7).track()
    tensor<real32> reduced = try vision.downsample_mean(source, 2)
    print(reduced.is_tracked())
    print(NL)
    math.mean(reduced).backward(&source)
    // mean over 2 * 2 * 3 outputs, then / 4 per block sample
    real32 expected = real32(1) / real32(48)
    print(source.grad[1, 0, 3, 5].item() == expected)
    print(NL)
    print(source.grad[0, 0, 4, 0].item() == real32(0) and source.grad[0, 0, 0, 6].item() == real32(0))
    print(NL)

    // A tracked non-contiguous view is copied with one graph-preserving
    // gather and then uses the native kernel; values and gradients match the
    // contiguous input exactly.
    tensor<real32> values = pattern(1, 2, 7, 8)
    tensor<real32> transposed = values.transpose(2, 3).contiguous()
    tensor<real32> view_source = transposed.transpose(2, 3).track()
    tensor<real32> native_source = values.track()
    tensor<real32> portable = try vision.downsample_mean(view_source, 3)
    tensor<real32> native_result = try vision.downsample_mean(native_source, 3)
    print(view_source.is_contiguous() == false)
    print(NL)
    print(differences(portable.untrack(), native_result.untrack()))
    print(NL)
    tensor<real32> weights = pattern(1, 2, 2, 2)
    math.mean(portable * weights).backward(&view_source)
    math.mean(native_result * weights).backward(&native_source)
    print(differences(view_source.grad, native_source.grad))
    print(NL)

    // backward(track = true): the first gradient stays differentiable.
    // L = mean(y * y) over a 2 x 2 result of ones: dL/dx = y / 8 = 0.125,
    // and d mean(dL/dx) / dx = 1 / 128, accumulated onto the first gradient.
    tensor<real32> high_source = tensor.ones<real32>([1, 1, 4, 4]).track()
    tensor<real32> high = try vision.downsample_mean(high_source, 2)
    math.mean(high * high).backward(&high_source, track = true)
    tensor<real32> high_first = high_source.grad
    print(high_first.is_tracked())
    print(NL)
    print(high_first.untrack()[0, 0, 2, 1].item() == real32(0.125))
    print(NL)
    math.mean(high_first).backward(&high_source)
    print(high_source.grad.untrack()[0, 0, 3, 3].item() == real32(0.1328125))
    print(NL)

    // Third order alternates between the block mean and its adjoint.
    tensor<real32> third_source = tensor.ones<real32>([1, 1, 4, 4]).track()
    tensor<real32> third = try vision.downsample_mean(third_source, 2)
    math.mean(third * third * third).backward(&third_source, track = true)
    tensor<real32> third_first = third_source.grad
    math.mean(third_first * third_first).backward(&third_source, track = true)
    tensor<real32> third_second = third_source.grad
    print(third_second.is_tracked())
    print(NL)
    print(third_second.untrack()[0, 0, 0, 0].item() == real32(0.1962890625))
    print(NL)
    // With y = 1: d mean(3y^2/16 + 9y^3/1024) / dx = (6/16 + 27/1024) / 16 =
    // 411/16384, accumulated onto 3216/16384.
    math.mean(third_second).backward(&third_source)
    print(third_source.grad.untrack()[0, 0, 1, 3].item() == real32(0.22137451171875))
    print(NL)

    // The portable composition gives the native values and gradients, also
    // below a transpose and through backward(track = true).
    tensor<real32> native_high = pattern(2, 2, 9, 11).track()
    tensor<real32> portable_high = pattern(2, 2, 9, 11).track()
    tensor<real32> native_blocks = try vision.downsample_mean(native_high, 3)
    tensor<real32> portable_blocks = composition.block_mean<real32>(portable_high, 3)
    print(differences(native_blocks.untrack(), portable_blocks.untrack()))
    print(NL)
    tensor<real32> block_weights = pattern(2, 2, 3, 3)
    math.mean(native_blocks.transpose(2, 3) * block_weights * native_blocks.transpose(2, 3)).backward(&native_high, track = true)
    math.mean(portable_blocks.transpose(2, 3) * block_weights * portable_blocks.transpose(2, 3)).backward(&portable_high, track = true)
    tensor<real32> native_high_first = native_high.grad
    tensor<real32> portable_high_first = portable_high.grad
    print(differences(native_high_first.untrack(), portable_high_first.untrack()))
    print(NL)
    math.mean(native_high_first.transpose(2, 3) * native_high_first.transpose(2, 3)).backward(&native_high)
    math.mean(portable_high_first.transpose(2, 3) * portable_high_first.transpose(2, 3)).backward(&portable_high)
    print(differences(native_high.grad.untrack(), portable_high.grad.untrack()))
    print(NL)

    tensor<real64> wide_source = tensor.ones<real64>([1, 3, 3]).track()
    tensor<real64> wide = try vision.downsample_mean(wide_source, 2)
    math.mean(wide).backward(&wide_source)
    print(wide_source.grad[0, 1, 1].item() == 0.25 and wide_source.grad[0, 2, 2].item() == 0.0)
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

downsample_output="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/downsample-autograd.qui")"
downsample_expected="$(printf 'true\ntrue\ntrue\ntrue\n0\n0\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\n0\n0\n0\ntrue')"
if [[ "$downsample_output" != "$downsample_expected" ]]; then
    echo "unexpected Vision downsample_mean autograd output:" >&2
    printf '%s\n' "$downsample_output" >&2
    exit 1
fi

echo "vision develop autograd contracts: ok"
