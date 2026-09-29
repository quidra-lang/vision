#!/usr/bin/env bash
set -euo pipefail

QUIDRA="$1"
REPOSITORY_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PACKAGE_ROOT="$(dirname "$REPOSITORY_ROOT")"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

cat > "$TMP/geometry-autograd.qui" <<'QUI'
import vision

int | error run()
    tensor<float32> crop_source = tensor.ones<float32>([2, 1, 2, 3]).track()
    tensor<float32> cropped = try vision.crop(
        crop_source, top = 0, left = 1, height = 2, width = 2
    )
    cropped.mean().backward(&crop_source)
    print(cropped.shape()[0] == 2)
    print(cropped.shape()[1] == 1)
    print(cropped.shape()[2] == 2)
    print(cropped.shape()[3] == 2)
    print(crop_source.grad.shape()[0] == 2)
    print(crop_source.grad[0, 0, 0, 0].item() == float32(0))
    float32 crop_grad = crop_source.grad[0, 0, 0, 1].item()
    print(crop_grad > float32(0.1249) and crop_grad < float32(0.1251))

    tensor<float32> resize_source = tensor.ones<float32>([1, 1, 2, 2]).track()
    tensor<float32> resized = try vision.resize(resize_source, height = 4, width = 4)
    resized.mean().backward(&resize_source)
    print(resized.shape()[2] == 4 and resized.shape()[3] == 4)
    float32 resize_grad = resize_source.grad[0, 0, 0, 0].item()
    print(resize_grad > float32(0.2499) and resize_grad < float32(0.2501))

    tensor<float32> horizontal_source = tensor.ones<float32>([1, 1, 2, 3]).track()
    tensor<float32> horizontal = try vision.flip_horizontal(horizontal_source)
    horizontal.mean().backward(&horizontal_source)
    print(horizontal_source.grad[0, 0, 0, 0].item() > float32(0))

    tensor<float32> vertical_source = tensor.ones<float32>([1, 1, 2, 3]).track()
    tensor<float32> vertical = try vision.flip_vertical(vertical_source)
    vertical.mean().backward(&vertical_source)
    print(vertical_source.grad[0, 0, 0, 0].item() > float32(0))

    tensor<float32> rotate90_source = tensor.ones<float32>([1, 1, 2, 3]).track()
    tensor<float32> turned90 = try vision.rotate90(rotate90_source)
    turned90.mean().backward(&rotate90_source)
    print(turned90.shape()[2] == 3 and turned90.shape()[3] == 2)
    print(rotate90_source.grad[0, 0, 0, 0].item() > float32(0))

    tensor<float32> rotate180_source = tensor.ones<float32>([1, 1, 2, 3]).track()
    tensor<float32> turned180 = try vision.rotate180(rotate180_source)
    turned180.mean().backward(&rotate180_source)
    print(rotate180_source.grad[0, 0, 0, 0].item() > float32(0))

    tensor<float32> rotate270_source = tensor.ones<float32>([1, 1, 2, 3]).track()
    tensor<float32> turned270 = try vision.rotate270(rotate270_source)
    turned270.mean().backward(&rotate270_source)
    print(turned270.shape()[2] == 3 and turned270.shape()[3] == 2)
    print(rotate270_source.grad[0, 0, 0, 0].item() > float32(0))
    tensor<float32> high_crop_source = tensor.ones<float32>([1, 1, 1, 1]).track()
    tensor<float32> high_crop = try vision.crop(
        high_crop_source, top = 0, left = 0, height = 1, width = 1
    )
    (high_crop * high_crop).mean().backward(&high_crop_source, track = true)
    tensor<float32> high_crop_first = high_crop_source.grad
    print(high_crop_first.is_tracked())
    high_crop_first.mean().backward(&high_crop_source)
    float32 high_crop_total = high_crop_source.grad.untrack()[0, 0, 0, 0].item()
    print(high_crop_total > float32(3.99) and high_crop_total < float32(4.01))

    tensor<float32> high_resize_source = tensor.ones<float32>([1, 1, 1, 1]).track()
    tensor<float32> high_resize = try vision.resize(high_resize_source, height = 2, width = 2)
    (high_resize * high_resize).mean().backward(&high_resize_source, track = true)
    tensor<float32> high_resize_first = high_resize_source.grad
    print(high_resize_first.is_tracked())
    high_resize_first.mean().backward(&high_resize_source)
    float32 high_resize_total = high_resize_source.grad.untrack()[0, 0, 0, 0].item()
    print(high_resize_total > float32(3.99) and high_resize_total < float32(4.01))

    tensor<float32> high_horizontal_source = tensor.ones<float32>([1, 1, 1, 1]).track()
    tensor<float32> high_horizontal = try vision.flip_horizontal(high_horizontal_source)
    (high_horizontal * high_horizontal).mean().backward(&high_horizontal_source, track = true)
    tensor<float32> high_horizontal_first = high_horizontal_source.grad
    print(high_horizontal_first.is_tracked())
    high_horizontal_first.mean().backward(&high_horizontal_source)
    float32 high_horizontal_total = high_horizontal_source.grad.untrack()[0, 0, 0, 0].item()
    print(high_horizontal_total > float32(3.99) and high_horizontal_total < float32(4.01))

    tensor<float32> high_vertical_source = tensor.ones<float32>([1, 1, 1, 1]).track()
    tensor<float32> high_vertical = try vision.flip_vertical(high_vertical_source)
    (high_vertical * high_vertical).mean().backward(&high_vertical_source, track = true)
    tensor<float32> high_vertical_first = high_vertical_source.grad
    print(high_vertical_first.is_tracked())
    high_vertical_first.mean().backward(&high_vertical_source)
    float32 high_vertical_total = high_vertical_source.grad.untrack()[0, 0, 0, 0].item()
    print(high_vertical_total > float32(3.99) and high_vertical_total < float32(4.01))

    tensor<float32> high_rotate90_source = tensor.ones<float32>([1, 1, 1, 1]).track()
    tensor<float32> high_rotate90 = try vision.rotate90(high_rotate90_source)
    (high_rotate90 * high_rotate90).mean().backward(&high_rotate90_source, track = true)
    tensor<float32> high_rotate90_first = high_rotate90_source.grad
    print(high_rotate90_first.is_tracked())
    high_rotate90_first.mean().backward(&high_rotate90_source)
    float32 high_rotate90_total = high_rotate90_source.grad.untrack()[0, 0, 0, 0].item()
    print(high_rotate90_total > float32(3.99) and high_rotate90_total < float32(4.01))

    tensor<float32> high_rotate180_source = tensor.ones<float32>([1, 1, 1, 1]).track()
    tensor<float32> high_rotate180 = try vision.rotate180(high_rotate180_source)
    (high_rotate180 * high_rotate180).mean().backward(&high_rotate180_source, track = true)
    tensor<float32> high_rotate180_first = high_rotate180_source.grad
    print(high_rotate180_first.is_tracked())
    high_rotate180_first.mean().backward(&high_rotate180_source)
    float32 high_rotate180_total = high_rotate180_source.grad.untrack()[0, 0, 0, 0].item()
    print(high_rotate180_total > float32(3.99) and high_rotate180_total < float32(4.01))

    tensor<float32> high_rotate270_source = tensor.ones<float32>([1, 1, 1, 1]).track()
    tensor<float32> high_rotate270 = try vision.rotate270(high_rotate270_source)
    (high_rotate270 * high_rotate270).mean().backward(&high_rotate270_source, track = true)
    tensor<float32> high_rotate270_first = high_rotate270_source.grad
    print(high_rotate270_first.is_tracked())
    high_rotate270_first.mean().backward(&high_rotate270_source)
    float32 high_rotate270_total = high_rotate270_source.grad.untrack()[0, 0, 0, 0].item()
    print(high_rotate270_total > float32(3.99) and high_rotate270_total < float32(4.01))

    return 0

auto | error result = run()
match result
    int
        int ignored = result
    error problem
        print(problem)
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

int | error run()
    tensor<float32> gray_source = tensor.ones<float32>([1, 3, 2, 2]).track()
    tensor<float32> gray = try vision.grayscale(gray_source)
    gray.mean().backward(&gray_source)
    print(gray.is_tracked())
    print(gray_source.has_grad())

    tensor<float32> blur_source = tensor.ones<float32>([1, 1, 3, 3]).track()
    tensor<float32> blurred = try vision.blur(blur_source, radius = 1)
    blurred.mean().backward(&blur_source)
    print(blurred.is_tracked())
    print(blur_source.has_grad())

    tensor<float32> filter_source = tensor.ones<float32>([1, 1, 3, 3]).track()
    tensor<float32> kernel = tensor.ones<float32>([3, 3])
    tensor<float32> filtered = try vision.filter(
        filter_source, kernel, float32(1), float32(0)
    )
    filtered.mean().backward(&filter_source)
    print(filtered.is_tracked())
    print(filter_source.has_grad())

    tensor<float32> high_gray_source = tensor.ones<float32>([1, 3, 1, 1]).track()
    tensor<float32> high_gray = try vision.grayscale(high_gray_source)
    (high_gray * high_gray).mean().backward(&high_gray_source, track = true)
    tensor<float32> high_gray_first = high_gray_source.grad
    print(high_gray_first.is_tracked())
    high_gray_first.mean().backward(&high_gray_source)
    print(high_gray_source.grad.untrack()[0, 0, 0, 0].item() > float32(0))

    tensor<float32> high_blur_source = tensor.ones<float32>([1, 1, 2, 2]).track()
    tensor<float32> high_blur = try vision.blur(high_blur_source, radius = 1)
    (high_blur * high_blur).mean().backward(&high_blur_source, track = true)
    tensor<float32> high_blur_first = high_blur_source.grad
    print(high_blur_first.is_tracked())
    high_blur_first.mean().backward(&high_blur_source)
    print(high_blur_source.grad.untrack()[0, 0, 0, 0].item() > float32(0))

    tensor<float32> high_filter_source = tensor.ones<float32>([1, 1, 2, 2]).track()
    tensor<float32> high_kernel = tensor.ones<float32>([1, 1])
    tensor<float32> high_filter = try vision.filter(high_filter_source, high_kernel)
    (high_filter * high_filter).mean().backward(&high_filter_source, track = true)
    tensor<float32> high_filter_first = high_filter_source.grad
    print(high_filter_first.is_tracked())
    high_filter_first.mean().backward(&high_filter_source)
    print(high_filter_source.grad.untrack()[0, 0, 0, 0].item() > float32(0))

    tensor<float32> dilate_source = tensor.ones<float32>([1, 1, 2, 2]).track()
    tensor<float32> dilated = try vision.dilate(dilate_source, radius = 1)
    dilated.mean().backward(&dilate_source)
    print(dilated.is_tracked())
    print(dilate_source.has_grad())

    tensor<float32> erode_source = tensor.ones<float32>([1, 1, 2, 2]).track()
    tensor<float32> eroded = try vision.erode(erode_source, radius = 1)
    eroded.mean().backward(&erode_source)
    print(eroded.is_tracked())
    print(erode_source.has_grad())

    tensor<float32> high_dilate_values = tensor.zeros<float32>([1, 1, 2, 2])
    high_dilate_values[0, 0, 0, 0] = float32(1)
    high_dilate_values[0, 0, 0, 1] = float32(2)
    high_dilate_values[0, 0, 1, 0] = float32(3)
    high_dilate_values[0, 0, 1, 1] = float32(4)
    tensor<float32> high_dilate_source = high_dilate_values.track()
    tensor<float32> high_dilate = try vision.dilate(high_dilate_source, radius = 1)
    (high_dilate * high_dilate).mean().backward(&high_dilate_source, track = true)
    tensor<float32> high_dilate_first = high_dilate_source.grad
    print(high_dilate_first.is_tracked())
    high_dilate_first.mean().backward(&high_dilate_source)
    print(high_dilate_source.grad.untrack()[0, 0, 1, 1].item() > float32(0))

    tensor<float32> high_erode_values = tensor.zeros<float32>([1, 1, 2, 2])
    high_erode_values[0, 0, 0, 0] = float32(1)
    high_erode_values[0, 0, 0, 1] = float32(2)
    high_erode_values[0, 0, 1, 0] = float32(3)
    high_erode_values[0, 0, 1, 1] = float32(4)
    tensor<float32> high_erode_source = high_erode_values.track()
    tensor<float32> high_erode = try vision.erode(high_erode_source, radius = 1)
    (high_erode * high_erode).mean().backward(&high_erode_source, track = true)
    tensor<float32> high_erode_first = high_erode_source.grad
    print(high_erode_first.is_tracked())
    high_erode_first.mean().backward(&high_erode_source)
    print(high_erode_source.grad.untrack()[0, 0, 0, 0].item() > float32(0))

    tensor<float32> threshold_source = tensor.ones<float32>([1, 1, 2, 2]).track()
    tensor<float32> | error thresholded = vision.threshold(
        threshold_source, float32(0.5), float32(0), float32(1)
    )
    bool threshold_rejected = false
    match thresholded
        tensor<float32>
            threshold_rejected = false
        error
            threshold_rejected = true
    print(threshold_rejected)
    return 0

auto | error result = run()
match result
    int
        int ignored = result
    error problem
        print(problem)
QUI

floating_output="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/floating-vision-autograd.qui")"
floating_expected="$(printf 'true\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue\ntrue')"
if [[ "$floating_output" != "$floating_expected" ]]; then
    echo "unexpected floating Vision autograd output:" >&2
    printf '%s\n' "$floating_output" >&2
    exit 1
fi

echo "vision develop autograd contracts: ok"
