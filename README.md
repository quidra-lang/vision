# vision

`vision` is Quidra's first-party tensor image-processing package. It works
directly on rank-3 CHW tensors; file I/O is provided by the standard `image`
namespace, so there is no separate Image wrapper or conversion layer.

Geometry operations and morphology preserve the input tensor dtype. Operations
whose semantics are currently defined in the 8-bit image domain (`grayscale`,
`threshold`, `blur`, and `filter`) explicitly use `tensor<uint8>`.

Invalid shapes or parameters are returned as `error`; they are never silently
reinterpreted. This includes zero-size resize targets, out-of-bounds crops,
unsupported grayscale channel counts, negative window radii, invalid filter
kernels, and a zero filter divisor.

## Install

Clone this repository and install it with the Quidra package command:

```sh
quidra package install . --name vision
```

Then import it normally:

```quidra
import vision

int | error transform(tensor<uint8> pixels)
    tensor<uint8> resized = try vision.resize(
        pixels, height = 256, width = 256
    )
    auto written = image.write("output.png", resized)
    match written
        void
            return 0
        error problem
            return problem

tensor<uint8> | error loaded = image.read("input.png")
match loaded
    tensor<uint8> pixels
        auto result = transform(pixels)
        match result
            int
                print("saved")
            error problem
                print(problem)
    error problem
        print(problem)
```

The explicit expected type above says that this pipeline accepts an 8-bit
image. If the decoded file has another representable dtype, `image.read`
returns `error` rather than silently converting it. Use `auto` plus exhaustive
matching only when the source dtype is genuinely unknown.

For development, place the repository in a directory listed by
`QUIDRA_PACKAGE_PATH` instead of installing it.

## Data layout

Images are rank-3 CHW tensors: channel, height, width. Grayscale images have one
channel; color images normally have three or four channels.

`vision` does not normalize or reinterpret sample values. Dtype-preserving
operations return the same `tensor<T>` element type they receive.

## API

| Operation | Signature |
| --- | --- |
| `crop` | `crop<T>(pixels, top, left, height, width) -> tensor<T> | error` |
| `resize` | `resize<T>(pixels, height, width) -> tensor<T> | error` |
| `flip_horizontal` | `flip_horizontal<T>(pixels) -> tensor<T> | error` |
| `flip_vertical` | `flip_vertical<T>(pixels) -> tensor<T> | error` |
| `rotate90` | `rotate90<T>(pixels) -> tensor<T> | error` |
| `rotate180` | `rotate180<T>(pixels) -> tensor<T> | error` |
| `rotate270` | `rotate270<T>(pixels) -> tensor<T> | error` |
| `grayscale` | `grayscale(tensor<uint8>) -> tensor<uint8> | error` |
| `threshold` | `threshold(tensor<uint8>, cutoff, low = uint8(0), high = uint8(255)) -> tensor<uint8> | error` |
| `blur` | `blur(tensor<uint8>, radius = 1) -> tensor<uint8> | error` |
| `filter` | `filter(tensor<uint8>, kernel, divisor = 1, offset = 0) -> tensor<uint8> | error` |
| `dilate` | `dilate<T>(pixels, radius = 1) -> tensor<T> | error` |
| `erode` | `erode<T>(pixels, radius = 1) -> tensor<T> | error` |

`resize` uses nearest-neighbor sampling. `rotate90` turns clockwise and
`rotate270` turns counter-clockwise; both exchange height and width. These
operations only relocate samples and therefore preserve dtype exactly.

`grayscale` accepts one, three, or four channels and returns one channel. For
three- or four-channel input it combines RGB using the coefficients
`0.299 * R + 0.587 * G + 0.114 * B` in floating-point and rounds only the final
luminance to `uint8`; an alpha channel is intentionally ignored rather than
silently mixed into luminance.

`threshold` writes `high` where the value is greater than or equal to `cutoff`
and `low` elsewhere. `filter` accepts a non-empty rank-2 integer kernel, divides
the accumulated sum by a nonzero `divisor`, adds `offset`, and clamps the result
to the `uint8` range. `blur` averages the window. `dilate` takes the maximum and
`erode` takes the minimum while preserving the source dtype. Every window
operation shrinks the window at the border rather than inventing padded values,
and `crop` requires the requested rectangle to lie inside the image.

## Example

See [`examples/process.qui`](examples/process.qui).

## License

MIT
