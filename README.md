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

vision v0.1.0 supports Quidra `>=0.2.0 <0.3.0`.

With Quidra v0.2.0, install the exact released source:

```sh
git clone --depth 1 --branch v0.1.0 https://github.com/quidra-lang/vision.git
cd vision
quidra package install . --name vision
```

Quidra versions that provide the release-aware short package CLI can install the
same immutable release directly:

```sh
quidra install vision@0.1.0
```

Then import it normally:

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

## Device placement

Vision uses Quidra's tensor placement semantics directly. Image decoding through
the standard `image.read` API produces a CPU tensor by default. Moving image
data to a GPU is always an explicit caller action:

```quidra
tensor<uint8> image_cpu = try image.read("input.png")
tensor<uint8> image_gpu = image_cpu.gpu(0)
```

Vision never moves an input to CPU or GPU implicitly. A GPU-capable Vision
operation must execute on the input device and return its result on that same
device. If a Vision operation has no implementation for the active GPU backend,
it fails explicitly rather than iterating over hidden CPU storage or returning a
CPU result. The public Vision API is vendor-independent; backend selection is an
implementation detail of Quidra/Vision.

The released package dependency remains tied only to released Quidra versions.
During development, CI additionally builds the current Quidra `feature` branch
and checks the GPU placement/transfer contracts without changing
`requires.quidra` to an unreleased branch.

## Example

See [`examples/process.qui`](examples/process.qui).

## Development

See [`docs/development.md`](docs/development.md) for the canonical main/develop and release procedure.

## License

MIT
