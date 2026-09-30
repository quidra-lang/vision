# Quidra Vision

Quidra Vision is Quidra's first-party tensor image-processing package, imported
as `vision`. It works directly on tensors with rank >= 3 and trailing
`(..., C, H, W)` dimensions. Image file I/O is owned by Vision itself through
`vision.read` / `vision.write`; Core has no image codec or image namespace.

Geometry operations and morphology preserve the input tensor element type. `grayscale`, `blur`, and `filter` use one public name across the `tensor<uint8>` image path and differentiable floating paths; Quidra's generic specialization resolves the dtype-specific implementation statically. `threshold` uses the same public name across the `tensor<uint8>` and floating paths. Because thresholding is discontinuous, tracked floating input is rejected rather than implicitly detached; untracked floating input is processed normally without creating an autograd graph.

Invalid shapes or parameters are returned as `error`; they are never silently
reinterpreted. This includes zero-size resize targets, out-of-bounds crops,
unsupported grayscale channel counts, negative window radii, invalid filter
kernels, and a zero filter divisor.

## Install

Install a released tag; `quidra.package` declares the Quidra range that tag
supports.

```sh
git clone --depth 1 --branch vX.Y.Z https://github.com/quidra-lang/vision.git
cd vision
quidra install .
```

Quidra versions that provide the release-aware short package CLI can install the
same immutable release directly:

```sh
quidra install quidra-vision@X.Y.Z
```

The package-manager identity is `quidra-vision`; the Quidra source import
identifier remains `vision`. This distinction keeps installation names globally
recognizable without making source imports longer.

Then import it normally:

```quidra
import vision
```

## API

| Operation | Signature |
| --- | --- |
| `read` | `read<T: numeric>(string path, int channels = 0) -> tensor<T> \| error` |
| `write` | `write<T: numeric>(string path, tensor<T> pixels, int quality = 90) -> void \| error` |
| `crop` | `crop<T>(tensor<T> pixels, int top, int left, int height, int width) -> tensor<T> \| error` |
| `resize` | `resize<T>(tensor<T> pixels, int height, int width) -> tensor<T> \| error` |
| `flip_horizontal` | `flip_horizontal<T>(tensor<T> pixels) -> tensor<T> \| error` |
| `flip_vertical` | `flip_vertical<T>(tensor<T> pixels) -> tensor<T> \| error` |
| `rotate90` | `rotate90<T>(tensor<T> pixels) -> tensor<T> \| error` |
| `rotate180` | `rotate180<T>(tensor<T> pixels) -> tensor<T> \| error` |
| `rotate270` | `rotate270<T>(tensor<T> pixels) -> tensor<T> \| error` |
| `grayscale` | `grayscale(tensor<uint8> pixels) -> tensor<uint8> \| error`; `grayscale<T: floating>(tensor<T> pixels) -> tensor<T> \| error` |
| `threshold` | `threshold(tensor<uint8> pixels, uint8 cutoff, uint8 low = uint8(0), uint8 high = uint8(255)) -> tensor<uint8> \| error`; `threshold<T: floating>(tensor<T> pixels, T cutoff, T low = 0.0, T high = 1.0) -> tensor<T> \| error` |
| `blur` | `blur(tensor<uint8> pixels, int radius = 1) -> tensor<uint8> \| error`; `blur<T: floating>(tensor<T> pixels, int radius = 1) -> tensor<T> \| error` |
| `filter` | `filter(tensor<uint8> pixels, tensor<int> kernel, int divisor = 1, int offset = 0) -> tensor<uint8> \| error`; `filter<T: floating, K: floating>(tensor<T> pixels, tensor<K> kernel, K divisor = 1.0, K offset = 0.0) -> tensor<T> \| error` |
| `dilate` | `dilate<T>(tensor<T> pixels, int radius = 1) -> tensor<T> \| error` |
| `erode` | `erode<T>(tensor<T> pixels, int radius = 1) -> tensor<T> \| error` |

`crop`, `resize`, flips, and rotations accept tensors with rank >= 3 and interpret the trailing dimensions as `(..., C, H, W)`, preserving every leading dimension. Floating tracked tensors remain tracked through these geometry operations and participate in autograd. Floating grayscale, blur/filter, and morphology paths are compositions of Core tensor primitives, so their graphs remain differentiable through `backward(track = true)` for higher-order derivatives as well as ordinary first-order backward. Thresholding never detaches implicitly and rejects tracked floating input.

`resize` uses nearest-neighbor sampling. `rotate90` turns clockwise and
`rotate270` turns counter-clockwise; both exchange height and width. `rotate180`
uses one direct geometry pass rather than composing two flips. These operations
only relocate samples and therefore preserve the element type exactly.

`grayscale` accepts one, three, or four channels and returns one channel. For
three- or four-channel input it combines RGB using the coefficients
`0.299 * R + 0.587 * G + 0.114 * B` in floating-point and rounds only the final
luminance to `uint8`; an alpha channel is intentionally ignored rather than
silently mixed into luminance.

`threshold` writes `high` where the value is greater than or equal to `cutoff`
and `low` elsewhere. `filter` accepts a non-empty rank-2 `tensor<int>` kernel, divides
the accumulated sum by a nonzero `divisor`, adds `offset`, and clamps the result
to the `uint8` range. `blur` averages the window. `dilate` takes the maximum and
`erode` takes the minimum while preserving the source element type. Every window
operation shrinks the window at the border rather than inventing padded values,
and `crop` requires the requested rectangle to lie inside the image.

## Device placement

Vision uses Quidra's tensor placement semantics directly. Image decoding through `vision.read<T>` produces a CPU tensor by default. Moving image
data to a GPU is always an explicit caller action:

```quidra
tensor<uint8> image_cpu = vision.read<uint8>("input.png")
tensor<uint8> image_gpu = image_cpu.gpu(0)
```

Vision never moves an input to CPU or GPU implicitly. Tensor geometry
(`crop`, `resize`, flips and rotations), grayscale/threshold, blur/filter, and
morphology preserve the caller's device. Portable and differentiable paths use
Core's generic tensor/autograd primitives; Vision may replace domain operations
with package-owned native kernels when the device/layout contract matches. The
current untracked contiguous CPU `uint8` filter path uses
`native/vision_native.cpp` through Core's opaque native-extension ABI. If a backend/element-type combination is unavailable, the operation fails
explicitly rather than iterating over hidden CPU storage or returning a CPU
result. Codec and filesystem APIs remain host operations, so writing a GPU tensor
still requires an explicit `.cpu()`. `vision.write` also requires contiguous
storage and returns an error instead of materializing a hidden copy. The public Vision API is vendor-independent;
backend selection is an implementation detail of Quidra/Vision.

Vision uses the same `MAJOR.MINOR.PATCH` version as Core and Math. The release
workflow requires immutable Core and Math tags with exactly the Vision package
version, checks that both tags exist, and validates Vision against those exact
dependencies before tagging. The declared dependency ranges must admit that
shared version. During development, CI instead builds the current
Core and Math `develop` branches to catch forward-compatibility regressions
without changing the package's released dependency contracts. On real GPU
hardware,
`tests/real_gpu_integration.sh /path/to/quidra` compares CPU and GPU Vision
results; set `QUIDRA_REQUIRE_REAL_GPU=1` in a hardware runner to require the
device instead of skipping when none is present.

## Example

See [`examples/process.qui`](examples/process.qui).

## Development

See [`docs/development.md`](docs/development.md) for the canonical main/develop and release procedure.

## License

MIT


## Ownership boundary

Vision owns image-domain semantics end to end: transforms, codecs, native C/C++
implementations, and codec-library integration. The codec implementation lives in
`native/image_codec.cpp` and talks to tensors only through
`<quidra/native_extension.h>`. Core does not provide image primitives, codec
wrappers, or image-specific linker policy. The package manifest declares libpng,
libjpeg, libtiff, and libwebp through the generic `native.pkg.*` mechanism.
