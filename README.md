# Quidra Vision

Quidra Vision is Quidra's first-party tensor image-processing package, imported
as `vision`. It works directly on tensors with rank >= 3 and trailing
`(..., C, H, W)` dimensions. Image file I/O is owned by Vision itself through
`vision.read` / `vision.write`; Core has no image codec or image namespace.

Geometry operations and morphology preserve the input tensor element type. `grayscale`, `blur`, `filter`, and `downsample_mean` use one public name across the `tensor<nat8>` image path and differentiable floating paths; Quidra's generic specialization resolves the dtype-specific implementation statically. `threshold` uses the same public name across the `tensor<nat8>` and floating paths. Because thresholding is discontinuous, tracked floating input is rejected rather than implicitly detached; untracked floating input is processed normally without creating an autograd graph.

Invalid shapes or parameters are returned as `error`; they are never silently
reinterpreted. This includes zero-size resize targets, out-of-bounds crops,
downsample factors that are not positive or exceed the image, unsupported
grayscale channel counts, negative window radii, invalid filter kernels, and a
zero filter divisor.

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
| `downsample_mean` | `downsample_mean(tensor<nat8> pixels, int factor) -> tensor<nat8> \| error`; `downsample_mean<T: floating>(tensor<T> pixels, int factor) -> tensor<T> \| error` |
| `flip_horizontal` | `flip_horizontal<T>(tensor<T> pixels) -> tensor<T> \| error` |
| `flip_vertical` | `flip_vertical<T>(tensor<T> pixels) -> tensor<T> \| error` |
| `rotate90` | `rotate90<T>(tensor<T> pixels) -> tensor<T> \| error` |
| `rotate180` | `rotate180<T>(tensor<T> pixels) -> tensor<T> \| error` |
| `rotate270` | `rotate270<T>(tensor<T> pixels) -> tensor<T> \| error` |
| `grayscale` | `grayscale(tensor<nat8> pixels) -> tensor<nat8> \| error`; `grayscale<T: floating>(tensor<T> pixels) -> tensor<T> \| error` |
| `threshold` | `threshold(tensor<nat8> pixels, nat8 cutoff, nat8 low = nat8(0), nat8 high = nat8(255)) -> tensor<nat8> \| error`; `threshold<T: floating>(tensor<T> pixels, T cutoff, T low = 0.0, T high = 1.0) -> tensor<T> \| error` |
| `blur` | `blur(tensor<nat8> pixels, int radius = 1) -> tensor<nat8> \| error`; `blur<T: floating>(tensor<T> pixels, int radius = 1) -> tensor<T> \| error` |
| `filter` | `filter(tensor<nat8> pixels, tensor<int64> kernel, int divisor = 1, int offset = 0) -> tensor<nat8> \| error`; `filter<T: floating, K: floating>(tensor<T> pixels, tensor<K> kernel, K divisor = 1.0, K offset = 0.0) -> tensor<T> \| error` |
| `dilate` | `dilate<T>(tensor<T> pixels, int radius = 1) -> tensor<T> \| error` |
| `erode` | `erode<T>(tensor<T> pixels, int radius = 1) -> tensor<T> \| error` |

`crop`, `resize`, flips, and rotations accept tensors with rank >= 3 and interpret the trailing dimensions as `(..., C, H, W)`, preserving every leading dimension. Floating tracked tensors remain tracked through these geometry operations and participate in autograd. Floating grayscale, blur/filter, and morphology paths are compositions of Core tensor primitives, so their graphs remain differentiable through `backward(track = true)` for higher-order derivatives as well as ordinary first-order backward. Thresholding never detaches implicitly and rejects tracked floating input.

`resize` uses nearest-neighbor sampling. `downsample_mean` instead averages
areas: it divides height and width by an integer `factor` and replaces every
complete `factor x factor` block with its mean, so thin structures are blended
rather than aliased away. The result keeps every leading dimension and has
`H / factor` rows and `W / factor` columns (integer division); trailing rows and
columns that do not fill a whole block are ignored, so the result depends only
on the top-left `(H / factor * factor) x (W / factor * factor)` region. A factor
of 1 returns the input unchanged. Floating tensors accumulate each block in
their own element type in one fixed order - every row of the block left to
right, then the row sums top to bottom - and divide once by `factor * factor`;
the native CPU and Metal kernels and the portable composition all use that
order. The CPU and Metal kernels give bit-identical results for normal-range
`real32` values, infinities and NaN, including for views and gradients; Metal
flushes subnormal `real32` inputs and results to zero, where the CPU kernel
keeps them. `tensor<nat8>` blocks are summed exactly and the quotient is truncated,
as in the `nat8` `blur`. The block mean is linear, so tracked floating input
stays tracked: backward spreads each output gradient evenly over its block
(ignored samples receive zero), and `backward(track = true)` keeps that gradient
differentiable for higher-order derivatives.

`rotate90` turns clockwise and
`rotate270` turns counter-clockwise; both exchange height and width. `rotate180`
uses one direct geometry pass rather than composing two flips. These operations
only relocate samples and therefore preserve the element type exactly.

`grayscale` accepts one, three, or four channels and returns one channel. For
three- or four-channel input it combines RGB using the coefficients
`0.299 * R + 0.587 * G + 0.114 * B` in floating-point and rounds only the final
luminance to `nat8`; an alpha channel is intentionally ignored rather than
silently mixed into luminance.

`threshold` writes `high` where the value is greater than or equal to `cutoff`
and `low` elsewhere. `filter` accepts a non-empty rank-2 `tensor<int64>` kernel, divides
the accumulated sum by a nonzero `divisor`, adds `offset`, and clamps the result
to the `nat8` range. `blur` averages the window. `dilate` takes the maximum and
`erode` takes the minimum while preserving the source element type. Every window
operation shrinks the window at the border rather than inventing padded values,
and `crop` requires the requested rectangle to lie inside the image.

## Device placement

Vision uses Quidra's tensor placement semantics directly. Image decoding through `vision.read<T>` produces a CPU tensor by default. Moving image
data to a GPU is always an explicit caller action:

```quidra
tensor<nat8> image_cpu = vision.read<nat8>("input.png")
tensor<nat8> image_gpu = image_cpu.gpu(0)
```

Vision never moves an input to CPU or GPU implicitly. Tensor geometry
(`crop`, `resize`, flips and rotations), grayscale/threshold, blur/filter, and
morphology preserve the caller's device. Portable and differentiable paths use
Core's generic tensor/autograd primitives; Vision may replace domain operations
with package-owned native kernels when the device/layout contract matches. The
current untracked contiguous CPU `nat8` filter path uses
`native/vision_native.cpp` through Core's opaque native-extension ABI.
`downsample_mean` has Vision-owned CPU kernels (`native/vision_native.cpp`) and
Metal kernels (`native/vision_metal.mm`) with Vision-owned autograd for tracked
floating input. A non-contiguous view is first made contiguous on its device
(a tracked view through one graph-preserving gather). A backend without a
Vision kernel (CUDA, HIP), or a Metal tensor too large for the kernels' 32-bit
dispatch, uses the portable composition of Core gather/add/division on the
input's device; its values equal the CPU kernel's exactly when that backend's
division is correctly rounded, and may differ in the last bit where the backend
uses fast-math division. `float64` on Metal returns an error, because Metal has
no `float64` arithmetic for either path. A Vision kernel that fails (for example
a failed Metal command buffer) also returns an error instead of switching to the
composition. The
native device kernels read GPU storage through Core's device-handle ABI, which
has no initialization query yet, so an uninitialized GPU input is not reported
as `UNINITIALIZED` the way CPU input is. If a backend/element-type combination is unavailable, the operation fails
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
and Metal implementations, and codec-library integration. The codec implementation lives in
`native/image_codec.cpp` and talks to tensors only through
`<quidra/native_extension.h>`. Core does not provide image primitives, codec
wrappers, or image-specific linker policy. The package manifest declares libpng,
libjpeg, libtiff, and libwebp through the generic `native.pkg.*` mechanism.
