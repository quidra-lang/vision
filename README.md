# vision

`vision` is Quidra's first-party tensor image-processing package. Every
operation accepts and returns `tensor<uint8>` values; file I/O is provided by
the standard `image` namespace.

## Install

Clone this repository and install it with the Quidra package command:

```sh
quidra package install . --name vision
```

Then import it normally:

```quidra
import vision

auto loaded = image.read("input.png")
match loaded
    tensor<uint8> pixels
        tensor<uint8> resized = vision.resize(
            pixels, height = 256, width = 256
        )
        image.write("output.png", resized)
    error problem
        print(problem)
```

For development, place the repository in a directory listed by
`QUIDRA_PACKAGE_PATH` instead of installing it.

## Data layout

Images are rank-3 CHW tensors: channel, height, width. Grayscale images have one
channel; color images normally have three or four channels.

## API

| Operation | Signature |
| --- | --- |
| `crop` | `crop(pixels, top, left, height, width)` |
| `resize` | `resize(pixels, height, width)` |
| `flip_horizontal` | `flip_horizontal(pixels)` |
| `flip_vertical` | `flip_vertical(pixels)` |
| `rotate90` | `rotate90(pixels)` |
| `rotate180` | `rotate180(pixels)` |
| `rotate270` | `rotate270(pixels)` |
| `grayscale` | `grayscale(pixels)` |
| `threshold` | `threshold(pixels, cutoff, low = uint8(0), high = uint8(255))` |
| `blur` | `blur(pixels, radius = 1)` |
| `filter` | `filter(pixels, kernel, divisor = 1, offset = 0)` |
| `dilate` | `dilate(pixels, radius = 1)` |
| `erode` | `erode(pixels, radius = 1)` |

`resize` uses nearest-neighbor sampling. `rotate90` turns clockwise and
`rotate270` turns counter-clockwise; both exchange height and width.
`grayscale` returns one channel, combining the first three channels of a
multi-channel image with the integer weights 77, 150, and 29 over 256.

`threshold` writes `high` where the value is greater than or equal to `cutoff`
and `low` elsewhere. `filter` accepts a rank-2 integer kernel, divides the
accumulated sum by `divisor`, adds `offset`, and clamps the result to the
`uint8` range. `blur` averages the window, `dilate` takes its maximum, and
`erode` takes its minimum. Every window operation shrinks the window at the
border rather than inventing padded values, and `crop` requires the requested
rectangle to lie inside the image.

## Example

See [`examples/process.qui`](examples/process.qui).

## License

MIT
