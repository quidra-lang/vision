#include <quidra/native_extension.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <type_traits>
#include <cstring>
#include <limits>

#ifdef __APPLE__
// Vision-owned Metal kernels (native/vision_metal.mm). Both receive tensors
// whose dtype, device, layout, and block geometry were validated here. A
// nonzero `write_only` takes the block mean's output write-only (see
// block_mean_apply).
extern "C" int vision_metal_block_mean(
    const void* input, void* output, std::uint64_t factor, int write_only);
extern "C" int vision_metal_block_spread(
    const void* gradient_output, void* gradient_input, std::uint64_t factor);
#endif

namespace {

struct OperationFlags {
    OperationFlags() {
        qcore_op_register_flags("vision_native_filter_u8", QCORE_OP_NAN_PRESERVING);
        qcore_op_register_flags("vision_native_downsample_mean", QCORE_OP_VALIDATES_NAN);
    }
};
const OperationFlags operation_flags;

bool checked_mul_u8_i64(
    std::uint8_t left, std::int64_t right, std::int64_t& out) {
    if (left == 0 || right == 0) {
        out = 0;
        return true;
    }
    const auto value = static_cast<std::int64_t>(left);
    if (right > 0 && right > std::numeric_limits<std::int64_t>::max() / value)
        return false;
    if (right < 0 && right < std::numeric_limits<std::int64_t>::min() / value)
        return false;
    out = value * right;
    return true;
}

bool checked_add_i64(
    std::int64_t left, std::int64_t right, std::int64_t& out) {
    if (right > 0 &&
        left > std::numeric_limits<std::int64_t>::max() - right)
        return false;
    if (right < 0 &&
        left < std::numeric_limits<std::int64_t>::min() - right)
        return false;
    out = left + right;
    return true;
}

bool positive(long long value) {
    return value > 0;
}

// Host kernels split their independent outputs - rows of output samples -
// with Core's qcore_parallel_for. Every row is computed exactly as the serial
// loop computes it, so the results are bitwise the same for every thread
// count. Bodies take no Core calls; every pointer is taken before the split.
// The grain only decides how many rows a chunk gets: about `target` samples
// of work, so a small image stays one chunk on the calling thread.
std::uint64_t rows_per_chunk(std::size_t work_per_row) {
    constexpr std::size_t target = 16384;
    if (work_per_row == 0 || work_per_row >= target) return 1;
    return target / work_per_row;
}

struct FilterJob {
    const std::uint8_t* source;
    const std::int64_t* weights;
    std::uint8_t* destination;
    std::size_t height;
    std::size_t width;
    std::size_t kernel_height;
    std::size_t kernel_width;
    long long divisor;
    long long offset;
};

// Rows [begin, end) of the planes stacked top to bottom. A failure returns
// the status of the first failing sample of the chunk in serial order;
// Core returns the status of the lowest failing chunk, which is therefore
// the status the serial loop returns.
int filter_rows(std::uint64_t begin, std::uint64_t end, void* raw) {
    const auto& job = *static_cast<const FilterJob*>(raw);
    const auto height = job.height;
    const auto width = job.width;
    const auto center_y = job.kernel_height / 2;
    const auto center_x = job.kernel_width / 2;
    for (std::uint64_t index = begin; index < end; ++index) {
        const auto plane = static_cast<std::size_t>(index) / height;
        const auto y = static_cast<std::size_t>(index) % height;
        for (std::size_t x = 0; x < width; ++x) {
            std::int64_t sum = 0;
            for (std::size_t ky = 0; ky < job.kernel_height; ++ky) {
                if (ky > center_y + y) continue;
                const auto source_y = y + ky - center_y;
                if (source_y >= height) continue;
                for (std::size_t kx = 0; kx < job.kernel_width; ++kx) {
                    if (kx > center_x + x) continue;
                    const auto source_x = x + kx - center_x;
                    if (source_x >= width) continue;

                    const auto pixel =
                        job.source[(plane * height + source_y) * width + source_x];
                    const auto weight = job.weights[ky * job.kernel_width + kx];
                    std::int64_t product = 0;
                    std::int64_t next = 0;
                    if (!checked_mul_u8_i64(pixel, weight, product) ||
                        !checked_add_i64(sum, product, next)) {
                        return 9;
                    }
                    sum = next;
                }
            }

            if (sum == std::numeric_limits<std::int64_t>::min() &&
                job.divisor == -1) {
                return 10;
            }
            const auto divided = sum / job.divisor;
            std::int64_t adjusted = 0;
            if (!checked_add_i64(divided, job.offset, adjusted)) return 10;
            if (adjusted < 0) adjusted = 0;
            if (adjusted > 255) adjusted = 255;
            job.destination[(plane * height + y) * width + x] =
                static_cast<std::uint8_t>(adjusted);
        }
    }
    return 0;
}

} // namespace

extern "C" std::int32_t vision_native_filter_u8(
    const void* pixels,
    const void* kernel,
    void* output,
    long long divisor,
    long long offset) {
    if (qcore_native_abi_version() != QUIDRA_NATIVE_ABI_VERSION ||
        !pixels || !kernel || !output) {
        return 1;
    }

    if (divisor == 0 ||
        qcore_tensor_dtype(pixels) != QCORE_DTYPE_UINT8 ||
        qcore_tensor_dtype(kernel) != QCORE_DTYPE_INT64 ||
        qcore_tensor_dtype(output) != QCORE_DTYPE_UINT8 ||
        qcore_tensor_device(pixels) != -1 ||
        qcore_tensor_device(kernel) != -1 ||
        qcore_tensor_device(output) != -1) {
        return 2;
    }

    const auto rank = qcore_tensor_rank(pixels);
    if (rank < 3 || qcore_tensor_rank(output) != rank ||
        qcore_tensor_rank(kernel) != 2) {
        return 3;
    }
    for (unsigned long long axis = 0; axis < rank; ++axis) {
        const auto extent = qcore_tensor_extent(pixels, axis);
        if (!positive(extent) || qcore_tensor_extent(output, axis) != extent)
            return 4;
    }

    const auto height_raw = qcore_tensor_extent(pixels, rank - 2);
    const auto width_raw = qcore_tensor_extent(pixels, rank - 1);
    const auto kernel_height_raw = qcore_tensor_extent(kernel, 0);
    const auto kernel_width_raw = qcore_tensor_extent(kernel, 1);
    if (!positive(height_raw) || !positive(width_raw) ||
        !positive(kernel_height_raw) || !positive(kernel_width_raw)) {
        return 5;
    }

    const auto total = qcore_tensor_element_count(pixels);
    if (qcore_tensor_element_count(output) != total) return 6;

    const auto height = static_cast<std::size_t>(height_raw);
    const auto width = static_cast<std::size_t>(width_raw);
    const auto kernel_height = static_cast<std::size_t>(kernel_height_raw);
    const auto kernel_width = static_cast<std::size_t>(kernel_width_raw);
    if (height != 0 &&
        width > std::numeric_limits<std::size_t>::max() / height) {
        return 7;
    }
    const auto plane_size = height * width;
    if (plane_size == 0 || total % plane_size != 0) return 7;
    const auto planes = static_cast<std::size_t>(total / plane_size);

    const auto* source = static_cast<const std::uint8_t*>(
        qcore_tensor_cpu_data_const(pixels));
    const auto* weights = static_cast<const std::int64_t*>(
        qcore_tensor_cpu_data_const(kernel));
    auto* destination = static_cast<std::uint8_t*>(
        qcore_tensor_cpu_data(output));
    if (!source || !weights || !destination) return 8;

    // The kernel tensor holds kernel_height * kernel_width elements, so the
    // product fits; the work estimate saturates.
    const auto taps = kernel_height * kernel_width;
    const auto work_per_row =
        taps != 0 && width > std::numeric_limits<std::size_t>::max() / taps
            ? std::numeric_limits<std::size_t>::max()
            : width * taps;
    FilterJob job{source, weights, destination, height, width,
                  kernel_height, kernel_width, divisor, offset};
    const int status = qcore_parallel_for(
        0, static_cast<std::uint64_t>(planes) * height,
        rows_per_chunk(work_per_row), filter_rows, &job);
    // A negative status is Core's: its parallel call itself failed (11). Like
    // every nonzero status, it leaves vision.filter to its portable path.
    if (status < 0) return 11;
    return status;
}


// ---------------------------------------------------------------------------
// Block-mean downsampling.
//
// output[..., c, oy, ox] is the mean of the complete factor x factor block whose
// top-left sample is input[..., c, oy * factor, ox * factor]. Trailing rows and
// columns that do not fill a whole block are ignored. Floating blocks are
// accumulated in their own element type in one fixed order - each row of the
// block left to right, then the row sums top to bottom - and divided once by
// factor * factor. The Metal kernels and the portable Quidra composition in
// internal.qui use the same order.
// uint8 blocks are summed exactly and the quotient is truncated, like the
// uint8 blur. The operation is linear, so its autograd needs no saved tensors:
// the backward "spreads" each output gradient over its block divided by
// factor * factor, and the backward of that spread is the block mean again.

namespace {

// vision.downsample_mean (internal.qui block_native_unavailable) uses the
// portable composition only for the statuses that mean "no Vision kernel can
// run here": 2 (no kernel for this dtype), 5 (Core lends no storage: its CPU
// accessors and device handles refuse a view that contains an uninitialized
// element, on every backend, and the composition then reports Core's
// UNINITIALIZED) and 6 (no kernel for this backend or size).
// Every other status is a failure that is returned as an error, never hidden
// behind the composition.
enum BlockStatus : std::int32_t {
    block_ok = 0,
    block_invalid_argument = 1,
    block_unsupported_dtype = 2,
    block_device_mismatch = 3,
    block_layout_mismatch = 4,
    block_storage_unavailable = 5,
    block_backend_unavailable = 6,
    // The kernel could not run: a device kernel could not be encoded, or
    // Core's parallel call refused a host kernel. A device kernel that fails
    // while it runs is reported by Core at the device's next synchronization
    // point, like every other device error.
    block_backend_failed = 7,
    block_autograd_failed = 8,
    // The backend has no arithmetic for this dtype at all (Metal has no
    // float64), so the portable composition could not run either.
    block_dtype_unavailable = 9,
};

struct BlockGeometry {
    std::size_t planes{};
    std::size_t height{};
    std::size_t width{};
    std::size_t output_height{};
    std::size_t output_width{};
    std::size_t factor{};
};

struct BlockMetadata {
    std::uint64_t factor;
};

bool floating_dtype(int dtype) {
    return dtype == QCORE_DTYPE_FLOAT32 || dtype == QCORE_DTYPE_FLOAT64;
}

// Checks that `reduced` is the factor-reduced (..., C, H / f, W / f) shape of
// `full` and that both are contiguous tensors of one dtype on one device.
std::int32_t block_geometry(
    const void* full,
    const void* reduced,
    std::uint64_t factor,
    BlockGeometry& geometry) {
    if (!full || !reduced || factor == 0) return block_invalid_argument;
    const auto dtype = qcore_tensor_dtype(full);
    if (dtype != qcore_tensor_dtype(reduced) ||
        (dtype != QCORE_DTYPE_UINT8 && !floating_dtype(dtype)))
        return block_unsupported_dtype;
    if (qcore_tensor_device(full) != qcore_tensor_device(reduced) ||
        qcore_tensor_backend(full) != qcore_tensor_backend(reduced) ||
        qcore_tensor_backend(full) < 0)
        return block_device_mismatch;
    if (!qcore_tensor_is_contiguous(full) ||
        !qcore_tensor_is_contiguous(reduced))
        return block_layout_mismatch;

    const auto rank = qcore_tensor_rank(full);
    if (rank < 3 || qcore_tensor_rank(reduced) != rank)
        return block_layout_mismatch;
    std::size_t planes = 1;
    for (unsigned long long axis = 0; axis + 2 < rank; ++axis) {
        const auto extent = qcore_tensor_extent(full, axis);
        if (!positive(extent) || qcore_tensor_extent(reduced, axis) != extent)
            return block_layout_mismatch;
        const auto value = static_cast<std::size_t>(extent);
        if (planes > std::numeric_limits<std::size_t>::max() / value)
            return block_layout_mismatch;
        planes *= value;
    }
    const auto height = qcore_tensor_extent(full, rank - 2);
    const auto width = qcore_tensor_extent(full, rank - 1);
    if (!positive(height) || !positive(width) ||
        factor > static_cast<std::uint64_t>(height) ||
        factor > static_cast<std::uint64_t>(width))
        return block_layout_mismatch;
    const auto output_height = static_cast<std::uint64_t>(height) / factor;
    const auto output_width = static_cast<std::uint64_t>(width) / factor;
    if (qcore_tensor_extent(reduced, rank - 2) !=
            static_cast<long long>(output_height) ||
        qcore_tensor_extent(reduced, rank - 1) !=
            static_cast<long long>(output_width))
        return block_layout_mismatch;

    const auto rows = static_cast<std::size_t>(height);
    const auto columns = static_cast<std::size_t>(width);
    if (columns > std::numeric_limits<std::size_t>::max() / rows ||
        planes > std::numeric_limits<std::size_t>::max() / (rows * columns) ||
        qcore_tensor_element_count(full) != planes * rows * columns ||
        qcore_tensor_element_count(reduced) !=
            planes * static_cast<std::size_t>(output_height) *
                static_cast<std::size_t>(output_width))
        return block_layout_mismatch;

    geometry.planes = planes;
    geometry.height = rows;
    geometry.width = columns;
    geometry.output_height = static_cast<std::size_t>(output_height);
    geometry.output_width = static_cast<std::size_t>(output_width);
    geometry.factor = static_cast<std::size_t>(factor);
    return block_ok;
}

// CPU tensors expose checked contiguous storage. The test backend's device
// memory is host-addressable, so the reference loops below also serve it.
template <typename T>
const T* block_read(const void* tensor) {
    const auto backend = qcore_tensor_backend(tensor);
    if (backend == QCORE_BACKEND_CPU)
        return static_cast<const T*>(qcore_tensor_cpu_data_const(tensor));
    if (backend == QCORE_BACKEND_TEST) {
        const auto handle = qcore_tensor_device_handle_const(tensor);
        const auto offset = qcore_tensor_device_offset_bytes(tensor);
        if (handle == 0 ||
            offset > std::numeric_limits<std::uint64_t>::max() - handle)
            return nullptr;
        return reinterpret_cast<const T*>(
            static_cast<std::uintptr_t>(handle + offset));
    }
    return nullptr;
}

// `write_only` takes a device output that the kernel stores completely with
// qcore_tensor_output_handle: Core lends it without a fill, and
// block_written marks it written once the kernel has stored it. Gradients
// Core lends to a backward callback are taken like any mutable tensor. A CPU
// output arrives zeroed and is written through the CPU accessor.
template <typename T>
T* block_write(void* tensor, bool write_only) {
    const auto backend = qcore_tensor_backend(tensor);
    if (backend == QCORE_BACKEND_CPU)
        return static_cast<T*>(qcore_tensor_cpu_data(tensor));
    if (backend == QCORE_BACKEND_TEST) {
        const auto handle = write_only ? qcore_tensor_output_handle(tensor)
                                       : qcore_tensor_device_handle(tensor);
        const auto offset = qcore_tensor_device_offset_bytes(tensor);
        if (handle == 0 ||
            offset > std::numeric_limits<std::uint64_t>::max() - handle)
            return nullptr;
        return reinterpret_cast<T*>(
            static_cast<std::uintptr_t>(handle + offset));
    }
    return nullptr;
}

bool block_written(void* tensor, bool write_only) {
    return !write_only || qcore_tensor_backend(tensor) != QCORE_BACKEND_TEST ||
           qcore_tensor_mark_written(tensor) == 0;
}

// Output samples of one pass over the rows of a block row. The running block
// sums of a pass live on the stack, which stays small: parallel bodies may
// run on threads with small stacks.
constexpr std::size_t block_tile = 256;

// Sum of each block of output row (plane, oy) in the documented order;
// Accumulator is T for floating input and an exact 64-bit integer for uint8.
template <typename T, typename Accumulator, typename Finish>
bool block_mean_row(
    const T* input,
    T* output,
    const BlockGeometry& geometry,
    const Finish& finish,
    std::size_t plane,
    std::size_t oy) {
    const auto factor = geometry.factor;
    const T* top =
        input + (plane * geometry.height + oy * factor) * geometry.width;
    T* destination = output +
        (plane * geometry.output_height + oy) * geometry.output_width;
    bool nonfinite = false;
    Accumulator totals[block_tile];
    for (std::size_t first = 0; first < geometry.output_width;
         first += block_tile) {
        const auto last = std::min(first + block_tile, geometry.output_width);
        for (std::size_t dy = 0; dy < factor; ++dy) {
            const T* row = top + dy * geometry.width;
            for (std::size_t ox = first; ox < last; ++ox) {
                const T* run = row + ox * factor;
                Accumulator sum = static_cast<Accumulator>(run[0]);
                for (std::size_t dx = 1; dx < factor; ++dx)
                    sum += static_cast<Accumulator>(run[dx]);
                totals[ox - first] = dy == 0 ? sum : totals[ox - first] + sum;
            }
        }
        for (std::size_t ox = first; ox < last; ++ox) {
            const T value = finish(totals[ox - first]);
            destination[ox] = value;
            if constexpr (std::is_floating_point_v<T>)
                nonfinite |= std::isnan(value);
        }
    }
    return nonfinite;
}

template <typename T, typename Finish>
struct BlockMeanJob {
    const T* input;
    T* output;
    const BlockGeometry* geometry;
    Finish finish;
};

template <typename T, typename Accumulator, typename Finish>
int block_mean_rows(std::uint64_t begin, std::uint64_t end, void* raw) {
    const auto& job = *static_cast<const BlockMeanJob<T, Finish>*>(raw);
    const auto& geometry = *job.geometry;
    bool nonfinite = false;
    for (std::uint64_t index = begin; index < end; ++index) {
        nonfinite |= block_mean_row<T, Accumulator>(
            job.input, job.output, geometry, job.finish,
            static_cast<std::size_t>(index) / geometry.output_height,
            static_cast<std::size_t>(index) % geometry.output_height);
    }
    return nonfinite ? 11 : 0;
}

// Every output row (plane, oy) is independent.
template <typename T, typename Accumulator, typename Finish>
bool block_mean_host(
    const void* tensor,
    const T* input,
    T* output,
    const BlockGeometry& geometry,
    Finish finish) {
    BlockMeanJob<T, Finish> job{input, output, &geometry, finish};
    const int status = qcore_parallel_for(
               0,
               static_cast<std::uint64_t>(geometry.planes) *
                   geometry.output_height,
               rows_per_chunk(geometry.factor * geometry.width),
               block_mean_rows<T, Accumulator, Finish>, &job);
    if (status == 11)
        qcore_tensor_report_status(tensor, "vision.downsample_mean", 1);
    return status == 0;
}

template <typename T>
struct BlockSpreadJob {
    const T* gradient_output;
    T* gradient_input;
    const BlockGeometry* geometry;
};

// Rows (plane, y) of the full-resolution gradient.
template <typename T>
int block_spread_rows(std::uint64_t begin, std::uint64_t end, void* raw) {
    const auto& job = *static_cast<const BlockSpreadJob<T>*>(raw);
    const auto& geometry = *job.geometry;
    const auto factor = geometry.factor;
    const T divisor = static_cast<T>(factor * factor);
    const auto covered_height = geometry.output_height * factor;
    for (std::uint64_t index = begin; index < end; ++index) {
        const auto plane = static_cast<std::size_t>(index) / geometry.height;
        const auto y = static_cast<std::size_t>(index) % geometry.height;
        T* row = job.gradient_input +
            (plane * geometry.height + y) * geometry.width;
        std::size_t x = 0;
        if (y < covered_height) {
            const T* source = job.gradient_output +
                (plane * geometry.output_height + y / factor) *
                    geometry.output_width;
            for (std::size_t ox = 0; ox < geometry.output_width; ++ox) {
                const T value = source[ox] / divisor;
                for (std::size_t dx = 0; dx < factor; ++dx)
                    row[x++] = value;
            }
        }
        for (; x < geometry.width; ++x) row[x] = T{};
    }
    return 0;
}

// Every full-resolution row (plane, y) is independent.
template <typename T>
bool block_spread_host(
    const T* gradient_output,
    T* gradient_input,
    const BlockGeometry& geometry) {
    BlockSpreadJob<T> job{gradient_output, gradient_input, &geometry};
    return qcore_parallel_for(
               0, static_cast<std::uint64_t>(geometry.planes) * geometry.height,
               rows_per_chunk(geometry.width), block_spread_rows<T>, &job) == 0;
}

// full -> reduced block mean on the tensors' shared backend. Every kernel
// stores each element of `reduced`, so the forward takes a device output
// write-only (`write_only`); as the adjoint of the spread it writes the
// gradient Core lends.
std::int32_t block_mean_apply(
    const void* full, void* reduced, std::uint64_t factor, bool write_only) {
    BlockGeometry geometry;
    const auto status = block_geometry(full, reduced, factor, geometry);
    if (status != block_ok) return status;

    const auto backend = qcore_tensor_backend(full);
    if (backend == QCORE_BACKEND_CPU || backend == QCORE_BACKEND_TEST) {
        switch (qcore_tensor_dtype(full)) {
            case QCORE_DTYPE_FLOAT32: {
                const auto* input = block_read<float>(full);
                auto* output = block_write<float>(reduced, write_only);
                if (!input || !output) return block_storage_unavailable;
                const float divisor = static_cast<float>(factor * factor);
                if (!block_mean_host<float, float>(
                        reduced, input, output, geometry,
                        [divisor](float total) { return total / divisor; }))
                    return block_backend_failed;
                return block_written(reduced, write_only)
                    ? block_ok
                    : block_backend_failed;
            }
            case QCORE_DTYPE_FLOAT64: {
                const auto* input = block_read<double>(full);
                auto* output = block_write<double>(reduced, write_only);
                if (!input || !output) return block_storage_unavailable;
                const double divisor = static_cast<double>(factor * factor);
                if (!block_mean_host<double, double>(
                        reduced, input, output, geometry,
                        [divisor](double total) { return total / divisor; }))
                    return block_backend_failed;
                return block_written(reduced, write_only)
                    ? block_ok
                    : block_backend_failed;
            }
            case QCORE_DTYPE_UINT8: {
                const auto* input = block_read<std::uint8_t>(full);
                auto* output = block_write<std::uint8_t>(reduced, write_only);
                if (!input || !output) return block_storage_unavailable;
                const std::uint64_t divisor = factor * factor;
                if (!block_mean_host<std::uint8_t, std::uint64_t>(
                        reduced, input, output, geometry,
                        [divisor](std::uint64_t total) {
                            return static_cast<std::uint8_t>(total / divisor);
                        }))
                    return block_backend_failed;
                return block_written(reduced, write_only)
                    ? block_ok
                    : block_backend_failed;
            }
            default:
                return block_unsupported_dtype;
        }
    }
#ifdef __APPLE__
    if (backend == QCORE_BACKEND_METAL)
        return vision_metal_block_mean(full, reduced, factor, write_only ? 1 : 0);
#endif
    return block_backend_unavailable;
}

// reduced gradient -> full gradient: the adjoint of block_mean_apply.
std::int32_t block_spread_apply(
    const void* gradient_output, void* gradient_input, std::uint64_t factor) {
    BlockGeometry geometry;
    const auto status =
        block_geometry(gradient_input, gradient_output, factor, geometry);
    if (status != block_ok) return status;
    if (!floating_dtype(qcore_tensor_dtype(gradient_output)))
        return block_unsupported_dtype;

    const auto backend = qcore_tensor_backend(gradient_output);
    if (backend == QCORE_BACKEND_CPU || backend == QCORE_BACKEND_TEST) {
        if (qcore_tensor_dtype(gradient_output) == QCORE_DTYPE_FLOAT32) {
            const auto* source = block_read<float>(gradient_output);
            auto* destination = block_write<float>(gradient_input, false);
            if (!source || !destination) return block_storage_unavailable;
            return block_spread_host(source, destination, geometry)
                ? block_ok
                : block_backend_failed;
        }
        const auto* source = block_read<double>(gradient_output);
        auto* destination = block_write<double>(gradient_input, false);
        if (!source || !destination) return block_storage_unavailable;
        return block_spread_host(source, destination, geometry)
            ? block_ok
            : block_backend_failed;
    }
#ifdef __APPLE__
    if (backend == QCORE_BACKEND_METAL)
        return vision_metal_block_spread(gradient_output, gradient_input, factor);
#endif
    return block_backend_unavailable;
}

bool block_factor(
    const void* metadata, std::uint64_t metadata_size, std::uint64_t& factor) {
    if (!metadata || metadata_size != sizeof(BlockMetadata)) return false;
    BlockMetadata value{};
    std::memcpy(&value, metadata, sizeof(value));
    factor = value.factor;
    return factor != 0;
}

int block_mean_backward(
    const void* const*, std::uint64_t,
    const void*, void* const*, std::uint64_t,
    const void*, std::uint64_t);
int block_mean_backward_tracked(
    const void* const*, std::uint64_t,
    const void* const*, std::uint64_t,
    const void*, void* const*, std::uint64_t,
    const void*, std::uint64_t);
int block_spread_backward(
    const void* const*, std::uint64_t,
    const void*, void* const*, std::uint64_t,
    const void*, std::uint64_t);
int block_spread_backward_tracked(
    const void* const*, std::uint64_t,
    const void* const*, std::uint64_t,
    const void*, void* const*, std::uint64_t,
    const void*, std::uint64_t);

// First-order backward through Core's masked ABI
// (qcore_tensor_attach_custom_autograd_masked). Core calls a node only when a
// backward target is reachable through its input, and skips the zero fill of
// a gradient declared at attach time as a full write. The spread stores every
// element of the full-resolution gradient (zero where a sample lies outside
// every complete block), and the block mean every element of the reduced
// one, so both nodes declare their one gradient a full write and report it
// written.
constexpr std::uint8_t full_write[] = {1};

// The masked form of a single-input backward: the one gradient is needed
// whenever Core calls the node.
template <qcore_autograd_backward_fn Backward>
int single_input_masked(
    const void* const* saved_tensors,
    std::uint64_t saved_tensor_count,
    const void* gradient_output,
    void* const* gradient_inputs,
    const std::uint8_t* needed,
    std::uint8_t* fully_written,
    std::uint64_t gradient_input_count,
    const void* metadata,
    std::uint64_t metadata_size) {
    if (!needed || !fully_written || gradient_input_count != 1)
        return block_invalid_argument;
    if (!needed[0]) return 0;
    const int status = Backward(
        saved_tensors, saved_tensor_count, gradient_output, gradient_inputs,
        gradient_input_count, metadata, metadata_size);
    if (status == 0) fully_written[0] = 1;
    return status;
}

// Core hands every callback below its upstream gradient (and, for the tracked
// callbacks, every input and gradient) contiguous at offset 0 on the node's
// device; it materializes a strided gradient, for example one below a
// transpose(), before the call. So block_geometry's contiguity check holds
// for them, and the kernels read the gradients without strides.

// First-order backward of the block mean: spread the output gradient.
int block_mean_backward(
    const void* const*,
    std::uint64_t,
    const void* gradient_output,
    void* const* gradient_inputs,
    std::uint64_t gradient_input_count,
    const void* metadata,
    std::uint64_t metadata_size) {
    std::uint64_t factor = 0;
    if (!gradient_output || !gradient_inputs || gradient_input_count != 1 ||
        !gradient_inputs[0] || !block_factor(metadata, metadata_size, factor))
        return block_invalid_argument;
    return block_spread_apply(gradient_output, gradient_inputs[0], factor);
}

// backward(track = true): the spread is itself linear in the incoming
// gradient, so its provenance is one Vision node whose backward is the block
// mean. Higher orders keep alternating between the two adjoint operations.
int block_mean_backward_tracked(
    const void* const*,
    std::uint64_t,
    const void* const* saved_tensors,
    std::uint64_t saved_tensor_count,
    const void* gradient_output,
    void* const* gradient_inputs,
    std::uint64_t gradient_input_count,
    const void* metadata,
    std::uint64_t metadata_size) {
    const int status = block_mean_backward(
        saved_tensors, saved_tensor_count, gradient_output,
        gradient_inputs, gradient_input_count, metadata, metadata_size);
    if (status != 0) return status;
    const void* parents[] = {gradient_output};
    return qcore_tensor_attach_custom_autograd_masked(
        gradient_inputs[0], parents, 1, nullptr, 0,
        single_input_masked<block_spread_backward>,
        block_spread_backward_tracked, full_write,
        metadata, metadata_size) == 0
        ? 0
        : block_autograd_failed;
}

// Backward of the spread: the block mean of the incoming gradient.
int block_spread_backward(
    const void* const*,
    std::uint64_t,
    const void* gradient_output,
    void* const* gradient_inputs,
    std::uint64_t gradient_input_count,
    const void* metadata,
    std::uint64_t metadata_size) {
    std::uint64_t factor = 0;
    if (!gradient_output || !gradient_inputs || gradient_input_count != 1 ||
        !gradient_inputs[0] || !block_factor(metadata, metadata_size, factor))
        return block_invalid_argument;
    return block_mean_apply(gradient_output, gradient_inputs[0], factor, false);
}

int block_spread_backward_tracked(
    const void* const*,
    std::uint64_t,
    const void* const* saved_tensors,
    std::uint64_t saved_tensor_count,
    const void* gradient_output,
    void* const* gradient_inputs,
    std::uint64_t gradient_input_count,
    const void* metadata,
    std::uint64_t metadata_size) {
    const int status = block_spread_backward(
        saved_tensors, saved_tensor_count, gradient_output,
        gradient_inputs, gradient_input_count, metadata, metadata_size);
    if (status != 0) return status;
    const void* parents[] = {gradient_output};
    return qcore_tensor_attach_custom_autograd_masked(
        gradient_inputs[0], parents, 1, nullptr, 0,
        single_input_masked<block_mean_backward>,
        block_mean_backward_tracked, full_write,
        metadata, metadata_size) == 0
        ? 0
        : block_autograd_failed;
}

} // namespace

// Private bridge for vision.downsample_mean. `output` is a fresh tensor with
// the reduced shape on the input's device (internal.native_output: zeroed on
// the CPU, allocated without a fill on a device), which the kernel stores
// completely. BlockStatus says which nonzero statuses leave the caller to use
// the portable Vision composition.
extern "C" std::int32_t vision_native_downsample_mean(
    const void* pixels,
    void* output,
    long long factor) {
    // Kernels built for another native ABI cannot be used; the portable
    // composition needs no native code.
    if (qcore_native_abi_version() != QUIDRA_NATIVE_ABI_VERSION)
        return block_backend_unavailable;
    if (!pixels || !output || factor < 1) return block_invalid_argument;
    const auto block = static_cast<std::uint64_t>(factor);
    const auto status = block_mean_apply(pixels, output, block, true);
    if (status != block_ok) return status;
    if (!floating_dtype(qcore_tensor_dtype(pixels))) return block_ok;

    // Attaches nothing for untracked input.
    const void* inputs[] = {pixels};
    const BlockMetadata metadata{block};
    return qcore_tensor_attach_custom_autograd_masked(
               output, inputs, 1, nullptr, 0,
               single_input_masked<block_mean_backward>,
               block_mean_backward_tracked, full_write,
               &metadata, sizeof(metadata)) == 0
        ? block_ok
        : block_autograd_failed;
}
