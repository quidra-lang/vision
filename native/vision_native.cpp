#include <quidra/native_extension.h>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace {

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

    const auto center_y = kernel_height / 2;
    const auto center_x = kernel_width / 2;
    for (std::size_t plane = 0; plane < planes; ++plane) {
        for (std::size_t y = 0; y < height; ++y) {
            for (std::size_t x = 0; x < width; ++x) {
                std::int64_t sum = 0;
                for (std::size_t ky = 0; ky < kernel_height; ++ky) {
                    if (ky > center_y + y) continue;
                    const auto source_y = y + ky - center_y;
                    if (source_y >= height) continue;
                    for (std::size_t kx = 0; kx < kernel_width; ++kx) {
                        if (kx > center_x + x) continue;
                        const auto source_x = x + kx - center_x;
                        if (source_x >= width) continue;

                        const auto pixel =
                            source[(plane * height + source_y) * width + source_x];
                        const auto weight = weights[ky * kernel_width + kx];
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
                    divisor == -1) {
                    return 10;
                }
                const auto divided = sum / divisor;
                std::int64_t adjusted = 0;
                if (!checked_add_i64(divided, offset, adjusted)) return 10;
                if (adjusted < 0) adjusted = 0;
                if (adjusted > 255) adjusted = 255;
                destination[(plane * height + y) * width + x] =
                    static_cast<std::uint8_t>(adjusted);
            }
        }
    }
    return 0;
}
