#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <quidra/native_extension.h>

#include <cstdint>
#include <limits>
#include <map>
#include <mutex>

// Vision-owned Metal kernels. Core lends its command queue and tensor buffers
// through the native-extension ABI; the image-domain semantics stay here.
// native/vision_native.cpp validates dtype, device, layout, and block geometry
// before calling these entry points.

namespace {

// Status values shared with native/vision_native.cpp (BlockStatus).
constexpr int status_ok = 0;
constexpr int status_unsupported_dtype = 2;
constexpr int status_backend_unavailable = 6;
constexpr int status_backend_failed = 7;
constexpr int status_dtype_unavailable = 9;

struct VisionMetalPrograms {
    id<MTLLibrary> library = nil;
    id<MTLComputePipelineState> mean_f32 = nil;
    id<MTLComputePipelineState> mean_u8 = nil;
    id<MTLComputePipelineState> spread_f32 = nil;
};

std::mutex programs_mutex;
std::map<std::uintptr_t, VisionMetalPrograms> programs_by_device;

// Mirrors BlockShape in the kernel source below.
struct BlockShape {
    std::uint32_t planes;
    std::uint32_t height;
    std::uint32_t width;
    std::uint32_t output_height;
    std::uint32_t output_width;
    std::uint32_t factor;
    std::uint32_t count;
    float divisor;
};

NSString* vision_metal_source() {
    static NSString* source = [[NSString alloc] initWithUTF8String:R"MSL(
#include <metal_stdlib>
using namespace metal;

struct BlockShape {
    uint planes;
    uint height;
    uint width;
    uint output_height;
    uint output_width;
    uint factor;
    uint count;
    float divisor;
};

// One thread per output sample. Each row of the block is summed left to
// right, then the row sums top to bottom, matching the CPU kernel.
kernel void vision_block_mean_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant BlockShape& shape [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= shape.count) return;
    const uint ox = index % shape.output_width;
    const uint rest = index / shape.output_width;
    const uint oy = rest % shape.output_height;
    const uint plane = rest / shape.output_height;
    const uint factor = shape.factor;
    device const float* top =
        input + (plane * shape.height + oy * factor) * shape.width + ox * factor;
    float total = 0.0f;
    for (uint dy = 0; dy < factor; ++dy) {
        device const float* run = top + dy * shape.width;
        float sum = run[0];
        for (uint dx = 1; dx < factor; ++dx) sum += run[dx];
        total = dy == 0 ? sum : total + sum;
    }
    output[index] = total / shape.divisor;
}

kernel void vision_block_mean_u8(
    device const uchar* input [[buffer(0)]],
    device uchar* output [[buffer(1)]],
    constant BlockShape& shape [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= shape.count) return;
    const uint ox = index % shape.output_width;
    const uint rest = index / shape.output_width;
    const uint oy = rest % shape.output_height;
    const uint plane = rest / shape.output_height;
    const uint factor = shape.factor;
    device const uchar* top =
        input + (plane * shape.height + oy * factor) * shape.width + ox * factor;
    ulong total = 0;
    for (uint dy = 0; dy < factor; ++dy) {
        device const uchar* run = top + dy * shape.width;
        for (uint dx = 0; dx < factor; ++dx) total += ulong(run[dx]);
    }
    output[index] = uchar(total / (ulong(factor) * ulong(factor)));
}

// One thread per full-resolution gradient sample: the adjoint of the block
// mean. Samples outside every complete block receive zero.
kernel void vision_block_spread_f32(
    device const float* gradient_output [[buffer(0)]],
    device float* gradient_input [[buffer(1)]],
    constant BlockShape& shape [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= shape.count) return;
    const uint x = index % shape.width;
    const uint rest = index / shape.width;
    const uint y = rest % shape.height;
    const uint plane = rest / shape.height;
    const uint oy = y / shape.factor;
    const uint ox = x / shape.factor;
    float value = 0.0f;
    if (oy < shape.output_height && ox < shape.output_width) {
        value = gradient_output[
            (plane * shape.output_height + oy) * shape.output_width + ox] /
            shape.divisor;
    }
    gradient_input[index] = value;
}
)MSL"];
    return source;
}

id<MTLComputePipelineState> make_pipeline(
    id<MTLDevice> device,
    id<MTLLibrary> library,
    NSString* name) {
    id<MTLFunction> function = [library newFunctionWithName:name];
    if (!function) return nil;
    NSError* error = nil;
    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithFunction:function error:&error];
    [function release];
    return pipeline;
}

VisionMetalPrograms* programs_for(id<MTLDevice> device) {
    if (!device) return nullptr;
    const auto key = reinterpret_cast<std::uintptr_t>((__bridge void*)device);
    std::lock_guard<std::mutex> lock(programs_mutex);
    auto [it, inserted] = programs_by_device.try_emplace(key);
    auto& programs = it->second;
    if (inserted) {
        // IEEE float semantics keep the documented summation order and an
        // exact division, so CPU and Metal agree on every sample in the
        // normal float32 range (and on Inf/NaN). The GPU still flushes
        // subnormal float32 inputs and results to zero, where the CPU keeps
        // them.
        MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
        if (@available(macOS 15.0, *)) {
            options.mathMode = MTLMathModeSafe;
        } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
            options.fastMathEnabled = NO;
#pragma clang diagnostic pop
        }
        NSError* error = nil;
        programs.library = [device newLibraryWithSource:vision_metal_source()
                                                options:options
                                                  error:&error];
        [options release];
        if (programs.library) {
            programs.mean_f32 =
                make_pipeline(device, programs.library, @"vision_block_mean_f32");
            programs.mean_u8 =
                make_pipeline(device, programs.library, @"vision_block_mean_u8");
            programs.spread_f32 = make_pipeline(
                device, programs.library, @"vision_block_spread_f32");
        }
    }
    if (!programs.library || !programs.mean_f32 || !programs.mean_u8 ||
        !programs.spread_f32)
        return nullptr;
    return &programs;
}

id<MTLCommandQueue> command_queue(const void* tensor) {
    const auto device = qcore_tensor_device(tensor);
    if (device < 0) return nil;
    const auto handle = qcore_device_queue_handle(device);
    if (handle == 0) return nil;
    return (__bridge id<MTLCommandQueue>)(
        reinterpret_cast<void*>(static_cast<std::uintptr_t>(handle)));
}

id<MTLBuffer> const_buffer(const void* tensor) {
    const auto handle = qcore_tensor_device_handle_const(tensor);
    if (handle == 0) return nil;
    return (__bridge id<MTLBuffer>)(
        reinterpret_cast<void*>(static_cast<std::uintptr_t>(handle)));
}

id<MTLBuffer> mutable_buffer(void* tensor) {
    const auto handle = qcore_tensor_device_handle(tensor);
    if (handle == 0) return nil;
    return (__bridge id<MTLBuffer>)(
        reinterpret_cast<void*>(static_cast<std::uintptr_t>(handle)));
}

bool fits_u32(unsigned long long value) {
    return value <= std::numeric_limits<std::uint32_t>::max();
}

// Reads the (..., C, H, W) geometry of the full-resolution tensor `full` and
// its block-reduced counterpart `reduced`; `count` is the dispatch size.
bool block_shape(
    const void* full,
    const void* reduced,
    std::uint64_t factor,
    unsigned long long count,
    BlockShape& shape) {
    const auto rank = qcore_tensor_rank(full);
    const auto full_count = qcore_tensor_element_count(full);
    if (rank < 3 || !fits_u32(full_count) || !fits_u32(count) ||
        !fits_u32(factor))
        return false;
    const auto height = qcore_tensor_extent(full, rank - 2);
    const auto width = qcore_tensor_extent(full, rank - 1);
    const auto output_height = qcore_tensor_extent(reduced, rank - 2);
    const auto output_width = qcore_tensor_extent(reduced, rank - 1);
    if (height <= 0 || width <= 0 || output_height <= 0 || output_width <= 0)
        return false;
    shape.height = static_cast<std::uint32_t>(height);
    shape.width = static_cast<std::uint32_t>(width);
    shape.planes = static_cast<std::uint32_t>(
        full_count / (static_cast<unsigned long long>(height) *
                      static_cast<unsigned long long>(width)));
    shape.output_height = static_cast<std::uint32_t>(output_height);
    shape.output_width = static_cast<std::uint32_t>(output_width);
    shape.factor = static_cast<std::uint32_t>(factor);
    shape.count = static_cast<std::uint32_t>(count);
    shape.divisor = static_cast<float>(factor * factor);
    return true;
}

NSUInteger thread_count(id<MTLComputePipelineState> pipeline) {
    NSUInteger width = pipeline.threadExecutionWidth;
    if (width == 0) width = 1;
    const NSUInteger maximum = pipeline.maxTotalThreadsPerThreadgroup;
    if (maximum != 0 && width > maximum) width = maximum;
    return width;
}

// Encodes one kernel on Core's queue for the tensors' device, so it is ordered
// after the work that produced its input and before later work on the result.
// The command buffer is then waited for: Core's host synchronization (.cpu(),
// .item(), gpu.sync) waits only for command buffers Core committed itself and
// returns immediately once those are complete, so an unwaited package kernel
// could still be running when the host reads its output.
int dispatch(
    id<MTLComputePipelineState> (*select)(VisionMetalPrograms&),
    const void* source,
    void* destination,
    const BlockShape& shape) {
    if (shape.count == 0) return status_ok;
    id<MTLCommandQueue> queue = command_queue(source);
    id<MTLBuffer> source_buffer = const_buffer(source);
    id<MTLBuffer> destination_buffer = mutable_buffer(destination);
    if (!queue || !source_buffer || !destination_buffer)
        return status_backend_unavailable;
    auto* programs = programs_for([queue device]);
    if (!programs) return status_backend_failed;
    id<MTLComputePipelineState> pipeline = select(*programs);

    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (!command || !encoder || !pipeline) return status_backend_failed;
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:source_buffer
                offset:static_cast<NSUInteger>(
                    qcore_tensor_device_offset_bytes(source))
               atIndex:0];
    [encoder setBuffer:destination_buffer
                offset:static_cast<NSUInteger>(
                    qcore_tensor_device_offset_bytes(destination))
               atIndex:1];
    [encoder setBytes:&shape length:sizeof(shape) atIndex:2];
    [encoder dispatchThreads:MTLSizeMake(shape.count, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(thread_count(pipeline), 1, 1)];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    return [command status] == MTLCommandBufferStatusCompleted
        ? status_ok
        : status_backend_failed;
}

} // namespace

extern "C" int vision_metal_block_mean(
    const void* input, void* output, std::uint64_t factor) {
    @autoreleasepool {
        BlockShape shape{};
        if (!block_shape(input, output, factor,
                         qcore_tensor_element_count(output), shape))
            return status_backend_unavailable;
        switch (qcore_tensor_dtype(input)) {
            case QCORE_DTYPE_FLOAT32:
                return dispatch(
                    [](VisionMetalPrograms& programs) { return programs.mean_f32; },
                    input, output, shape);
            case QCORE_DTYPE_UINT8:
                return dispatch(
                    [](VisionMetalPrograms& programs) { return programs.mean_u8; },
                    input, output, shape);
            case QCORE_DTYPE_FLOAT64:
                // Metal has no float64 arithmetic, so neither a Vision kernel
                // nor the portable composition can run; the caller returns an
                // error.
                return status_dtype_unavailable;
            default:
                return status_unsupported_dtype;
        }
    }
}

extern "C" int vision_metal_block_spread(
    const void* gradient_output, void* gradient_input, std::uint64_t factor) {
    @autoreleasepool {
        if (qcore_tensor_dtype(gradient_output) != QCORE_DTYPE_FLOAT32)
            return status_unsupported_dtype;
        BlockShape shape{};
        if (!block_shape(gradient_input, gradient_output, factor,
                         qcore_tensor_element_count(gradient_input), shape))
            return status_backend_unavailable;
        return dispatch(
            [](VisionMetalPrograms& programs) { return programs.spread_f32; },
            gradient_output, gradient_input, shape);
    }
}
