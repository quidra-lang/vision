#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <quidra/native_extension.h>

#include <cstdint>
#include <limits>
#include <map>
#include <mutex>

// Vision-owned Metal kernels. Core lends its command stream and tensor buffers
// through the native-extension ABI; the image-domain semantics stay here.
// native/vision_native.cpp validates dtype, device, layout, and block geometry
// before calling these entry points.

namespace {

// Status values shared with native/vision_native.cpp (BlockStatus).
constexpr int status_ok = 0;
constexpr int status_unsupported_dtype = 2;
constexpr int status_storage_unavailable = 5;
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
    device atomic_uint* status [[buffer(3)]],
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
    const float value = total / shape.divisor;
    output[index] = value;
    if ((as_type<uint>(value) & 0x7fffffffu) > 0x7f800000u)
        atomic_fetch_or_explicit(status, 1u, memory_order_relaxed);
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

template <typename T>
T bridge_handle(std::uint64_t handle) {
    return (__bridge T)(reinterpret_cast<void*>(
        static_cast<std::uintptr_t>(handle)));
}

// The device's MTLDevice, for compiling Vision's pipelines. A lookup that
// lends no queue, so it is allowed anywhere, also while Core's stream is held.
id<MTLDevice> native_device(long long device) {
    const auto handle = qcore_device_native_device(device);
    return handle == 0 ? nil : bridge_handle<id<MTLDevice>>(handle);
}

// A tensor buffer lent by Core, taken before the encoder scope opens: taking
// it may make Core work for it (a copy-on-write detach, an upload, storage of
// its own), which Core refuses inside the scope.
struct Borrowed {
    id<MTLBuffer> buffer = nil;
    NSUInteger offset = 0;

    explicit operator bool() const {
        return buffer != nil;
    }
};

Borrowed borrowed(const void* tensor, std::uint64_t handle) {
    if (handle == 0) return {};
    return {bridge_handle<id<MTLBuffer>>(handle),
            static_cast<NSUInteger>(qcore_tensor_device_offset_bytes(tensor))};
}

Borrowed borrow_const(const void* tensor) {
    return borrowed(tensor, qcore_tensor_device_handle_const(tensor));
}

Borrowed borrow_mutable(void* tensor) {
    return borrowed(tensor, qcore_tensor_device_handle(tensor));
}

// A write-only output: the kernel stores every element, so Core lends
// uninitialized storage with no fill (qcore_tensor_output_handle), and the
// output is marked written once the kernel is encoded. Only the block mean's
// forward output is write-only; a gradient Core lends to a backward callback
// is taken with borrow_mutable.
Borrowed borrow_written(void* tensor, bool write_only) {
    return write_only
        ? borrowed(tensor, qcore_tensor_output_handle(tensor))
        : borrow_mutable(tensor);
}

// One Vision submission on a Metal device.
//
// On Core's command stream it follows the encode-scope protocol of
// quidra/native_extension.h: the constructor holds the device's stream, the
// caller then takes every tensor handle, opens the scope (compute_encoder()),
// sets state and dispatches, and close() ends scope and hold with one
// qcore_metal_note_work. The work joins Core's open command buffer, which
// Core commits with its own work and waits for only where the host reads, so
// Vision never commits or waits.
//
// Without a Core stream (QUIDRA_METAL_STREAM=0) the same calls fill a command
// buffer of Vision's own on Core's queue, and close() commits it without
// waiting; Core's synchronization covers command buffers committed on its
// queue.
//
// A submission that is not closed (an early return) ends the hold with no
// work, and an own command buffer is dropped uncommitted.
class MetalSubmission {
public:
    explicit MetalSubmission(long long device)
        : device_(device), held_(qcore_device_encode_begin(device) != 0) {}

    MetalSubmission(const MetalSubmission&) = delete;
    MetalSubmission& operator=(const MetalSubmission&) = delete;

    ~MetalSubmission() {
        if (closed_) return;
        if (held_) {
            qcore_metal_note_work(device_, 0, 0);
        } else if (own_encoder_) {
            [encoder_ endEncoding];
        }
    }

    // Opens the compute-encoder scope. On Core's stream this is Core's open
    // encoder (serial dispatch, tracked resources), which Vision never ends.
    id<MTLComputeCommandEncoder> compute_encoder() {
        if (encoder_) return encoder_;
        if (held_) {
            const auto handle = qcore_device_compute_encoder(device_);
            if (handle != 0)
                encoder_ = bridge_handle<id<MTLComputeCommandEncoder>>(handle);
            return encoder_;
        }
        if (!command_) {
            const auto handle = qcore_device_queue_handle(device_);
            if (handle == 0) return nil;
            command_ =
                [bridge_handle<id<MTLCommandQueue>>(handle) commandBuffer];
            if (!command_) return nil;
        }
        encoder_ = [command_ computeCommandEncoder];
        own_encoder_ = encoder_ != nil;
        return encoder_;
    }

    // Ends the submission after a successful encode. `dispatches` and
    // `bytes` feed Core's commit policy.
    void close(std::uint32_t dispatches, std::uint64_t bytes) {
        if (closed_) return;
        closed_ = true;
        if (held_) {
            qcore_metal_note_work(device_, dispatches, bytes);
            return;
        }
        if (own_encoder_) [encoder_ endEncoding];
        if (command_) [command_ commit];
    }

private:
    long long device_;
    bool held_;
    bool closed_ = false;
    bool own_encoder_ = false;
    id<MTLCommandBuffer> command_ = nil;
    id<MTLComputeCommandEncoder> encoder_ = nil;
};

// A slot is taken before the encoder opens. After submission Core owns
// the deferred check and reports it at the next synchronization boundary.
class DeviceStatus {
public:
    explicit DeviceStatus(long long device) : device_(device) {}
    DeviceStatus(const DeviceStatus&) = delete;
    DeviceStatus& operator=(const DeviceStatus&) = delete;
    ~DeviceStatus() {
        if (taken_) qcore_device_status_release(device_, slot_);
    }
    bool take() {
        std::uint64_t handle = 0;
        if (qcore_device_status_slot(device_, &handle, &offset_, &slot_) != 0)
            return false;
        taken_ = true;
        buffer_ = bridge_handle<id<MTLBuffer>>(handle);
        return buffer_ != nil;
    }
    id<MTLBuffer> buffer() const { return buffer_; }
    NSUInteger offset() const { return static_cast<NSUInteger>(offset_); }
    bool defer() {
        if (!taken_) return false;
        // Core consumes the slot on successful handoff only.
        if (qcore_device_defer_status(device_, slot_,
                "package operation vision.downsample_mean produced a NaN real32 value") == 0) {
            taken_ = false;
            return true;
        }
        // close() has submitted the writer. Consume its valid slot only
        // after the queued writes finish; wait also releases the slot.
        waited_ = qcore_device_status_wait(device_, slot_, &value_) == 0;
        taken_ = false;
        return false;
    }
    bool failed_check() const { return waited_ && value_ != 0; }
private:
    long long device_;
    bool taken_ = false;
    bool waited_ = false;
    std::uint32_t value_ = 0;
    std::uint64_t offset_ = 0;
    std::uint64_t slot_ = 0;
    id<MTLBuffer> buffer_ = nil;
};

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

// Encodes one kernel on the device of the tensors, in program order with
// Core's work: after the work that produced `source` and before later work
// on `destination`. Vision's pipelines are compiled for the device before
// its stream is held. Nothing is committed or waited for here; Core runs the
// work with its own and synchronizes where the host reads, and it reports a
// failed command buffer at that synchronization point. Every kernel stores
// each element of `destination`; a `write_only` destination is marked written
// once the kernel is encoded.
int dispatch(
    id<MTLComputePipelineState> (*select)(VisionMetalPrograms&),
    const void* source,
    void* destination,
    bool write_only,
    const BlockShape& shape,
    std::uint64_t bytes,
    bool validates_nonfinite = false) {
    if (shape.count == 0) return status_ok;
    const auto device = qcore_tensor_device(source);
    if (device < 0) return status_backend_unavailable;
    auto* programs = programs_for(native_device(device));
    if (!programs) return status_backend_failed;
    id<MTLComputePipelineState> pipeline = select(*programs);
    if (!pipeline) return status_backend_failed;

    MetalSubmission submission(device);
    const Borrowed input = borrow_const(source);
    const Borrowed output = borrow_written(destination, write_only);
    // Core refuses the handle of a view that contains an uninitialized
    // element; the caller then takes the portable composition, which reads
    // through Core and reports the uninitialized read.
    if (!input || !output) return status_storage_unavailable;
    DeviceStatus status(device);
    if (validates_nonfinite && !status.take()) return status_backend_failed;
    id<MTLComputeCommandEncoder> encoder = submission.compute_encoder();
    if (!encoder) return status_backend_failed;
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:input.buffer offset:input.offset atIndex:0];
    [encoder setBuffer:output.buffer offset:output.offset atIndex:1];
    [encoder setBytes:&shape length:sizeof(shape) atIndex:2];
    if (validates_nonfinite)
        [encoder setBuffer:status.buffer() offset:status.offset() atIndex:3];
    [encoder dispatchThreads:MTLSizeMake(shape.count, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(thread_count(pipeline), 1, 1)];
    submission.close(1, bytes);
    if (validates_nonfinite && !status.defer()) {
        // The writer and slot are retired before failure reporting can unwind.
        if (status.failed_check())
            qcore_tensor_report_status(destination, "vision.downsample_mean", 1);
        return status_backend_failed;
    }
    if (write_only && qcore_tensor_mark_written(destination) != 0)
        return status_backend_failed;
    return status_ok;
}

// Bytes a block kernel reads and writes, for Core's commit policy.
std::uint64_t block_bytes(const void* full, const void* reduced,
                          std::uint64_t element_size) {
    return (qcore_tensor_element_count(full) +
            qcore_tensor_element_count(reduced)) * element_size;
}

} // namespace

extern "C" int vision_metal_block_mean(
    const void* input, void* output, std::uint64_t factor, int write_only) {
    @autoreleasepool {
        BlockShape shape{};
        if (!block_shape(input, output, factor,
                         qcore_tensor_element_count(output), shape))
            return status_backend_unavailable;
        switch (qcore_tensor_dtype(input)) {
            case QCORE_DTYPE_FLOAT32:
                return dispatch(
                    [](VisionMetalPrograms& programs) { return programs.mean_f32; },
                    input, output, write_only != 0, shape,
                    block_bytes(input, output, sizeof(float)), true);
            case QCORE_DTYPE_UINT8:
                return dispatch(
                    [](VisionMetalPrograms& programs) { return programs.mean_u8; },
                    input, output, write_only != 0, shape,
                    block_bytes(input, output, sizeof(std::uint8_t)));
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
            gradient_output, gradient_input, false, shape,
            block_bytes(gradient_input, gradient_output, sizeof(float)));
    }
}
