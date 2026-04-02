"""
OpenCL — Open Computing Language: Portable Heterogeneous Parallel Programming
===============================================================================

OpenCL (Open Computing Language) is the first open, royalty-free standard for
cross-platform, parallel programming of heterogeneous systems. Ratified in 2008
by the Khronos Group and backed by Apple, NVIDIA, AMD, Intel, ARM, and IBM,
OpenCL answered a question CUDA could not: how do you write one parallel
program that runs on ANY vendor's GPU, CPU, FPGA, or DSP?

The central idea: abstract hardware behind a layered model. Instead of writing
code to a specific chip's ISA, you write to a portable execution model — the
NDRange — that maps to whatever physical device you have. A kernel written
for an NVIDIA A100 compiles and runs, without source changes, on an AMD RX 7900,
an Intel Arc GPU, an Apple M3 GPU, a Qualcomm mobile DSP, or a multi-core CPU.
This portability is OpenCL's defining characteristic and its primary reason
for existence.

The cost of this abstraction is precision: OpenCL cannot expose every quirk of
every device. It therefore defines a rigorous abstract hardware model — the
platform model, execution model, memory model, and programming model — that
every conforming implementation must honour. Understanding these four models
is the complete theory of OpenCL.

This module covers the platform/device/context/queue stack, the NDRange and
work-group execution model, the four-tier memory hierarchy, the runtime
compilation pipeline, synchronization with barriers and events, buffer
management, the roofline model for heterogeneous devices, and every major
optimization technique. Every concept is backed by a runnable simulation.

"""

import textwrap
import re

TOPIC_NAME   = "OpenCL — Portable Heterogeneous Parallel Computing"
DISPLAY_NAME = "20 · OpenCL"
ICON         = "🔁"
SUBTITLE     = "NDRange, Work-Groups, and the Memory Model — One Kernel, Any Device"

# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

##### PART 1 — OPENCL VS CUDA: PORTABILITY VS PERFORMANCE

### The Problem OpenCL Solves

    In 2006 NVIDIA shipped CUDA — a brilliant GPU programming model, but one
    permanently tied to NVIDIA hardware. A CUDA program simply will not compile
    for any other chip. For companies deploying on cloud instances (which mix
    NVIDIA, AMD, and Intel GPUs), embedded systems (ARM Mali, Qualcomm Adreno),
    FPGAs (Xilinx, Intel), or supercomputers (Cray, Fujitsu), CUDA is a dead end.

    OpenCL's answer: decouple the programming model from the hardware.

        CUDA:   programmer → NVCC → PTX assembly → NVIDIA GPU only
        OpenCL: programmer → runtime compiler → ANY device's ISA

### Terminology Mapping: OpenCL to CUDA

    The concepts are identical; the names differ:

    ┌─────────────────────────────┬────────────────────────────┐
    │  OpenCL                     │  CUDA                      │
    ├─────────────────────────────┼────────────────────────────┤
    │  Work-item                  │  Thread                    │
    │  Work-group                 │  Block                     │
    │  NDRange                    │  Grid                      │
    │  Compute unit (CU)          │  Streaming Multiprocessor  │
    │  Processing element (PE)    │  CUDA core                 │
    │  Local memory               │  Shared memory             │
    │  Private memory             │  Registers                 │
    │  Global memory              │  Global (VRAM/HBM)         │
    │  Constant memory            │  Constant memory           │
    │  Kernel                     │  Kernel (__global__)       │
    │  Command queue              │  CUDA stream               │
    │  Buffer (cl_mem)            │  Device pointer (float*)   │
    │  Program + clBuildProgram   │  NVCC compilation          │
    │  Platform                   │  (no direct equivalent)    │
    │  Context                    │  CUDA context              │
    │  get_local_id(0)            │  threadIdx.x               │
    │  get_group_id(0)            │  blockIdx.x                │
    │  get_global_id(0)           │  blockIdx.x*blockDim.x     │
    │                             │    + threadIdx.x           │
    │  barrier(CLK_LOCAL_MEM_FENCE)│  __syncthreads()          │
    └─────────────────────────────┴────────────────────────────┘

### Key Differences in Practice

    1. RUNTIME COMPILATION
       CUDA compiles to PTX at build time (or is pre-compiled for specific GPUs).
       OpenCL compiles kernels AT RUNTIME using clBuildProgram().
       Consequence: first launch includes compilation overhead (100ms–several seconds).
       Benefit: the compiler targets the exact GPU that's present on that machine.

    2. EXPLICIT DEVICE DISCOVERY
       In CUDA: the runtime picks the device; you query with cudaGetDeviceProperties.
       In OpenCL: you enumerate platforms, then devices per platform, then choose.
       This makes multi-vendor code possible but adds boilerplate.

    3. EXPLICIT BUFFER MANAGEMENT
       CUDA: cudaMalloc + cudaMemcpy, or unified memory with cudaMallocManaged.
       OpenCL: clCreateBuffer, then clEnqueueWriteBuffer / clEnqueueReadBuffer.
       OpenCL's model is explicitly host-device; there is no equivalent of CUDA's
       Unified Memory in the core OpenCL 1.x/2.x spec (though extensions exist).

    4. PORTABILITY TAX ON PEAK PERFORMANCE
       On NVIDIA hardware:      CUDA typically 5–15% faster than OpenCL
       On AMD hardware:         OpenCL often matches or beats CUDA (ROCm/HIP)
       On Intel GPU/CPU:        OpenCL is the ONLY GPU API (no CUDA)
       On ARM/mobile:           OpenCL is UNIVERSAL (Vulkan compute also applies)
       Verdict: for pure NVIDIA deployment, use CUDA. For everything else, OpenCL.


##### PART 2 — THE PLATFORM MODEL

### The Four-Layer Stack

    OpenCL separates hardware abstraction into four nested layers:

    PLATFORM → DEVICE → CONTEXT → COMMAND QUEUE

    ┌──────────────────────────────────────────────────────────────────────┐
    │  HOST (CPU running your application)                                 │
    │                                                                      │
    │  ┌────────────────────┐   ┌────────────────────┐                    │
    │  │  Platform 0        │   │  Platform 1        │                    │
    │  │  (NVIDIA OpenCL)   │   │  (Intel OpenCL)    │  ← clGetPlatformIDs│
    │  │                    │   │                    │                    │
    │  │  ┌─────────────┐   │   │  ┌─────────────┐  │                    │
    │  │  │ Device 0    │   │   │  │ Device 0    │  │  ← clGetDeviceIDs  │
    │  │  │ (A100 GPU)  │   │   │  │ (Xeon CPU)  │  │                    │
    │  │  └──────┬──────┘   │   │  └──────┬──────┘  │                    │
    │  └─────────┼──────────┘   └─────────┼─────────┘                    │
    │            │                        │                               │
    │  ┌─────────▼────────────────────────▼────────────────────────────┐  │
    │  │  Context  (shared memory space, devices can interact here)    │  │
    │  │                                                               │  │
    │  │  ┌──────────────────┐   ┌───────────────────────────────────┐│  │
    │  │  │ Command Queue 0  │   │ Command Queue 1                   ││  │
    │  │  │ (Device 0: GPU)  │   │ (Device 1: CPU, or same GPU)      ││  │
    │  │  │ In-order         │   │ Out-of-order                      ││  │
    │  │  └──────────────────┘   └───────────────────────────────────┘│  │
    │  │                                                               │  │
    │  │  ┌───────────────────────────────────────────────────────────┐│  │
    │  │  │ Buffers / Images (cl_mem objects)                         ││  │
    │  │  └───────────────────────────────────────────────────────────┘│  │
    │  └───────────────────────────────────────────────────────────────┘  │
    └──────────────────────────────────────────────────────────────────────┘

### Platform

    A platform represents one vendor's OpenCL implementation.
    On a machine with both NVIDIA and Intel OpenCL drivers installed,
    clGetPlatformIDs() returns TWO platform objects.

    Each platform provides:
        ● Its own compiler (for kernels targeting its devices)
        ● Its own runtime (memory management, queue scheduling)
        ● Its own extensions (vendor-specific features like cl_intel_subgroups)

    Key platform properties:
        CL_PLATFORM_NAME:    "NVIDIA CUDA", "Intel(R) OpenCL", "AMD Accelerated..."
        CL_PLATFORM_VERSION: "OpenCL 3.0 CUDA 12.2.148"
        CL_PLATFORM_EXTENSIONS: list of extension strings

### Device

    A device is one physical processing unit: a GPU, CPU, FPGA, or DSP.
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, ...) returns all GPUs
    from that platform.

    Critical device properties to query before tuning:
        CL_DEVICE_MAX_COMPUTE_UNITS:        number of CUs (like SM count)
        CL_DEVICE_MAX_WORK_GROUP_SIZE:      max work-items per work-group
        CL_DEVICE_LOCAL_MEM_SIZE:           bytes of local (shared) memory
        CL_DEVICE_GLOBAL_MEM_SIZE:          total VRAM
        CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS: max NDRange dimensions (usually 3)
        CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT: preferred SIMD width for float
        CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE: constant memory size
        CL_DEVICE_EXTENSIONS:              "cl_khr_fp64 cl_khr_int64_base_atomics..."

    Example device properties for real hardware:
        AMD RX 7900 XTX:  CUs=96, local_mem=65536 B, global_mem=24 GB
        NVIDIA A100:      CUs=108 (SMs), local_mem=49152 B, global_mem=80 GB
        Intel Arc A770:   CUs=32, local_mem=65536 B, global_mem=16 GB
        Apple M3 GPU:     CUs=18, local_mem=32768 B, (unified memory)

### Context

    A context is a shared execution environment that can span multiple devices
    of the same platform (NOT across platforms — that requires separate contexts).

    Resources created in a context (buffers, programs, kernels) can be accessed
    by any device within that context.

    Creating a context for a single GPU:
        cl_context ctx = clCreateContext(NULL, 1, &device, NULL, NULL, &err);

    Creating a context for both a CPU and GPU device:
        cl_device_id devices[2] = {gpu, cpu};
        cl_context ctx = clCreateContext(NULL, 2, devices, NULL, NULL, &err);

### Command Queue

    Work is submitted to devices through command queues, not directly.
    Each queue is bound to ONE device and submits work in FIFO order.

    Two queue modes:
        IN-ORDER (default): commands execute in submission order.
            clEnqueueWriteBuffer → clEnqueueNDRangeKernel → clEnqueueReadBuffer
            Each waits for the previous to complete.

        OUT-OF-ORDER: commands may execute concurrently.
            Explicit event dependencies are used to express ordering.
            Allows overlapping data transfer with computation.

    cl_command_queue queue = clCreateCommandQueueWithProperties(
        ctx, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err);

    Queue operations:
        clEnqueueNDRangeKernel   — execute a kernel
        clEnqueueWriteBuffer     — host → device data transfer
        clEnqueueReadBuffer      — device → host data transfer
        clEnqueueCopyBuffer      — device → device copy
        clEnqueueMarker          — insert a sync point
        clFinish                 — block host until all queued commands complete


##### PART 3 — THE EXECUTION MODEL: NDRANGE, WORK-GROUPS, AND WORK-ITEMS

### The NDRange

    An NDRange (N-Dimensional Range) is the global index space for a kernel launch.
    It is 1D, 2D, or 3D. Every work-item has a unique global ID within this space.

    Syntax in the host code:
        size_t global[2] = {1024, 1024};   // 2D NDRange: 1024 × 1024 work-items
        size_t local[2]  = {16, 16};       // work-group: 16 × 16 = 256 work-items
        clEnqueueNDRangeKernel(queue, kernel, 2,
                               NULL,    // global offset (usually NULL)
                               global,  // global size
                               local,   // local (work-group) size
                               0, NULL, NULL);

    The NDRange is divided into work-groups, each of size local[i].
    Constraint: global[i] MUST be divisible by local[i] for each dimension.

    Requirement:
        global[0] % local[0] == 0    (else clEnqueueNDRangeKernel returns error)
        global[1] % local[1] == 0    (or use padding / boundary checks)

### Work-Item Identity

    Inside a kernel, every work-item knows its position:

        get_global_id(0):    x index in the full NDRange  (≡ blockIdx.x*blockDim.x + threadIdx.x)
        get_global_id(1):    y index in the full NDRange
        get_local_id(0):     x index within the work-group  (≡ threadIdx.x)
        get_local_id(1):     y index within the work-group
        get_group_id(0):     work-group x index  (≡ blockIdx.x)
        get_group_id(1):     work-group y index
        get_global_size(0):  total NDRange x dimension
        get_local_size(0):   work-group x dimension
        get_num_groups(0):   number of work-groups in x direction

    Canonical 1D kernel pattern:
        __kernel void add(__global float* A, __global float* B, __global float* C,
                          int N) {
            int gid = get_global_id(0);
            if (gid < N)
                C[gid] = A[gid] + B[gid];
        }

    Canonical 2D kernel pattern (image processing, matrix operations):
        __kernel void process(__global float* img, int W, int H) {
            int x = get_global_id(0);
            int y = get_global_id(1);
            if (x < W && y < H) {
                int idx = y * W + x;
                img[idx] = img[idx] * 2.0f;
            }
        }

### Work-Group Constraints and the SIMD Engine

    A work-group executes on ONE compute unit (CU/SM).
    Work-items within a group may synchronize with each other via barriers.
    Work-items in DIFFERENT groups cannot synchronize (no global barrier in OpenCL 1.x/2.x).

    Under the hood, work-items are not individually scheduled.
    GPUs execute them in SIMD vectors (the equivalent of CUDA warps):
        NVIDIA (via OpenCL):  subgroup size = 32 (same as warp)
        AMD RDNA:             wave32 mode: subgroup = 32; wave64: subgroup = 64
        Intel Arc/Xe:         subgroup size = 8, 16, or 32 (device-dependent)
        ARM Mali-G series:    warp = 4 to 16 work-items

    OpenCL 2.0+ exposes subgroups explicitly:
        get_sub_group_id()        — which subgroup am I in?
        get_sub_group_local_id()  — my lane within the subgroup
        sub_group_barrier()       — lightweight barrier for subgroup only
        sub_group_reduce_add(x)   — warp-shuffle-style reduction (no shared mem)

### Work-Group Size Selection Rules

    Rule 1: Must be a multiple of the subgroup/warp size.
        NVIDIA: multiples of 32 (32, 64, 128, 256, 512)
        AMD:    multiples of 32 or 64 depending on wave mode
        Intel:  multiples of 8 or 16

    Rule 2: Must not exceed CL_DEVICE_MAX_WORK_GROUP_SIZE (typically 256-1024).

    Rule 3: Must leave enough local memory per CU for multiple groups.
        If local_mem_used / group = 32 KB and max = 64 KB → only 1 group/CU → low occupancy.

    Rule 4: For 2D/3D work: keep inner dimension (dim 0) = subgroup size.
        local[0]=16, local[1]=16 → 256 work-items, 16 in the fast (x) direction.
        WRONG:  local[0]=4, local[1]=64 → x-stride of 4 fragments coalescing.

    Query the optimal local size from the kernel itself:
        clGetKernelWorkGroupInfo(kernel, device,
            CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, ...);


##### PART 4 — THE MEMORY MODEL

### The Four Address Spaces

    OpenCL defines four distinct address spaces with different performance
    characteristics. Each must be declared explicitly in kernel code.

    ┌───────────────────────────────────────────────────────────────────────┐
    │  ADDRESS SPACE  │ QUALIFIER  │ LATENCY  │ BANDWIDTH  │ SCOPE         │
    ├───────────────────────────────────────────────────────────────────────┤
    │  Private        │ (none)     │ 1 cycle  │ ~50 TB/s   │ 1 work-item   │
    │  Local          │ __local    │ 1–5 cy   │ ~20 TB/s   │ 1 work-group  │
    │  Constant       │ __constant │ 1–5 cy   │ ~20 TB/s   │ read-only all │
    │  Global         │ __global   │ 300–700cy│ 1–3 TB/s   │ all items     │
    └───────────────────────────────────────────────────────────────────────┘

    __private  (register-equivalent):
        Declared as ordinary local variables inside the kernel.
        Lives in the register file of each work-item.
        Fastest possible access (~1 cycle).
        Spills to global memory if register pressure is exceeded.

    __local  (shared memory equivalent):
        Declared with the __local qualifier OR as a kernel argument
        with __local attribute.
        Visible to ALL work-items in the same work-group.
        Survives the work-group's lifetime.
        Access requires barrier() to ensure consistency.

        __kernel void use_local(__global float* in, __global float* out) {
            __local float tile[256];                 // 256 × 4 = 1 KB
            int lid = get_local_id(0);
            int gid = get_global_id(0);
            tile[lid] = in[gid];                     // load global → local
            barrier(CLK_LOCAL_MEM_FENCE);            // ensure all loads complete
            out[gid] = tile[lid] + tile[(lid+1) % 256]; // use local memory
        }

    __constant  (constant cache):
        Read-only global data broadcast to all work-items.
        Cached in a dedicated constant cache (fast repeated reads).
        Maximum size: CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE (often 64 KB).
        Best for: lookup tables, filter kernels, transformation matrices.

        __constant float gauss_filter[9] = { 1,2,1, 2,4,2, 1,2,1 };  // in kernel

    __global  (device VRAM):
        All buffers passed to kernels live here.
        Largest address space (up to tens of GB).
        Highest latency (~300–700 cycles).
        Coalescing rules apply exactly as in CUDA global memory.

### The Memory Fence and Barrier

    Memory operations in OpenCL are NOT automatically ordered across work-items.
    You must use explicit fences and barriers.

    barrier(CLK_LOCAL_MEM_FENCE):
        All work-items in the work-group must reach this point.
        Ensures local memory writes by ANY work-item are visible to ALL.
        Roughly equivalent to __syncthreads() in CUDA.

    barrier(CLK_GLOBAL_MEM_FENCE):
        Same as above, but also flushes global memory.
        Heavier — use only when global writes need to be seen by the group.

    mem_fence(CLK_LOCAL_MEM_FENCE):
        Does NOT synchronize threads — only orders memory ops within ONE work-item.
        Useful when you need write-before-read guarantees in the same thread.

    CRITICAL RULE: If work-item A writes to local memory and work-item B reads it,
    there MUST be a barrier() between the write and the read, or the result is
    undefined (data race, undefined behavior in the OpenCL spec).

### Buffer Management and Transfer Strategy

    OpenCL manages device memory through cl_mem objects (buffer handles).

    Host → Device (upload):
        clEnqueueWriteBuffer(queue, buffer, CL_TRUE,   // CL_TRUE = blocking
            0, N*sizeof(float), host_ptr, 0, NULL, NULL);

    Device → Host (download):
        clEnqueueReadBuffer(queue, buffer, CL_TRUE,
            0, N*sizeof(float), host_ptr, 0, NULL, NULL);

    Device → Device copy:
        clEnqueueCopyBuffer(queue, src_buf, dst_buf, 0, 0, N*sizeof(float),
            0, NULL, NULL);

    Buffer creation flags:
        CL_MEM_READ_WRITE:   kernel can read and write (default)
        CL_MEM_READ_ONLY:    kernel reads only (allows optimization)
        CL_MEM_WRITE_ONLY:   kernel writes only (allows optimization)
        CL_MEM_USE_HOST_PTR: buffer is pinned host memory (zero-copy where possible)
        CL_MEM_COPY_HOST_PTR:copy host data immediately on buffer creation
        CL_MEM_ALLOC_HOST_PTR:allocate page-locked host memory for fast transfer

    Zero-copy with CL_MEM_USE_HOST_PTR (for integrated GPUs / CPU devices):
        float* data = malloc(N * sizeof(float));
        cl_mem buf  = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                                     N*sizeof(float), data, &err);
        // On Intel HD / AMD APU: GPU reads directly from CPU RAM — no transfer needed.
        // On discrete GPU: this is inefficient — the GPU must DMA from system RAM.

    Map/Unmap API (efficient ping-pong between host and device):
        float* ptr = (float*)clEnqueueMapBuffer(queue, buf, CL_TRUE, CL_MAP_READ,
                                                0, N*sizeof(float), 0, NULL, NULL, &err);
        // Read/modify ptr on host...
        clEnqueueUnmapMemObject(queue, buf, ptr, 0, NULL, NULL);


##### PART 5 — THE PROGRAMMING MODEL: KERNELS AND THE COMPILATION PIPELINE

### OpenCL C Kernel Language

    OpenCL kernels are written in OpenCL C — a strict subset of C99 with
    extensions for parallelism. Key differences from standard C:

    QUALIFIERS:
        __kernel:    marks a function as a kernel entry point (callable from host)
        __global:    pointer to global memory
        __local:     pointer to/variable in local memory
        __constant:  pointer to constant memory
        __private:   (default) pointer to private registers

    BUILT-IN FUNCTIONS:
        Math:    native_sin, native_cos, native_sqrt, native_recip, mad, fma
        Atomic:  atomic_add, atomic_xchg, atomic_cmpxchg (global and local)
        Vector:  float4 v = (float4)(a, b, c, d);  v.x, v.y, v.xy, v.wzyx
        Shuffle: shuffle(v, mask)

    VECTOR TYPES (SIMD within one work-item):
        float2, float4, float8, float16  (and int, char, short variants)
        float4 a = (float4)(1.0f, 2.0f, 3.0f, 4.0f);
        float4 b = a * a;            // element-wise: all 4 muls in one instruction
        float sum = a.x + a.y + a.z + a.w;

    Example: SAXPY kernel (y = alpha*x + y)
        __kernel void saxpy(__global float* x, __global float* y,
                            float alpha, int N) {
            int gid = get_global_id(0);
            if (gid < N)
                y[gid] = alpha * x[gid] + y[gid];
        }

    Example: local memory tiled matrix multiply
        __kernel void matmul(__global float* A, __global float* B, __global float* C,
                             int M, int K, int N) {
            __local float As[TILE][TILE];
            __local float Bs[TILE][TILE];
            int ty = get_local_id(1), tx = get_local_id(0);
            int row = get_group_id(1)*TILE + ty;
            int col = get_group_id(0)*TILE + tx;
            float sum = 0.0f;
            for (int t = 0; t < K/TILE; t++) {
                As[ty][tx] = A[row*K + t*TILE + tx];
                Bs[ty][tx] = B[(t*TILE + ty)*N + col];
                barrier(CLK_LOCAL_MEM_FENCE);
                for (int k = 0; k < TILE; k++)
                    sum += As[ty][k] * Bs[k][tx];
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            C[row*N + col] = sum;
        }

### The Runtime Compilation Pipeline

    Unlike CUDA (which pre-compiles with NVCC), OpenCL compiles at runtime:

    1. clCreateProgramWithSource(ctx, 1, &kernel_source, NULL, &err)
       → Creates a program object from a string of OpenCL C source code.

    2. clBuildProgram(program, 1, &device, "-cl-fast-relaxed-math", NULL, NULL)
       → Compiles and links the program for the specified device(s).
       → The vendor's JIT compiler runs HERE (LLVM, IGC, etc.).
       → Compilation flags:
           -cl-fast-relaxed-math      enables flush-to-zero, relaxed precision
           -cl-mad-enable             allows fused multiply-add
           -cl-opt-disable            disable optimizations (for debugging)
           -cl-no-signed-zeros        enables reordering of FP operations
           -DTILE=16 -DDEBUG=1        preprocessor defines, like NVCC -D

    3. clCreateKernel(program, "saxpy", &err)
       → Creates a callable kernel object from the built program.

    4. clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf_x)  // arg 0
       clSetKernelArg(kernel, 1, sizeof(cl_mem), &buf_y)  // arg 1
       clSetKernelArg(kernel, 2, sizeof(float),  &alpha)  // arg 2
       → Sets each kernel argument individually.

    5. clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local, 0, NULL, &event)
       → Enqueues the kernel for execution.

    Error recovery from compilation:
        if (err != CL_SUCCESS) {
            size_t log_size;
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
            char* log = malloc(log_size);
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
            printf("Build error:\\n%s\\n", log);
        }

    Binary caching: save and reload compiled binaries to avoid recompilation:
        clGetProgramInfo(program, CL_PROGRAM_BINARIES, ...)  // save binary
        clCreateProgramWithBinary(...)                        // reload binary


##### PART 6 — SYNCHRONIZATION: EVENTS, BARRIERS, AND PROFILING

### Event Objects: Fine-Grained Synchronization

    Every enqueue call can produce a cl_event object that represents the
    completion of that specific command. Events can be:
        1. Waited on before another command executes (dependency chain)
        2. Queried for status (CL_QUEUED, CL_SUBMITTED, CL_RUNNING, CL_COMPLETE)
        3. Used for profiling (extract start/end timestamps)

    Event-based dependency chain (non-blocking pattern):
        cl_event write_done, kernel_done;

        // Async upload — don't block host
        clEnqueueWriteBuffer(queue, buf, CL_FALSE, 0, N*4, host_data,
                             0, NULL, &write_done);

        // Kernel waits for upload to finish
        clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local,
                               1, &write_done, &kernel_done);

        // Download waits for kernel to finish
        clEnqueueReadBuffer(queue, buf, CL_FALSE, 0, N*4, out_data,
                            1, &kernel_done, NULL);

        clFlush(queue);        // submit all commands
        clFinish(queue);       // block until all done

### Event Profiling: Measuring Kernel Time

    Enable profiling on the command queue:
        cl_command_queue queue = clCreateCommandQueueWithProperties(ctx, device,
            CL_QUEUE_PROFILING_ENABLE, &err);

    Extract timestamps (nanoseconds since epoch):
        cl_ulong t_submit, t_start, t_end;
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_SUBMIT, sizeof(cl_ulong), &t_submit, NULL);
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START,  sizeof(cl_ulong), &t_start,  NULL);
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,    sizeof(cl_ulong), &t_end,    NULL);

        double queue_latency_ms = (t_start - t_submit) / 1e6;
        double execution_ms     = (t_end   - t_start)  / 1e6;

    From this, compute:
        Achieved bandwidth:  N_bytes / execution_s  (in GB/s)
        Achieved GFLOPS:     N_flops / execution_s

### Kernel-Level Barrier vs Work-Group Barrier

    barrier(CLK_LOCAL_MEM_FENCE):
        Every work-item in the work-group hits this point before any proceeds.
        Scope: within one work-group. No effect on other groups.
        Cost: typically 5–30 cycles (warp-synchronous if the whole warp hits it).

    OpenCL 2.0 work_group_barrier(CLK_LOCAL_MEM_FENCE, memory_scope_work_group):
        Explicit scope argument (work_group, device, all_svm_devices).
        Equivalent to the above for GPU use.

    Global synchronization (between work-groups) requires splitting into two kernels:
        Kernel 1: compute partial results, write to global buffer
        Kernel 2: reduce partial results to final answer
        Barrier between the kernels is clFinish() or an event dependency.

    This is WHY multi-pass algorithms (prefix sum, sort, reduce) require multiple
    kernel launches in OpenCL — there is no device-wide __syncthreads().


##### PART 7 — OPENCL OPTIMIZATION: DEVICE-PORTABLE BEST PRACTICES

### 1. Work-Group Size Tuning (Most Impactful)

    Use CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE to auto-tune:
        size_t preferred;
        clGetKernelWorkGroupInfo(kernel, device,
            CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
            sizeof(size_t), &preferred, NULL);

    For NVIDIA: preferred = 32
    For AMD:    preferred = 32 or 64
    For Intel:  preferred = 8, 16, or 32
    For CPU:    preferred = 1 (no SIMD benefit from large groups on CPU)

### 2. Coalesced Global Memory Access

    Same rule as CUDA: consecutive work-items should access consecutive addresses.
        Coalesced:   buf[get_global_id(0)]             → 1 transaction per wave
        Strided:     buf[get_global_id(0) * stride]    → stride transactions per wave
        Column-major matrix access: buf[col * N + row] → stride = N (BAD)

    Fix with local memory tiling: same as CUDA's shared memory tile trick.

### 3. Local Memory Tiling

    For any kernel with O(N) reuse of input data:
        ● Load data tile into __local memory once (globally coalesced)
        ● Perform computation from __local (fast, low-latency)
        ● barrier() between load and compute

### 4. Vector Types for ILP

    OpenCL C vector types allow single work-items to compute multiple elements:
        float4 a = vload4(0, &buf[get_global_id(0) * 4]);  // load 4 floats
        float4 b = a * a + (float4)(1.0f);                  // 4 muls + 4 adds
        vstore4(b, 0, &out[get_global_id(0) * 4]);         // store 4 floats

    This doubles or quadruples instruction-level parallelism (ILP).
    On Intel GPU: float16 maps to a full 512-bit SIMD lane.
    On AMD:       float4/float8 improve ALU utilization.
    On NVIDIA:    less effective (CUDA's approach is preferred).

### 5. Constant Memory for Broadcast Data

    Any read-only data used identically by all work-items should be __constant:
        __constant float coeffs[16] = {...};  // cached in constant cache
    vs.
        __global const float* coeffs;         // goes through global cache

    Impact: 10–50% speedup for filter/convolution kernels.

### 6. Avoid Divergent Branching

    Same rules as CUDA. Prefer:
        float cond  = (float)(get_local_id(0) < 16);   // compute both paths
        float result = cond * path_a + (1.0f - cond) * path_b;  // select
    over:
        if (get_local_id(0) < 16) { ... } else { ... }   // SIMD divergence

### 7. Image Objects for 2D Spatial Locality

    For 2D data with spatial access patterns, use image2d_t instead of buffers:
        ● Hardware sampler handles boundary conditions automatically
        ● 2D spatial cache (better cache hit rates for non-row-major access)
        ● Built-in bilinear interpolation at hardware speed

        read_imagef(image, sampler, (int2)(x, y));

### 8. Async Data Transfer + Double Buffering

    Overlap PCIe transfer with kernel execution using two buffers:
        While kernel processes buffer[0]: transfer new data into buffer[1]
        While kernel processes buffer[1]: transfer results from buffer[0], upload next into buffer[0]

    Requires out-of-order queue or two queues with event synchronization.
    Peak effective throughput: compute + transfer happen simultaneously.
    Achievable utilization: can reach 95%+ of compute peak on long workloads.

"""

# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ── 1 ─────────────────────────────────────────────────────────────────────
    "1 · Platform Discovery Simulator — Enumerate Devices and Query Properties": {
        "description": (
            "Simulate the complete OpenCL platform/device discovery process. "
            "Model a realistic multi-vendor environment (NVIDIA, AMD, Intel), "
            "enumerate platforms and devices, query all key device properties, "
            "and implement a scoring heuristic to auto-select the best device "
            "for a compute workload. Show how to extract the properties that "
            "drive kernel tuning decisions."
        ),
        "language": "python",
        "code": '''
import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

print("=" * 65)
print("  OPENCL PLATFORM DISCOVERY SIMULATOR")
print("=" * 65)
print()

# ─────────────────────────────────────────────────────────────────────────
# Data model: Platform → Device hierarchy
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class OpenCLDevice:
    """
    Simulates the properties returned by clGetDeviceInfo().
    Every field corresponds to a real CL_DEVICE_* query constant.
    """
    name:                       str
    vendor:                     str
    device_type:                str          # GPU | CPU | ACCELERATOR
    compute_units:              int          # CL_DEVICE_MAX_COMPUTE_UNITS
    max_work_group_size:        int          # CL_DEVICE_MAX_WORK_GROUP_SIZE
    max_work_item_dims:         int          # CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS
    max_work_item_sizes:        Tuple        # CL_DEVICE_MAX_WORK_ITEM_SIZES
    preferred_wg_size_multiple: int          # per-kernel query (subgroup size)
    local_mem_size_bytes:       int          # CL_DEVICE_LOCAL_MEM_SIZE
    global_mem_size_bytes:      int          # CL_DEVICE_GLOBAL_MEM_SIZE
    global_mem_cache_size:      int          # CL_DEVICE_GLOBAL_MEM_CACHE_SIZE
    max_constant_buffer_size:   int          # CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE
    max_mem_alloc_size:         int          # CL_DEVICE_MAX_MEM_ALLOC_SIZE (per buffer)
    global_mem_bw_GBs:          float        # measured/spec peak global memory BW
    peak_gflops_fp32:           float        # theoretical peak FP32 GFLOPS
    opencl_version:             str          # CL_DEVICE_OPENCL_C_VERSION
    fp64_support:               bool         # cl_khr_fp64 extension
    fp16_support:               bool         # cl_khr_fp16 extension
    svm_support:                bool         # shared virtual memory (OpenCL 2.0)
    extensions:                 List[str]    # CL_DEVICE_EXTENSIONS

    @property
    def local_mem_KB(self):
        return self.local_mem_size_bytes // 1024

    @property
    def global_mem_GB(self):
        return self.global_mem_size_bytes / (1024**3)

    @property
    def ridge_point(self):
        """Arithmetic intensity at which the device transitions from memory to compute bound."""
        return self.peak_gflops_fp32 / (self.global_mem_bw_GBs / 1)  # GFLOP/s / GB/s = FLOP/byte

    @property
    def max_warps_estimate(self):
        """Estimate max concurrent warps (subgroups) per CU."""
        return 2048 // self.preferred_wg_size_multiple


@dataclass
class OpenCLPlatform:
    name:       str
    vendor:     str
    version:    str
    devices:    List[OpenCLDevice] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────
# Simulated hardware: three platforms, five devices
# ─────────────────────────────────────────────────────────────────────────

PLATFORMS = [
    OpenCLPlatform(
        name="NVIDIA CUDA",
        vendor="NVIDIA Corporation",
        version="OpenCL 3.0 CUDA 12.2.148",
        devices=[
            OpenCLDevice(
                name="NVIDIA A100-SXM4-80GB",
                vendor="NVIDIA Corporation",
                device_type="GPU",
                compute_units=108,
                max_work_group_size=1024,
                max_work_item_dims=3,
                max_work_item_sizes=(1024, 1024, 64),
                preferred_wg_size_multiple=32,
                local_mem_size_bytes=49152,           # 48 KB (OpenCL exposes 48 KB)
                global_mem_size_bytes=80 * 1024**3,
                global_mem_cache_size=40 * 1024**2,   # 40 MB L2
                max_constant_buffer_size=64 * 1024,
                max_mem_alloc_size=67 * 1024**3,
                global_mem_bw_GBs=2000,
                peak_gflops_fp32=19500,
                opencl_version="OpenCL C 1.2",
                fp64_support=True,
                fp16_support=True,
                svm_support=False,                    # NVIDIA OpenCL 1.x: no SVM
                extensions=["cl_khr_fp64", "cl_khr_fp16", "cl_nv_device_attribute_query",
                            "cl_nv_pragma_unroll", "cl_khr_int64_base_atomics",
                            "cl_khr_global_int32_base_atomics"],
            ),
        ]
    ),

    OpenCLPlatform(
        name="AMD Accelerated Parallel Processing",
        vendor="Advanced Micro Devices, Inc.",
        version="OpenCL 2.0 AMD-APP.3590.0",
        devices=[
            OpenCLDevice(
                name="AMD Radeon RX 7900 XTX",
                vendor="Advanced Micro Devices, Inc.",
                device_type="GPU",
                compute_units=96,
                max_work_group_size=1024,
                max_work_item_dims=3,
                max_work_item_sizes=(1024, 1024, 1024),
                preferred_wg_size_multiple=32,         # wave32 mode (RDNA 3)
                local_mem_size_bytes=65536,             # 64 KB
                global_mem_size_bytes=24 * 1024**3,
                global_mem_cache_size=6 * 1024**2,
                max_constant_buffer_size=64 * 1024,
                max_mem_alloc_size=15 * 1024**3,
                global_mem_bw_GBs=960,
                peak_gflops_fp32=61400,
                opencl_version="OpenCL C 2.0",
                fp64_support=True,
                fp16_support=True,
                svm_support=True,
                extensions=["cl_khr_fp64", "cl_khr_fp16", "cl_amd_device_attribute_query",
                            "cl_khr_subgroups", "cl_khr_int64_base_atomics",
                            "cl_khr_global_int32_base_atomics", "cl_amd_media_ops"],
            ),
            OpenCLDevice(
                name="AMD EPYC 7702 64-Core Processor",
                vendor="Advanced Micro Devices, Inc.",
                device_type="CPU",
                compute_units=128,                  # 64 cores × 2 HT
                max_work_group_size=4096,
                max_work_item_dims=3,
                max_work_item_sizes=(4096, 4096, 4096),
                preferred_wg_size_multiple=1,       # CPU: no SIMD benefit from WG size
                local_mem_size_bytes=32 * 1024,
                global_mem_size_bytes=512 * 1024**3,
                global_mem_cache_size=256 * 1024**2,  # 256 MB L3
                max_constant_buffer_size=128 * 1024,
                max_mem_alloc_size=128 * 1024**3,
                global_mem_bw_GBs=204,
                peak_gflops_fp32=4915,
                opencl_version="OpenCL C 2.0",
                fp64_support=True,
                fp16_support=False,
                svm_support=True,
                extensions=["cl_khr_fp64", "cl_khr_int64_base_atomics",
                            "cl_khr_global_int32_base_atomics"],
            ),
        ]
    ),

    OpenCLPlatform(
        name="Intel(R) OpenCL Graphics",
        vendor="Intel(R) Corporation",
        version="OpenCL 3.0",
        devices=[
            OpenCLDevice(
                name="Intel(R) Arc(TM) A770 Graphics",
                vendor="Intel(R) Corporation",
                device_type="GPU",
                compute_units=32,
                max_work_group_size=1024,
                max_work_item_dims=3,
                max_work_item_sizes=(1024, 1024, 1024),
                preferred_wg_size_multiple=16,        # Intel Xe: subgroup=16
                local_mem_size_bytes=65536,
                global_mem_size_bytes=16 * 1024**3,
                global_mem_cache_size=4 * 1024**2,
                max_constant_buffer_size=64 * 1024,
                max_mem_alloc_size=12 * 1024**3,
                global_mem_bw_GBs=560,
                peak_gflops_fp32=19660,
                opencl_version="OpenCL C 3.0",
                fp64_support=False,                   # Arc: limited FP64
                fp16_support=True,
                svm_support=True,
                extensions=["cl_khr_fp16", "cl_intel_subgroups", "cl_intel_required_subgroup_size",
                            "cl_khr_subgroups", "cl_khr_int64_base_atomics",
                            "cl_intel_planar_yuv", "cl_khr_3d_image_writes"],
            ),
        ]
    ),
]


# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Platform and device enumeration
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Platform and device enumeration")
print("━" * 65)
print()
print(f"  clGetPlatformIDs() → {len(PLATFORMS)} platform(s) found")
print()

for p_idx, platform in enumerate(PLATFORMS):
    print(f"  Platform {p_idx}: {platform.name}")
    print(f"    Vendor:  {platform.vendor}")
    print(f"    Version: {platform.version}")
    print(f"    clGetDeviceIDs() → {len(platform.devices)} device(s)")
    print()
    for d_idx, dev in enumerate(platform.devices):
        dtype_icon = {"GPU": "🖥️ ", "CPU": "🔲 ", "ACCELERATOR": "⚡ "}.get(dev.device_type, "")
        print(f"    Device {d_idx}: {dtype_icon} [{dev.device_type}] {dev.name}")
    print()


# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Detailed device property query
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Detailed device property query (all devices)")
print("━" * 65)
print()

all_devices = [(p.name, d) for p in PLATFORMS for d in p.devices]

props = [
    ("CL_DEVICE_MAX_COMPUTE_UNITS",        lambda d: f"{d.compute_units}"),
    ("CL_DEVICE_MAX_WORK_GROUP_SIZE",      lambda d: f"{d.max_work_group_size}"),
    ("CL_DEVICE_PREFERRED_WG_SIZE_MULT",   lambda d: f"{d.preferred_wg_size_multiple}  (subgroup/warp size)"),
    ("CL_DEVICE_LOCAL_MEM_SIZE",           lambda d: f"{d.local_mem_KB} KB"),
    ("CL_DEVICE_GLOBAL_MEM_SIZE",          lambda d: f"{d.global_mem_GB:.0f} GB"),
    ("CL_DEVICE_GLOBAL_MEM_CACHE_SIZE",    lambda d: f"{d.global_mem_cache_size//1024//1024} MB"),
    ("CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE", lambda d: f"{d.max_constant_buffer_size//1024} KB"),
    ("Peak FP32 (GFLOPS)",                 lambda d: f"{d.peak_gflops_fp32:,.0f}"),
    ("Peak Bandwidth (GB/s)",              lambda d: f"{d.global_mem_bw_GBs:,.0f}"),
    ("Ridge Point (FLOP/byte)",            lambda d: f"{d.ridge_point:.1f}"),
    ("CL_DEVICE_SVM_CAPABILITIES",         lambda d: "SVM ✅" if d.svm_support else "no SVM"),
    ("FP64 support",                       lambda d: "cl_khr_fp64 ✅" if d.fp64_support else "no FP64"),
    ("FP16 support",                       lambda d: "cl_khr_fp16 ✅" if d.fp16_support else "no FP16"),
]

# Header
hdr = f"  {'Property':<40}"
for pname, dev in all_devices:
    hdr += f" {dev.name[:22]:>22}"
print(hdr)
print("  " + "─" * (40 + 23 * len(all_devices)))
for label, fn in props:
    row = f"  {label:<40}"
    for _, dev in all_devices:
        row += f" {fn(dev):>22}"
    print(row)
print()


# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Device scoring for auto-selection
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Auto-select best device (scoring heuristic)")
print("━" * 65)
print()
print("  A real OpenCL app queries device properties at startup")
print("  and selects the best device for its workload type.")
print()

def score_device(dev: OpenCLDevice, workload: str) -> float:
    """
    Score a device for a given workload type.
    workload: "compute" | "bandwidth" | "latency" | "ml_training"
    """
    score = 0.0
    if workload == "compute":
        # Maximize compute throughput
        score  = dev.peak_gflops_fp32 * 0.6
        score += dev.compute_units    * 10.0
        score += dev.fp16_support     * 5000.0   # bonus for FP16

    elif workload == "bandwidth":
        # Maximize memory bandwidth
        score  = dev.global_mem_bw_GBs * 5.0
        score += dev.global_mem_GB     * 100.0

    elif workload == "latency":
        # Prefer CPU for lowest kernel launch latency
        score  = 10000.0 if dev.device_type == "CPU" else 1000.0
        score += dev.compute_units

    elif workload == "ml_training":
        # Maximize FP16 throughput and memory
        score  = dev.peak_gflops_fp32   * (2.0 if dev.fp16_support else 1.0)
        score += dev.global_mem_GB      * 500.0
        score += dev.global_mem_bw_GBs  * 3.0
        score -= (0 if dev.device_type == "GPU" else 5000.0)

    return score

WORKLOADS = ["compute", "bandwidth", "latency", "ml_training"]

print(f"  {'Workload':<16}  {'Best Device':<30}  {'Score':>8}  {'Reason'}")
print(f"  {'─'*78}")

for wl in WORKLOADS:
    scores = [(score_device(dev, wl), pname, dev) for pname, dev in all_devices]
    scores.sort(reverse=True)
    best_score, best_pname, best_dev = scores[0]
    reasons = {
        "compute":      "highest FP32 GFLOPS + FP16 bonus",
        "bandwidth":    "2 TB/s HBM2e bandwidth",
        "latency":      "CPU: lowest kernel launch overhead",
        "ml_training":  "FP16 + 80 GB VRAM + 2 TB/s BW",
    }
    print(f"  {wl:<16}  {best_dev.name[:30]:<30}  {best_score:>8.0f}  {reasons[wl]}")

print()


# ─────────────────────────────────────────────────────────────────────────
# SECTION 4: Work-group size constraints per device
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 4 — Valid work-group sizes for each device")
print("━" * 65)
print()
print("  Rule: local_size must be a multiple of preferred_wg_size_multiple")
print("  AND <= max_work_group_size AND leave >= 2 groups in local memory")
print()

CANDIDATE_WG_SIZES = [1, 8, 16, 32, 64, 128, 256, 512, 1024]

for _, dev in all_devices:
    valid = []
    for wgs in CANDIDATE_WG_SIZES:
        mult_ok  = (wgs % dev.preferred_wg_size_multiple == 0)
        size_ok  = (wgs <= dev.max_work_group_size)
        # Can at least 2 groups fit in local memory? (assume 8 KB local per group)
        mem_ok   = (dev.local_mem_size_bytes // 8192 >= 2)
        if mult_ok and size_ok and mem_ok:
            valid.append(str(wgs))
    print(f"  {dev.name[:35]:<35}  valid WG sizes: {', '.join(valid)}")
print()
print("  For NVIDIA (subgroup=32): powers-of-2 from 32 upward")
print("  For AMD RDNA (wave32):    same as NVIDIA in wave32 mode")
print("  For Intel Xe (sub=16):    multiples of 16 (16, 32, 64, 128, 256...)")
print("  For AMD CPU (sub=1):      any size — no SIMD constraint")
''',
    },

    # ── 2 ─────────────────────────────────────────────────────────────────────
    "2 · NDRange and Work-Group Execution — Mapping and Scheduling": {
        "description": (
            "Simulate the complete NDRange → work-group → work-item mapping in 1D, 2D, "
            "and 3D configurations. Show how work-groups are scheduled onto compute units, "
            "compute occupancy from OpenCL's four resource constraints (local memory, "
            "work-items, work-groups, registers), and demonstrate the wave/wavefront "
            "execution model including the AMD wave32/wave64 distinction. "
            "Show boundary-condition handling for non-divisible global sizes."
        ),
        "language": "python",
        "code": '''
import math
from dataclasses import dataclass
from typing import Tuple, List, Dict

print("=" * 65)
print("  NDRANGE AND WORK-GROUP EXECUTION SIMULATOR")
print("=" * 65)
print()

SUBGROUP_SIZE = 32   # simulate NVIDIA/AMD wave32 behavior

# ─────────────────────────────────────────────────────────────────────────
# NDRange work-item ID calculation
# ─────────────────────────────────────────────────────────────────────────

def get_global_id(global_id_flat: int, local_size: Tuple[int,...],
                  global_size: Tuple[int,...]) -> Tuple[int,...]:
    """Compute (gx, gy, gz) from a flat global index."""
    if len(global_size) == 1:
        return (global_id_flat,)
    elif len(global_size) == 2:
        gx = global_id_flat % global_size[0]
        gy = global_id_flat // global_size[0]
        return (gx, gy)
    else:
        gx  = global_id_flat % global_size[0]
        rem = global_id_flat // global_size[0]
        gy  = rem % global_size[1]
        gz  = rem // global_size[1]
        return (gx, gy, gz)

def work_item_ids(local_size: Tuple[int,...], global_size: Tuple[int,...],
                  group_id:   Tuple[int,...]) -> List[Dict]:
    """
    Enumerate all work-items in one work-group and compute their IDs,
    exactly as a real GPU would assign them.
    """
    ndim  = len(global_size)
    items = []
    if ndim == 1:
        for lx in range(local_size[0]):
            gx  = group_id[0] * local_size[0] + lx
            warp = lx // SUBGROUP_SIZE
            lane = lx %  SUBGROUP_SIZE
            items.append({
                "local":  (lx,),
                "global": (gx,),
                "group":  group_id,
                "subgrp": warp,
                "lane":   lane,
            })
    elif ndim == 2:
        for ly in range(local_size[1]):
            for lx in range(local_size[0]):
                gx = group_id[0] * local_size[0] + lx
                gy = group_id[1] * local_size[1] + ly
                flat_lid = ly * local_size[0] + lx
                warp = flat_lid // SUBGROUP_SIZE
                lane = flat_lid %  SUBGROUP_SIZE
                items.append({
                    "local":  (lx, ly),
                    "global": (gx, gy),
                    "group":  group_id,
                    "subgrp": warp,
                    "lane":   lane,
                })
    return items


# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: 1D NDRange — explicit ID mapping
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — 1D NDRange: ID mapping and subgroup structure")
print("━" * 65)
print()

global_1d = (64,)
local_1d  = (16,)
n_groups_1d = global_1d[0] // local_1d[0]

print(f"  NDRange: global_size=({global_1d[0]},), local_size=({local_1d[0]},)")
print(f"  Work-groups: {n_groups_1d}   Subgroups per group: {local_1d[0]//SUBGROUP_SIZE if local_1d[0] >= SUBGROUP_SIZE else 1}")
print()

# Show work-group 0 in detail
items_wg0 = work_item_ids(local_1d, global_1d, (0,))
print(f"  Work-group 0 detail (local_size={local_1d[0]}):")
print(f"  {'local_id':>10} | {'global_id':>10} | {'subgroup':>10} | {'lane':>6} | "
      f"{'buf access (coalesced?)':>24}")
print(f"  {'─'*68}")
for item in items_wg0:
    # Simulate accessing buf[get_global_id(0)]: consecutive → coalesced
    access = f"buf[{item['global'][0]}]"
    coal = "✅" if item['lane'] == item['local'][0] % SUBGROUP_SIZE else ""
    print(f"  {item['local'][0]:>10} | {item['global'][0]:>10} | {item['subgrp']:>10} | "
          f"{item['lane']:>6} | {access:>24}")
print()
print(f"  All work-items in subgroup 0 access consecutive addresses → coalesced ✅")
print()


# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: 2D NDRange — image processing configuration
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — 2D NDRange: image processing, column-major vs row-major")
print("━" * 65)
print()

IMAGE_W, IMAGE_H = 32, 16
local_2d_good = (16, 8)    # x-dim = 16 ≥ subgroup size → coalesced
local_2d_bad  = (8, 16)    # x-dim = 8 < subgroup size → SUBOPTIMAL
n_groups_x_g = IMAGE_W // local_2d_good[0]
n_groups_y_g = IMAGE_H // local_2d_good[1]
n_groups_x_b = IMAGE_W // local_2d_bad[0]
n_groups_y_b = IMAGE_H // local_2d_bad[1]

print(f"  Image: {IMAGE_W}×{IMAGE_H} pixels")
print()

for label, local_2d, n_gx, n_gy in [
    ("GOOD (local[0]=16 ≥ subgroup_size)", local_2d_good, n_groups_x_g, n_groups_y_g),
    ("BAD  (local[0]=8  < subgroup_size)", local_2d_bad,  n_groups_x_b, n_groups_y_b),
]:
    items = work_item_ids(local_2d, (IMAGE_W, IMAGE_H), (0, 0))
    print(f"  Config: local=({local_2d[0]},{local_2d[1]}) — {label}")
    print(f"  Work-groups: {n_gx}×{n_gy} = {n_gx*n_gy}  |  "
          f"Work-items/group: {local_2d[0]*local_2d[1]}  |  "
          f"Subgroups/group: {local_2d[0]*local_2d[1]//SUBGROUP_SIZE or 1}")
    print()

    # Show subgroup 0 and its global memory access pattern
    sg0_items = [it for it in items if it["subgrp"] == 0][:8]
    print(f"  Subgroup 0 global memory access pattern (first 8 work-items):")
    print(f"  {'local':>12} | {'global':>14} | {'flat buf addr':>14} | "
          f"{'stride from prev':>18}")
    prev_addr = None
    for it in sg0_items:
        lx, ly = it["local"]
        gx, gy = it["global"]
        flat   = gy * IMAGE_W + gx
        stride = (flat - prev_addr) if prev_addr is not None else 0
        flag   = "✅" if (stride == 1 or prev_addr is None) else f"⚠️  stride={stride}"
        print(f"  ({lx:2d},{ly:2d}) local  | ({gx:3d},{gy:3d}) global | "
              f"  addr={flat:6d}    |   {flag}")
        prev_addr = flat
    print()
    consecutive_x = all(
        (items[i+1]["global"][0] - items[i]["global"][0] == 1)
        for i in range(min(len(items)-1, local_2d[0]-1))
        if items[i]["subgrp"] == items[i+1]["subgrp"]
    )
    verdict = ("COALESCED: x-dim iterates fastest in subgroup" if consecutive_x
               else "NOT COALESCED: x-stride > 1 fragments transactions")
    print(f"  Access verdict: {verdict}")
    print()


# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Compute unit occupancy from the four resource limits
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Occupancy: four resource limits on a CU")
print("━" * 65)
print()
print("  OpenCL occupancy mirrors CUDA occupancy — three hardware limits")
print("  plus the explicit work-group size cap determine concurrency.")
print()

@dataclass
class DeviceCapacity:
    max_threads_per_cu:    int
    max_groups_per_cu:     int
    local_mem_bytes:       int
    registers_per_cu:      int
    subgroup_size:         int

@dataclass
class KernelRequirements:
    name:               str
    threads_per_group:  int
    local_mem_bytes:    int
    regs_per_thread:    int

def compute_opencl_occupancy(dev: DeviceCapacity, kern: KernelRequirements) -> dict:
    """Compute theoretical occupancy for one kernel on one CU."""
    spg = kern.threads_per_group   # subgroups per group: ceil(threads / subgroup_size)
    warp_size = dev.subgroup_size

    # Limit 1: threads (work-items)
    groups_threads = dev.max_threads_per_cu // kern.threads_per_group

    # Limit 2: hardware group cap
    groups_hw_cap  = dev.max_groups_per_cu

    # Limit 3: local memory
    if kern.local_mem_bytes > 0:
        groups_lmem = dev.local_mem_bytes // kern.local_mem_bytes
    else:
        groups_lmem = dev.max_groups_per_cu

    # Limit 4: register file
    regs_per_group = kern.regs_per_thread * kern.threads_per_group
    # Register file allocated in 256-reg granules per warp
    regs_granule   = 256
    regs_per_warp  = (math.ceil(kern.regs_per_thread * warp_size / regs_granule)
                      * regs_granule)
    warps_per_group = math.ceil(kern.threads_per_group / warp_size)
    regs_per_group_actual = regs_per_warp * warps_per_group
    if regs_per_group_actual > 0:
        groups_regs = dev.registers_per_cu // regs_per_group_actual
    else:
        groups_regs = dev.max_groups_per_cu

    active_groups = min(groups_threads, groups_hw_cap, groups_lmem, groups_regs)
    active_groups = max(0, active_groups)

    active_threads = active_groups * kern.threads_per_group
    active_subgroups = active_groups * math.ceil(kern.threads_per_group / warp_size)
    max_subgroups = dev.max_threads_per_cu // warp_size
    occupancy = active_subgroups / max_subgroups if max_subgroups > 0 else 0

    limits = {
        "threads":    groups_threads,
        "hw_cap":     groups_hw_cap,
        "local_mem":  groups_lmem,
        "registers":  groups_regs,
    }
    binding = min(limits, key=limits.get)

    return {
        "active_groups":   active_groups,
        "active_subgroups": active_subgroups,
        "occupancy_pct":   occupancy * 100,
        "binding":         binding,
        "limits":          limits,
    }

# Simulate an AMD RDNA CU (similar to NVIDIA SM)
CU = DeviceCapacity(
    max_threads_per_cu=2048,
    max_groups_per_cu=32,
    local_mem_bytes=65536,         # 64 KB
    registers_per_cu=65536,
    subgroup_size=32,
)

KERNELS = [
    KernelRequirements("Vector add (lightweight)",       256,  0,        8),
    KernelRequirements("Tiled matmul (moderate regs)",   256,  8*1024,  32),
    KernelRequirements("Attention (high regs)",          128, 32*1024,  64),
    KernelRequirements("Large LDS (high shared mem)",    512, 48*1024,  16),
    KernelRequirements("Register-heavy kernel",          256,  4*1024, 128),
    KernelRequirements("Small group (low parallelism)",   32,  1*1024,  20),
]

print(f"  CU spec: {CU.max_threads_per_cu} max threads, {CU.local_mem_bytes//1024} KB LDS, "
      f"{CU.registers_per_cu} registers, subgroup={CU.subgroup_size}")
print()
print(f"  {'Kernel':<38} | {'Grps/CU':>8} | {'Sgrps/CU':>9} | "
      f"{'Occupancy':>10} | {'Binding':>12}")
print(f"  {'─'*82}")

for kern in KERNELS:
    r = compute_opencl_occupancy(CU, kern)
    occ_bar = "█" * int(r["occupancy_pct"] / 5) + "░" * (20 - int(r["occupancy_pct"] / 5))
    print(f"  {kern.name:<38} | {r['active_groups']:8d} | "
          f"{r['active_subgroups']:9d} | "
          f"{r['occupancy_pct']:9.1f}% | {r['binding']:>12}")

print()
print("  Rules remain identical to CUDA occupancy:")
print("    >75% → good latency hiding    50–75% → acceptable    <50% → investigate")
print()


# ─────────────────────────────────────────────────────────────────────────
# SECTION 4: AMD wave32 vs wave64 occupancy comparison
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 4 — AMD wave32 vs wave64: occupancy and throughput trade-off")
print("━" * 65)
print()
print("  AMD RDNA/CDNA can execute in wave32 (32 work-items) or wave64 (64).")
print("  wave32: fewer items per wave → more waves per CU → better occupancy,")
print("          but each wave uses same register allocation → 2× wave overhead.")
print("  wave64: double work per wave → higher VALU utilization per wave.")
print()

CU_W32 = DeviceCapacity(max_threads_per_cu=2048, max_groups_per_cu=32,
                          local_mem_bytes=65536, registers_per_cu=65536, subgroup_size=32)
CU_W64 = DeviceCapacity(max_threads_per_cu=2048, max_groups_per_cu=32,
                          local_mem_bytes=65536, registers_per_cu=65536, subgroup_size=64)

test_kern = KernelRequirements("MatMul kernel", 256, 8*1024, 40)

r32 = compute_opencl_occupancy(CU_W32, test_kern)
r64 = compute_opencl_occupancy(CU_W64, test_kern)

print(f"  Kernel: {test_kern.name}  ({test_kern.threads_per_group} threads, "
      f"{test_kern.local_mem_bytes//1024} KB LDS, {test_kern.regs_per_thread} regs/thread)")
print()
print(f"  {'Metric':<35} | {'wave32':>10} | {'wave64':>10} | {'Notes'}")
print(f"  {'─'*72}")

metrics = [
    ("Active groups per CU",       r32['active_groups'],    r64['active_groups'],   ""),
    ("Active subgroups per CU",     r32['active_subgroups'], r64['active_subgroups'],""),
    ("Occupancy (%)",               f"{r32['occupancy_pct']:.1f}", f"{r64['occupancy_pct']:.1f}", ""),
    ("Binding resource",            r32['binding'],          r64['binding'],         ""),
    ("VALU util (wider = better)",  "1.0×",                  "~1.2×",                "AMD claims wave64 better for ALU-heavy"),
    ("Latency hiding",              "Better",                "Slightly worse",       "More waves hide stalls"),
]
for label, v32, v64, note in metrics:
    print(f"  {label:<35} | {str(v32):>10} | {str(v64):>10} | {note}")
print()
print("  wave32 default in RDNA (gaming workloads). wave64 preferred for compute.")
print("  HIP/ROCm: hip_wave_size = 64 on CDNA (MI200, MI300). 32 on RDNA.")
print("  OpenCL:   cl_amd_device_attribute_query extension exposes wave size.")
''',
    },

    # ── 3 ─────────────────────────────────────────────────────────────────────
    "3 · Memory Model — Address Spaces, Coalescing, and Local Memory Tiling": {
        "description": (
            "Simulate all four OpenCL address spaces and their performance characteristics. "
            "Model global memory coalescing across different access patterns and devices. "
            "Simulate local memory bank conflicts (with device-specific bank widths). "
            "Demonstrate the local-memory tiling algorithm for matrix operations with "
            "quantitative bandwidth and latency analysis. Compare constant memory "
            "vs global memory for read-only broadcast data."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math
from collections import defaultdict

print("=" * 65)
print("  OPENCL MEMORY MODEL — ADDRESS SPACES AND ACCESS PATTERNS")
print("=" * 65)
print()

np.random.seed(3)

# ─────────────────────────────────────────────────────────────────────────
# Device memory specifications
# ─────────────────────────────────────────────────────────────────────────

DEVICES = {
    "NVIDIA A100 (via OpenCL)": {
        "global_bw_TBs": 2.0,
        "local_bw_TBs": 20.0,
        "global_lat_cycles": 700,
        "local_lat_cycles":    5,
        "cache_line_bytes":  128,
        "subgroup_size":      32,
        "n_local_banks":      32,
        "local_bank_bytes":    4,    # 32-bit banks
        "ridge_flop_per_byte": 9.75, # 19.5 TFLOPS / 2 TB/s
    },
    "AMD RX 7900 XTX (ROCm/OpenCL)": {
        "global_bw_TBs": 0.96,
        "local_bw_TBs": 20.0,
        "global_lat_cycles": 500,
        "local_lat_cycles":    4,
        "cache_line_bytes":  128,
        "subgroup_size":      32,
        "n_local_banks":      32,
        "local_bank_bytes":    4,
        "ridge_flop_per_byte": 64.0,
    },
    "Intel Arc A770 (IGC/OpenCL)": {
        "global_bw_TBs": 0.56,
        "local_bw_TBs":  16.0,
        "global_lat_cycles": 400,
        "local_lat_cycles":    5,
        "cache_line_bytes":   64,
        "subgroup_size":      16,
        "n_local_banks":      16,
        "local_bank_bytes":    4,
        "ridge_flop_per_byte": 35.1,
    },
}

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Address space performance table
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — OpenCL address spaces (A100 baseline)")
print("━" * 65)
print()

dev = DEVICES["NVIDIA A100 (via OpenCL)"]

spaces = [
    ("__private (registers)",   1,          50.0,  "per work-item, fastest"),
    ("__local   (LDS/SMEM)",    5,          20.0,  "per work-group, needs barrier()"),
    ("__constant (const cache)",10,         10.0,  "read-only broadcast, cached"),
    ("__global  (VRAM/HBM)",    700,         2.0,  "all work-items, latency hidden by occupancy"),
    ("__global  (system RAM)",  100000,      0.064,"host-accessible, very slow from device"),
]

print(f"  {'Address Space':<30} | {'Latency (cycles)':>18} | {'Bandwidth (TB/s)':>17} | {'Notes'}")
print(f"  {'─'*85}")
for name, lat_cy, bw, note in spaces:
    lat_ns = lat_cy / 1.41  # nanoseconds at 1.41 GHz
    print(f"  {name:<30} | {lat_cy:>12,d} ({lat_ns:4.0f}ns) | {bw:17.2f} | {note}")

print()
print("  The bandwidth gap: __private is 25× faster than __global.")
print("  The latency gap:   __private is 700× lower latency than __global.")
print("  Takeaway: every avoidable global load is a 700-cycle penalty.")
print()


# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Global memory coalescing model
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Global memory coalescing across three devices")
print("━" * 65)
print()

def coalescing_transactions(stride: int, dtype_bytes: int,
                             subgroup_size: int, cache_line: int) -> dict:
    """
    Count 128-byte cache line transactions for a strided access by one subgroup.
    stride: element stride between adjacent work-items.
    """
    touched_lines = set()
    for lane in range(subgroup_size):
        byte_addr = lane * stride * dtype_bytes
        line      = byte_addr // cache_line
        touched_lines.add(line)

    n_tx          = len(touched_lines)
    useful_bytes  = subgroup_size * dtype_bytes
    transferred   = n_tx * cache_line
    efficiency    = useful_bytes / transferred

    return {
        "transactions":  n_tx,
        "efficiency":    efficiency,
        "useful_bytes":  useful_bytes,
        "transferred":   transferred,
    }

print(f"  Subgroup reads one float per lane: buf[get_global_id(0) * stride]")
print()

# Header with device names
header = f"  {'Stride':>8} | {'Pattern':>26}"
for dname in DEVICES:
    header += f" | {dname[:20]:>20} eff%"
print(header)
print("  " + "─" * (36 + 25 * len(DEVICES)))

STRIDES    = [1, 2, 4, 8, 32, 64]
STRIDE_LBL = {1: "Coalesced",   2: "Stride-2",
              4: "Stride-4",   8: "Stride-8",
              32: "1 CL/thread", 64: "Scattered"}

for s in STRIDES:
    row = f"  {s:8d} | {STRIDE_LBL[s]:>26}"
    for dname, dspec in DEVICES.items():
        r = coalescing_transactions(s, 4, dspec["subgroup_size"], dspec["cache_line_bytes"])
        eff_str = f"{r['efficiency']*100:5.1f}%"
        icon    = "✅" if r["efficiency"] > 0.9 else ("⚠️ " if r["efficiency"] > 0.3 else "🔴")
        row += f" | {icon}{eff_str:>18}"
    print(row)

print()
print("  Intel Arc has 64-byte cache lines (vs 128 for NVIDIA/AMD).")
print("  Stride-2 is worse on A100: 2 cache lines touched vs Intel's 1.")
print("  Conclusion: write stride=1 kernels — portable AND fast on all devices.")
print()


# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Local memory bank conflicts (OpenCL __local)
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Local memory (LDS) bank conflict analysis")
print("━" * 65)
print()
print("  __local memory is divided into N banks (32 on NVIDIA/AMD, 16 on Intel).")
print("  bank(addr) = (addr / bank_bytes) % n_banks")
print("  Conflict = 2+ work-items in the same subgroup hit the same bank.")
print()

def lds_bank_conflicts(access_pattern: list, n_banks: int) -> dict:
    """Compute bank conflict degree for one subgroup's local memory access."""
    bank_hits = defaultdict(list)
    for lane, word_idx in enumerate(access_pattern):
        bank = word_idx % n_banks
        bank_hits[bank].append(lane)

    all_same = len(set(access_pattern)) == 1     # broadcast
    max_way  = max(len(v) for v in bank_hits.values())
    if all_same:
        max_way = 1   # broadcast: single transaction

    n_conflict_banks = sum(1 for v in bank_hits.values() if len(v) > 1)

    return {
        "max_way":          max_way,
        "conflict_banks":   n_conflict_banks,
        "is_broadcast":     all_same,
        "is_conflict_free": max_way == 1,
        "transactions":     max_way,
        "efficiency":       1.0 / max_way,
    }

# Access patterns (word indices)
patterns = [
    (list(range(32)),                  "Sequential: local[lid]"),
    ([i * 2 for i in range(32)],       "Stride-2:   local[lid*2]"),
    ([i * 4 for i in range(32)],       "Stride-4:   local[lid*4]"),
    ([i * 32 for i in range(32)],      "Stride-32:  local[lid*32] (all bank 0)"),
    ([0] * 32,                         "Broadcast:  local[0] — all same addr"),
    ([i * 33 for i in range(32)],      "Stride-33:  local[lid*33] (banks rotate)"),
    ([i + (i // 32) for i in range(32)], "Padded:   local[lid + lid/32]"),
]

for dev_label, n_banks in [("NVIDIA/AMD (32 banks)", 32),
                             ("Intel Xe  (16 banks)", 16)]:
    print(f"  Device: {dev_label}")
    print(f"  {'Pattern':<44} | {'Conflict':>10} | {'Efficiency':>11} | {'Note'}")
    print(f"  {'─'*80}")
    for pat, label in patterns:
        r = lds_bank_conflicts(pat, n_banks)
        conflict_str = f"{r['max_way']}-way" if r['max_way'] > 1 else "None"
        note = ""
        if r['is_broadcast']:     note = "broadcast (fast path)"
        elif r['is_conflict_free']: note = "ideal"
        elif r['max_way'] == n_banks: note = "worst case"
        eff_str = f"{r['efficiency']*100:.1f}%"
        print(f"  {label:<44} | {conflict_str:>10} | {eff_str:>11} | {note}")
    print()

print("  FIX for stride-N conflict: add +1 padding per row")
print("    Bad:  __local float tile[32][32];    // stride=32 → 32-way conflict on col access")
print("    Good: __local float tile[32][33];    // stride=33 → rotated banks, no conflict")
print()


# ─────────────────────────────────────────────────────────────────────────
# SECTION 4: Constant memory performance model
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 4 — __constant vs __global for read-only data")
print("━" * 65)
print()
print("  __constant: dedicated constant cache (broadcast to all work-items).")
print("  Best for: Gaussian filter coefficients, transformation matrices,")
print("            lookup tables accessed identically by all work-items.")
print()

FILTER_SIZES = [3, 5, 7, 9, 11, 15, 25]
MAX_CONST_BYTES = 64 * 1024   # 64 KB typical constant buffer limit

print(f"  Convolution kernel: accessing filter coefficients")
print(f"  Max __constant buffer: {MAX_CONST_BYTES//1024} KB ({MAX_CONST_BYTES//4} floats)")
print()
print(f"  {'Filter':>12} | {'Coeff (floats)':>15} | {'Fits in const?':>15} | "
      f"{'Recommended':>15} | {'Speedup est.':>14}")
print(f"  {'─'*80}")

for fsz in FILTER_SIZES:
    n_coeffs    = fsz * fsz
    n_bytes     = n_coeffs * 4
    fits        = n_bytes <= MAX_CONST_BYTES
    rec         = "__constant" if fits else "__global (read-only)"
    # Speedup: constant cache hits (broadcast) vs L2 global reads
    # If all subgroups broadcast same coeff → ~5 cycle latency vs ~200 for L2
    speedup     = "~3-10×" if fits else "1× (global cache)"
    fits_str    = "✅ YES" if fits else "❌ NO"
    print(f"  {fsz:2d}×{fsz:<2d} filter | {n_coeffs:15d} | {fits_str:>15} | "
          f"{rec:>15} | {speedup:>14}")

print()
print("  Rule: anything <= 64 KB AND read by ALL work-items → use __constant.")
print("  __constant with broadcast access: same latency as __local (~5 cycles).")
print("  __global with repeated access: hits L2 (~200 cycles) or L1 (~20 cycles).")
''',
    },

    # ── 4 ─────────────────────────────────────────────────────────────────────
    "4 · Kernel Compilation Pipeline — Runtime JIT and Optimization Flags": {
        "description": (
            "Simulate the complete OpenCL kernel compilation pipeline: "
            "source string → clBuildProgram → binary → clCreateKernel → clSetKernelArg. "
            "Parse and analyse build logs for common errors. Model the impact of "
            "compiler flags on performance (fast-relaxed-math, mad-enable, unroll). "
            "Demonstrate the binary caching strategy to eliminate recompilation overhead. "
            "Contrast with CUDA's ahead-of-time NVCC pipeline."
        ),
        "language": "python",
        "code": '''
import hashlib
import time
import random
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

print("=" * 65)
print("  OPENCL KERNEL COMPILATION PIPELINE SIMULATOR")
print("=" * 65)
print()

random.seed(42)

# ─────────────────────────────────────────────────────────────────────────
# Kernel source examples (OpenCL C)
# ─────────────────────────────────────────────────────────────────────────

KERNEL_SOURCES = {
    "saxpy": """
__kernel void saxpy(__global float* x,
                    __global float* y,
                    float alpha, int N) {
    int gid = get_global_id(0);
    if (gid < N)
        y[gid] = alpha * x[gid] + y[gid];
}
""",
    "matmul_tiled": """
#define TILE 16
__kernel void matmul(__global float* A,
                     __global float* B,
                     __global float* C,
                     int M, int K, int N) {
    __local float As[TILE][TILE];
    __local float Bs[TILE][TILE];
    int ty = get_local_id(1), tx = get_local_id(0);
    int row = get_group_id(1)*TILE + ty;
    int col = get_group_id(0)*TILE + tx;
    float sum = 0.0f;
    for (int t = 0; t < K/TILE; t++) {
        As[ty][tx] = A[row*K + t*TILE + tx];
        Bs[ty][tx] = B[(t*TILE+ty)*N + col];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int k = 0; k < TILE; k++)
            sum += As[ty][k] * Bs[k][tx];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    C[row*N + col] = sum;
}
""",
    "reduction": """
__kernel void reduce(__global float* in,
                     __global float* out,
                     __local  float* scratch,
                     int N) {
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int lsz = get_local_size(0);
    scratch[lid] = (gid < N) ? in[gid] : 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = lsz/2; s > 0; s >>= 1) {
        if (lid < s)
            scratch[lid] += scratch[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0)
        out[get_group_id(0)] = scratch[0];
}
""",
    "broken_kernel": """
__kernel void broken(__global float* A, int N) {
    int gid = get_global_id(0)    // MISSING SEMICOLON
    float val = A[gid];
    __local float cache[256];
    cache[gid] = val;     // BUG: gid may exceed local size
    // Missing barrier before local read
    A[gid] = cache[gid + 1];  // out-of-bounds potential
}
""",
}

# ─────────────────────────────────────────────────────────────────────────
# Compiler flag sets
# ─────────────────────────────────────────────────────────────────────────

COMPILER_FLAG_SETS = {
    "Debug (no opts)":        "-cl-opt-disable",
    "Default":                "",
    "Fast math":              "-cl-fast-relaxed-math",
    "Fast math + MAD":        "-cl-fast-relaxed-math -cl-mad-enable",
    "Fast math + no-signed-zeros": "-cl-fast-relaxed-math -cl-no-signed-zeros",
    "Full optimization":      "-cl-fast-relaxed-math -cl-mad-enable -cl-no-signed-zeros -cl-unsafe-math-optimizations",
    "With defines":           "-cl-fast-relaxed-math -DTILE=16 -DDEBUG=0",
}

FLAG_DESCRIPTIONS = {
    "-cl-opt-disable":                "Disable all optimizations. Useful for debugging.",
    "-cl-fast-relaxed-math":          "Enables flush-to-zero, relaxed precision. Allows reciprocal approximation. ~10-20% speedup.",
    "-cl-mad-enable":                 "Allow a*b+c to use native MAD instruction (fused multiply-add). Free performance.",
    "-cl-no-signed-zeros":            "Allows compiler to ignore -0.0 semantics. Enables more FP reordering.",
    "-cl-unsafe-math-optimizations":  "Superset of all relaxed math flags. Maximum vectorization. Use if accuracy allows.",
    "-DTILE=16":                      "Preprocessor define. Equivalent to NVCC -DTILE=16.",
    "-DDEBUG=0":                      "Preprocessor define for conditional compilation.",
}

# ─────────────────────────────────────────────────────────────────────────
# Simulated compilation pipeline
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class CompilationResult:
    kernel_name:     str
    device:          str
    flags:           str
    success:         bool
    compile_time_ms: float
    binary_size_KB:  float
    estimated_speedup: float
    errors:          List[str] = field(default_factory=list)
    warnings:        List[str] = field(default_factory=list)
    binary_hash:     str = ""

def simulate_build(kernel_name: str, source: str,
                   device_name: str, flags: str) -> CompilationResult:
    """
    Simulate clBuildProgram() for a given kernel and flag set.
    Models: compilation time, binary size, speedup from flags.
    Injects realistic errors for the broken kernel.
    """
    errors, warnings = [], []

    # Detect obvious kernel issues
    if "broken" in kernel_name:
        errors.append("error: expected ';' after expression")
        errors.append("  int gid = get_global_id(0)    // MISSING SEMICOLON")
        errors.append("                                ^")
        errors.append("note: local buffer index may exceed CL_DEVICE_MAX_WORK_GROUP_SIZE")
        errors.append("warning: barrier() missing before local memory read in loop")

    if "matmul" in kernel_name and "-DTILE" not in flags and "TILE" in source:
        warnings.append("warning: TILE is defined in source; consider passing -DTILE=N for tuning")

    success = len(errors) == 0

    # Compilation time model: JIT is expensive
    base_compile_ms = {
        "saxpy":         45.0,
        "matmul_tiled": 210.0,
        "reduction":    130.0,
        "broken_kernel": 15.0,   # fails early
    }.get(kernel_name, 100.0)

    # Debug mode is faster to compile but slower to run
    if "-cl-opt-disable" in flags:
        compile_ms = base_compile_ms * 0.6
    elif "unsafe" in flags:
        compile_ms = base_compile_ms * 1.4   # more analysis
    else:
        compile_ms = base_compile_ms

    compile_ms += random.uniform(-10, 10)

    # Binary size
    base_binary_KB = {"saxpy": 8.0, "matmul_tiled": 28.0, "reduction": 18.0}.get(kernel_name, 12.0)
    if "-cl-opt-disable" in flags:
        binary_KB = base_binary_KB * 1.5   # unoptimized code is larger
    elif "unsafe" in flags:
        binary_KB = base_binary_KB * 0.7   # more aggressive dead-code elimination
    else:
        binary_KB = base_binary_KB

    # Estimated speedup from flags (relative to default)
    speedup = 1.0
    if "-cl-opt-disable" in flags:
        speedup = 0.45
    elif "-cl-fast-relaxed-math" in flags and "-cl-mad-enable" in flags and "-cl-unsafe" in flags:
        speedup = 1.35
    elif "-cl-fast-relaxed-math" in flags and "-cl-mad-enable" in flags:
        speedup = 1.22
    elif "-cl-fast-relaxed-math" in flags:
        speedup = 1.12

    # Binary hash for caching
    h = hashlib.md5((source + flags + device_name).encode()).hexdigest()[:12]

    return CompilationResult(
        kernel_name=kernel_name,
        device=device_name,
        flags=flags,
        success=success,
        compile_time_ms=compile_ms if success else compile_ms * 0.2,
        binary_size_KB=binary_KB if success else 0.0,
        estimated_speedup=speedup if success else 0.0,
        errors=errors,
        warnings=warnings,
        binary_hash=h,
    )


# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Compilation of valid kernels with different flag sets
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Kernel compilation with varying flag sets (MATMUL)")
print("━" * 65)
print()

DEVICE = "NVIDIA A100 (via OpenCL)"

print(f"  Device: {DEVICE}")
print(f"  Kernel: matmul_tiled  (local memory tiled 16×16 GEMM)")
print()
print(f"  {'Flag Set':<40} | {'Time (ms)':>10} | {'Binary (KB)':>11} | {'Speedup':>9} | {'Status'}")
print(f"  {'─'*82}")

for flag_label, flags in COMPILER_FLAG_SETS.items():
    r = simulate_build("matmul_tiled", KERNEL_SOURCES["matmul_tiled"], DEVICE, flags)
    status = "✅ OK" if r.success else "❌ ERROR"
    print(f"  {flag_label:<40} | {r.compile_time_ms:10.1f} | "
          f"{r.binary_size_KB:11.1f} | {r.estimated_speedup:9.2f}× | {status}")

print()
print("  Compilation flags explained:")
for flag, desc in FLAG_DESCRIPTIONS.items():
    print(f"    {flag:<42}: {desc}")
print()
print("  KEY INSIGHT: Compilation is expensive (45–250 ms).")
print("  On a production server, recompiling at every startup wastes time.")
print("  Solution: binary caching (see Section 3).")
print()


# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Error handling — broken kernel
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Build failure: error log (CL_PROGRAM_BUILD_LOG)")
print("━" * 65)
print()

r_broken = simulate_build("broken_kernel", KERNEL_SOURCES["broken_kernel"], DEVICE, "")
print(f"  clBuildProgram() returned: {'CL_SUCCESS' if r_broken.success else 'CL_BUILD_PROGRAM_FAILURE'}")
print(f"  Compilation time: {r_broken.compile_time_ms:.1f} ms")
print()
print("  CL_PROGRAM_BUILD_LOG output:")
print("  " + "─" * 60)
for line in r_broken.errors:
    print(f"  {line}")
for line in r_broken.warnings:
    print(f"  {line}")
print("  " + "─" * 60)
print()
print("  Recovery pattern in host code:")
print("    if (clBuildProgram(...) != CL_SUCCESS) {")
print("        size_t log_size;")
print("        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);")
print("        char* log = malloc(log_size);")
print("        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);")
print("        fprintf(stderr, \\"Kernel build error:\\n%s\\n\\", log);")
print("        exit(1);")
print("    }")
print()


# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Binary caching simulation
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Binary caching to eliminate recompilation overhead")
print("━" * 65)
print()
print("  OpenCL programs can be compiled once and saved as device binaries.")
print("  On next startup: skip compilation → load binary directly.")
print()

class BinaryCache:
    """
    Simulates clGetProgramInfo(CL_PROGRAM_BINARIES) / clCreateProgramWithBinary.
    Key = hash(source + flags + device). Value = (binary, metadata).
    """
    def __init__(self):
        self.cache: Dict[str, CompilationResult] = {}
        self.hits   = 0
        self.misses = 0

    def _key(self, source: str, flags: str, device: str) -> str:
        return hashlib.md5((source + flags + device).encode()).hexdigest()

    def get(self, source: str, flags: str, device: str) -> Optional[CompilationResult]:
        k = self._key(source, flags, device)
        if k in self.cache:
            self.hits += 1
            return self.cache[k]
        self.misses += 1
        return None

    def put(self, source: str, flags: str, device: str, result: CompilationResult):
        k = self.cache[self._key(source, flags, device)] = result

    @property
    def hit_rate(self):
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

cache = BinaryCache()
FLAGS = "-cl-fast-relaxed-math -cl-mad-enable"

COMPILATION_RUNS = [
    ("saxpy",         "cold startup — first ever launch"),
    ("matmul_tiled",  "cold startup — first ever launch"),
    ("reduction",     "cold startup — first ever launch"),
    ("saxpy",         "warm startup — cached"),
    ("matmul_tiled",  "warm startup — cached"),
    ("reduction",     "warm startup — cached"),
    ("saxpy",         "second warm run"),
]

total_cold_ms = 0.0
total_warm_ms = 0.0
BINARY_LOAD_MS = 2.5   # loading a pre-compiled binary is fast

print(f"  {'Run':>4} | {'Kernel':<16} | {'Scenario':<35} | {'Time (ms)':>10} | {'Action'}")
print(f"  {'─'*80}")

for i, (kname, scenario) in enumerate(COMPILATION_RUNS):
    src    = KERNEL_SOURCES[kname]
    cached = cache.get(src, FLAGS, DEVICE)

    if cached:
        elapsed = BINARY_LOAD_MS
        action  = "clCreateProgramWithBinary (cache hit)"
        total_warm_ms += elapsed
    else:
        result  = simulate_build(kname, src, DEVICE, FLAGS)
        cache.put(src, FLAGS, DEVICE, result)
        elapsed = result.compile_time_ms
        action  = "clBuildProgram (compiled from source)"
        total_cold_ms += elapsed

    print(f"  {i+1:4d} | {kname:<16} | {scenario:<35} | {elapsed:10.1f} | {action}")

print()
print(f"  Cache hit rate: {cache.hit_rate*100:.1f}%")
print(f"  Total cold compile time:  {total_cold_ms:.1f} ms  (first run overhead)")
print(f"  Total warm load time:     {total_warm_ms:.1f} ms  (subsequent runs)")
print(f"  Speedup from caching:     {total_cold_ms/total_warm_ms:.1f}×")
print()
print("  Implementation: use sha256(source + flags + device_name) as cache key.")
print("  Store: kernel_cache_dir/{hash}.bin  and  {hash}.meta.json")
print("  On startup: compute hash, check cache dir, call WithBinary if found.")
print()


# ─────────────────────────────────────────────────────────────────────────
# SECTION 4: OpenCL vs CUDA compilation pipeline comparison
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 4 — OpenCL vs CUDA compilation pipeline")
print("━" * 65)
print()

comparison = [
    ("Compilation trigger",  "Runtime (clBuildProgram)",     "Build time (nvcc)"),
    ("Input",                "OpenCL C source string",       "CUDA C++ .cu file"),
    ("Intermediate",         "SPIR-V (OpenCL 2.1+) or IR",  "PTX assembly"),
    ("Final format",         "Vendor ISA binary",            "CUBIN or PTX+JIT"),
    ("First-run overhead",   "50–500 ms JIT compilation",    "None (pre-compiled)"),
    ("Portability",          "Any OpenCL device",            "NVIDIA only"),
    ("Binary caching",       "Manual (app responsibility)",  "Driver cache (~/.nv/ComputeCache)"),
    ("Debug symbols",        "-cl-opt-disable flag",         "nvcc -G flag"),
    ("Perf analysis",        "clGetEventProfilingInfo",      "nvprof / nsight-compute"),
    ("Preprocessor defines", "-DNAME=VALUE in build options","nvcc -DNAME=VALUE"),
    ("Inline PTX/ISA",       "Not supported",                "__asm__ PTX inline"),
    ("Max performance gap",  "5-15% below CUDA on NVIDIA",   "Baseline (100%)"),
]

print(f"  {'Aspect':<30} | {'OpenCL':>28} | {'CUDA':>28}")
print(f"  {'─'*90}")
for aspect, ocl, cuda in comparison:
    print(f"  {aspect:<30} | {ocl:>28} | {cuda:>28}")
print()
print("  OpenCL 2.1+ added SPIR-V as an intermediate format:")
print("    GLSL / HLSL / SYCL → SPIR-V → clCreateProgramWithIL()")
print("  This lets you compile from multiple source languages to one portable IR.")
''',
    },

    # ── 5 ─────────────────────────────────────────────────────────────────────
    "5 · Event Profiling, Roofline, and Cross-Device Performance Model": {
        "description": (
            "Simulate OpenCL event-based profiling to extract kernel execution time, "
            "queue latency, and PCIe transfer time. Build a device-portable roofline "
            "model that classifies kernels as memory-bound or compute-bound on each "
            "target device. Model the full pipeline: H2D transfer → kernel → D2H transfer, "
            "and demonstrate double-buffering to overlap transfers with compute. "
            "Compare achieved vs theoretical performance across NVIDIA, AMD, and Intel."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

print("=" * 65)
print("  EVENT PROFILING, ROOFLINE, AND CROSS-DEVICE PERFORMANCE")
print("=" * 65)
print()

np.random.seed(9)

@dataclass
class DeviceRoofline:
    name:              str
    peak_fp32_tflops:  float
    peak_fp16_tflops:  float
    global_bw_TBs:     float
    pcie_bw_GBs:       float
    kernel_launch_us:  float

    @property
    def ridge_fp32(self):
        return self.peak_fp32_tflops / self.global_bw_TBs

    @property
    def ridge_fp16(self):
        return self.peak_fp16_tflops / self.global_bw_TBs

    def roofline_perf(self, flops, bytes_accessed, use_fp16=False):
        peak  = self.peak_fp16_tflops if use_fp16 else self.peak_fp32_tflops
        if peak == 0:
            peak = self.peak_fp32_tflops  # CPU fallback: no FP16 unit
        ai    = flops / bytes_accessed if bytes_accessed > 0 else 0.0
        ridge = self.ridge_fp16 if (use_fp16 and self.peak_fp16_tflops > 0) else self.ridge_fp32
        achievable = min(ai * self.global_bw_TBs, peak) if peak > 0 else 0.0
        exec_s = flops / (achievable * 1e12) if achievable > 0 else 0.0
        return {
            "ai": ai,
            "achievable_gflops": achievable * 1e3,
            "peak_gflops":       peak * 1e3,
            "peak_util_pct":     (achievable / peak * 100) if peak > 0 else 0.0,
            "exec_time_us":      exec_s * 1e6,
            "bottleneck":        "Compute" if ai > ridge else "Memory BW",
            "is_compute_bound":  ai > ridge,
            "ridge":             ridge,
        }

DEVICES = {
    "A100 (NVIDIA, OpenCL)":  DeviceRoofline("A100",   19.5,  77.0, 2.0,  64.0, 5.0),
    "RX 7900 XTX (AMD)":      DeviceRoofline("RX7900", 61.4, 122.8, 0.96, 32.0, 8.0),
    "Arc A770 (Intel)":       DeviceRoofline("ArcA770",19.66, 39.3, 0.56, 32.0, 12.0),
    "Xeon CPU (AMD OpenCL)":  DeviceRoofline("EPYC",    4.9,   0.0, 0.2,   0.0, 50.0),
}

@dataclass
class KernelSpec:
    name:          str
    flops:         float
    bytes_read:    float
    bytes_written: float
    use_fp16:      bool = False

    @property
    def bytes_total(self):
        return self.bytes_read + self.bytes_written

N = 4096

KERNELS = [
    KernelSpec("Vector add (C=A+B)",
               flops=N*N, bytes_read=2*N*N*4, bytes_written=N*N*4),
    KernelSpec("SAXPY (y=ax+y)",
               flops=2*N*N, bytes_read=N*N*4, bytes_written=N*N*4),
    KernelSpec("Layer Norm",
               flops=10*N*N, bytes_read=2*N*N*4, bytes_written=N*N*4),
    KernelSpec("Matrix multiply FP32 (N=4096)",
               flops=2*N**3, bytes_read=2*N*N*4, bytes_written=N*N*4),
    KernelSpec("Matrix multiply FP16 (N=4096)",
               flops=2*N**3, bytes_read=2*N*N*2, bytes_written=N*N*4, use_fp16=True),
    KernelSpec("Conv2D 3x3 256ch 256x256",
               flops=2*3*3*256*256*256*256,
               bytes_read=(9*256*256+256*256*256*2)*4, bytes_written=256*256*256*4),
    KernelSpec("Softmax (batch=256, seq=1024)",
               flops=5*256*1024, bytes_read=256*1024*4, bytes_written=256*1024*4),
]

# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Roofline classification across all devices")
print("━" * 65)
print()

for dname, dev in DEVICES.items():
    print(f"  Device: {dname}")
    print(f"    FP32 peak: {dev.peak_fp32_tflops:.1f} TFLOPS  |  "
          f"BW: {dev.global_bw_TBs*1e3:.0f} GB/s  |  "
          f"Ridge (FP32): {dev.ridge_fp32:.1f} FLOP/byte")
    print()
    print(f"    {'Kernel':<42} | {'AI':>7} | {'Bottleneck':>12} | "
          f"{'GFLOPS':>9} | {'Peak util':>10}")
    print(f"    {'─'*85}")
    for kern in KERNELS:
        r = dev.roofline_perf(kern.flops, kern.bytes_total, kern.use_fp16)
        icon = "🖥️ " if r["is_compute_bound"] else "💾 "
        print(f"    {kern.name:<42} | {r['ai']:7.1f} | "
              f"{icon}{r['bottleneck']:>9} | "
              f"{r['achievable_gflops']:9.1f} | {r['peak_util_pct']:9.1f}%")
    print()

# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — OpenCL event profiling simulation")
print("━" * 65)
print()
print("  clGetEventProfilingInfo extracts nanosecond timestamps per command:")
print("  CL_PROFILING_COMMAND_QUEUED / SUBMIT / START / END")
print()

@dataclass
class ProfilingEvent:
    name:     str
    t_queued: int
    t_submit: int
    t_start:  int
    t_end:    int

    @property
    def queue_latency_us(self): return (self.t_submit - self.t_queued) / 1000
    @property
    def submit_latency_us(self): return (self.t_start - self.t_submit) / 1000
    @property
    def execution_us(self): return (self.t_end - self.t_start) / 1000
    @property
    def total_us(self): return (self.t_end - self.t_queued) / 1000


def simulate_events(dev, kern, n_bytes_h2d, n_bytes_d2h):
    t = 0
    events = []

    h2d_ns = int(n_bytes_h2d / (dev.pcie_bw_GBs * 1e9) * 1e9) if dev.pcie_bw_GBs > 0 else 0
    q = t; t += 100; s = t; t += 200; st = t; t += h2d_ns
    events.append(ProfilingEvent("H2D write", q, s, st, t)); t += 500

    kern_r  = dev.roofline_perf(kern.flops, kern.bytes_total)
    kern_ns = int(kern_r["exec_time_us"] * 1000)
    lo_ns   = int(dev.kernel_launch_us * 1000)
    q = t; t += 50; s = t; t += lo_ns; st = t; t += kern_ns
    events.append(ProfilingEvent("Kernel exec", q, s, st, t)); t += 200

    d2h_ns = int(n_bytes_d2h / (dev.pcie_bw_GBs * 1e9) * 1e9) if dev.pcie_bw_GBs > 0 else 0
    q = t; t += 100; s = t; t += 200; st = t; t += d2h_ns
    events.append(ProfilingEvent("D2H read", q, s, st, t))
    return events

N_MB   = 64
n_bytes = N_MB * 1024 * 1024
test_kern = KERNELS[3]

print(f"  Kernel: {test_kern.name}  |  Data: {N_MB} MB")
print()

for dname in ["A100 (NVIDIA, OpenCL)", "RX 7900 XTX (AMD)", "Arc A770 (Intel)"]:
    dev = DEVICES[dname]
    events = simulate_events(dev, test_kern, n_bytes, n_bytes // 4)
    print(f"  Device: {dname}")
    print(f"  {'Event':<16} | {'Q.lat':>7} | {'S.lat':>7} | {'Exec':>10} | {'Throughput'}")
    print(f"  {'─'*60}")
    for ev in events:
        if ev.name == "H2D write":
            tp = f"{n_bytes/(ev.execution_us*1e-6)/1e9:.1f} GB/s H2D"
        elif ev.name == "D2H read":
            tp = f"{(n_bytes//4)/(ev.execution_us*1e-6)/1e9:.1f} GB/s D2H"
        else:
            gflops = test_kern.flops / (ev.execution_us * 1e-6) / 1e9
            tp = f"{gflops:.1f} GFLOPS"
        print(f"  {ev.name:<16} | {ev.queue_latency_us:6.1f}us | "
              f"{ev.submit_latency_us:6.1f}us | {ev.execution_us:9.1f}us | {tp}")
    total_us = sum(ev.total_us for ev in events)
    print(f"  Total pipeline: {total_us:.1f} us ({total_us/1000:.2f} ms)")
    print()

# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Double buffering: overlap PCIe transfer with compute")
print("━" * 65)
print()
print("  Single:  [H2D] → [KERN] → [D2H] → [H2D] → [KERN] → [D2H]")
print("  Double:  [H2D_0] → [KERN_0 || H2D_1] → [D2H_0 || KERN_1 || H2D_2] → ...")
print()

def pipeline_time(dev, kern, chunk_bytes, n_chunks, double):
    h2d_us = chunk_bytes / (dev.pcie_bw_GBs * 1e9) * 1e6 if dev.pcie_bw_GBs > 0 else 0
    kr     = dev.roofline_perf(kern.flops, kern.bytes_total)
    kern_us = kr["exec_time_us"]
    d2h_us  = (chunk_bytes // 4) / (dev.pcie_bw_GBs * 1e9) * 1e6 if dev.pcie_bw_GBs > 0 else 0

    if not double:
        total = n_chunks * (h2d_us + kern_us + d2h_us)
    else:
        stage = max(h2d_us, kern_us, d2h_us)
        total = h2d_us + kern_us + (n_chunks - 1) * stage + d2h_us
    return total, (n_chunks * chunk_bytes) / (total * 1e-6) / 1e9

N_CHUNKS = 16
CHUNK_BYTES = 32 * 1024 * 1024

print(f"  {N_CHUNKS} chunks x 32 MB = {N_CHUNKS*32} MB — {test_kern.name}")
print()
print(f"  {'Device':<30} | {'Single (ms)':>12} | {'Double (ms)':>12} | "
      f"{'Speedup':>9} | {'Double BW'}")
print(f"  {'─'*72}")

for dname in ["A100 (NVIDIA, OpenCL)", "RX 7900 XTX (AMD)", "Arc A770 (Intel)"]:
    dev = DEVICES[dname]
    t_s, bw_s = pipeline_time(dev, test_kern, CHUNK_BYTES, N_CHUNKS, False)
    t_d, bw_d = pipeline_time(dev, test_kern, CHUNK_BYTES, N_CHUNKS, True)
    print(f"  {dname:<30} | {t_s/1000:12.2f} | {t_d/1000:12.2f} | "
          f"{t_s/t_d:9.2f}x | {bw_d:.1f} GB/s")

print()
print("  Double buffering requires out-of-order queue or two command queues:")
print("    Queue A: kernels  (CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE)")
print("    Queue B: transfers")
print("    Sync via cl_event: enqueue kernel with wait_list=[h2d_event]")

# ─────────────────────────────────────────────────────────────────────────
print()
print("━" * 65)
print("  SECTION 4 — Performance portability score (Pennycook metric)")
print("━" * 65)
print()
print("  PP(a,H) = |H| / SUM_d(1/efficiency(a,d))")
print("  PP=1.0: peak on all devices.  PP<0.5: highly device-specific.")
print()

def pennycook_pp(effs):
    if not effs or any(e <= 0 for e in effs):
        return 0.0
    return len(effs) / sum(1.0/e for e in effs)

print(f"  {'Kernel':<40} | {'A100':>8} | {'RX7900':>8} | {'Arc':>8} | "
      f"{'PP':>8} | {'Verdict'}")
print(f"  {'─'*82}")

for kern in KERNELS:
    effs, eff_s = [], []
    for dname in ["A100 (NVIDIA, OpenCL)", "RX 7900 XTX (AMD)", "Arc A770 (Intel)"]:
        r = DEVICES[dname].roofline_perf(kern.flops, kern.bytes_total, kern.use_fp16)
        effs.append(r["peak_util_pct"] / 100)
        eff_s.append(f"{r['peak_util_pct']:5.1f}%")
    pp = pennycook_pp(effs)
    verdict = "Excellent" if pp > 0.7 else ("Good" if pp > 0.4 else "Poor")
    print(f"  {kern.name:<40} | {eff_s[0]:>8} | {eff_s[1]:>8} | {eff_s[2]:>8} | "
          f"{pp:8.3f} | {verdict}")

print()
print("  Memory-bound kernels (AI << ridge) → universally efficient → portable.")
print("  Compute-bound on some devices, memory-bound on others → low PP score.")
print("  Tuning: auto-select work-group size + tile size per device at startup.")
''',
    },

}

# ─────────────────────────────────────────────────────────────────────────────
# Dedent operation code strings
# ─────────────────────────────────────────────────────────────────────────────
for _op in OPERATIONS.values():
    _op["code"] = textwrap.dedent(_op["code"]).strip()


def _strip_ansi(text):
    return re.compile(r'\x1b\[[0-9;]*m').sub('', text)


def get_content():
    return {
        "display_name": DISPLAY_NAME,
        "icon":         ICON,
        "subtitle":     SUBTITLE,
        "theory":       THEORY,
        "visual_html":  "",
        "visual_height": 400,
        "complexity":   None,
        "operations":   OPERATIONS,
    }