"""
cuFFT — R2C / C2C Transforms, FFT-Based Convolution & Batched FFT
==================================================================

The Fast Fourier Transform is the most algorithmically important primitive
in signal processing, and on GPU hardware it is the canonical example of a
problem where memory access patterns, twiddle factor arithmetic, and
parallelism structure interact in non-obvious ways.

The naïve DFT has O(N²) complexity. The FFT achieves O(N log N) through
a divide-and-conquer factorisation — the Cooley-Tukey algorithm — that
decomposes an N-point DFT into two N/2-point DFTs. On GPU, cuFFT implements
this factorisation using a hierarchy of kernel launches tuned for:

    - Memory coalescing: butterflies are arranged so threads access
      contiguous memory within each stage.
    - SMEM tiling: each radix kernel loads a tile into SMEM, computes
      all butterflies in the tile, and writes back.
    - Twiddle factor LUTs: precomputed in texture/constant memory.
    - Batch parallelism: multiple transforms run independently in parallel,
      enabling the GPU to stay saturated even for moderate-length signals.

Three concrete topics define this module:

    R2C / C2C TRANSFORMS:
        Real-to-Complex (R2C) exploits Hermitian symmetry — for real input
        x[n], the DFT satisfies X[N-k] = X[k]*. Only N/2+1 unique complex
        outputs exist. R2C computes just these, halving memory and work.
        C2C is the general complex-to-complex forward or inverse transform.

    FFT-BASED CONVOLUTION:
        Direct convolution of two signals of length N costs O(N²).
        FFT convolution: FFT both signals (O(N log N)), pointwise multiply
        in frequency domain (O(N)), inverse FFT (O(N log N)).
        Total: O(N log N) vs O(N²). Crossover point: N ≈ 32–64 on CPU;
        on GPU, convolution can be faster via FFT even for N ≈ 16.

    BATCHED FFT:
        cuFFT supports transforming B independent signals of length N in
        a single cufftExecC2C call. Each signal in the batch maps to a
        separate set of CUDA thread blocks, enabling the GPU to parallelise
        across the batch dimension while keeping the per-signal FFT
        algorithms optimal.

"""

import textwrap
import re
import math
import numpy as np
from collections import defaultdict

TOPIC_NAME   = "cuFFT — R2C / C2C Transforms, FFT-Based Convolution & Batched FFT"
DISPLAY_NAME = "15 · cuFFT"
ICON         = "〰️"
SUBTITLE     = "R2C · C2C · Cooley-Tukey · FFT Convolution · Batched Plans · HBM Roofline"

# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

##### PART 1 — THE DISCRETE FOURIER TRANSFORM: MATHEMATICS AND COMPLEXITY

### The DFT Definition

    The N-point DFT of a signal x[0..N-1]:

        X[k] = Σ_{n=0}^{N-1}  x[n] × W_N^{nk}
                                where W_N = exp(-2πi/N)  (twiddle factor)

    The inverse DFT (IDFT):

        x[n] = (1/N) × Σ_{k=0}^{N-1}  X[k] × W_N^{-nk}

    COMPLEXITY OF NAÏVE DFT:
        For each of N output bins: N multiplications and N additions.
        Total: N² complex multiplications = O(N²).
        For N = 1024: 1,048,576 complex multiplications.
        For N = 1M:   10^12 multiplications — infeasible.

### The Cooley-Tukey FFT Algorithm (Radix-2 DIT)

    KEY INSIGHT: when N = 2^m, the N-point DFT splits into two N/2-point DFTs:

        X[k]   = DFT_even(k) + W_N^k × DFT_odd(k)      for k = 0..N/2-1
        X[k+N/2] = DFT_even(k) - W_N^k × DFT_odd(k)

    where DFT_even uses x[0], x[2], x[4], ... (even-indexed samples)
    and   DFT_odd  uses x[1], x[3], x[5], ... (odd-indexed samples).

    This is the BUTTERFLY operation — one W_N^k multiply and two adds.
    Applying recursively: log₂(N) stages, each with N/2 butterflies.

    TOTAL OPERATIONS:
        Complex multiplications: (N/2) × log₂(N)
        Complex additions:       N × log₂(N)
        For N = 1024: 5120 complex multiplications vs 1,048,576 for naïve.
        Speedup: 204× for N=1024; grows as N/log₂(N).

### Radix-2, Radix-4, Radix-8: The Radix Hierarchy

    RADIX-2: split N into 2 sub-problems of size N/2 (as above).
        Best when N = 2^m.
        log₂(N) stages. Each stage: N/2 butterflies.

    RADIX-4: split N into 4 sub-problems of size N/4.
        Each radix-4 butterfly: 3 complex multiplications vs 4 for radix-2.
        ~25% fewer multiplications than radix-2.
        Best when N = 4^m.

    RADIX-8: even more efficient for very large N = 8^m.
        Used by cuFFT for large sizes to maximise arithmetic per load.

    MIXED-RADIX: cuFFT factorises N into its prime factors.
        N = 2^a × 3^b × 5^c × 7^d → mixed radix-2/3/5/7 plan.
        Works for any N (not just powers of 2), with best performance when
        N has only small prime factors (2, 3, 5, 7).
        N with large prime factors: slower — approaches O(N²) in worst case.
        RULE: always pad signals to the next highly composite number.

### cuFFT Plan System

    A cuFFT PLAN encodes all decisions for a specific transform configuration:
        transform type (R2C, C2C, C2R)
        data type (FP32 or FP64)
        signal length N (or Nx × Ny for 2D)
        batch count B
        stride and distance parameters

    PLAN CREATION: cufftPlan1d, cufftPlanMany, cufftCreate + cufftMakePlanMany.
        Internal: cuFFT selects the radix decomposition, allocates workspace
        (for intermediate storage during multi-stage transforms), and compiles
        or looks up the optimal kernel variant for this GPU architecture.
        Cost: milliseconds to seconds (one-time, amortised over many transforms).

    PLAN REUSE: a plan created for (N=1024, batch=32) can be reused for
        all subsequent calls with the same configuration.
        For inference servers: create plans once at startup.

    WORK AREA:
        cuFFT may require internal scratch memory (the "work area").
        Query with: cufftGetSize(plan, &workSize).
        Managed by cuFFT automatically, or provided manually for fine-grained
        memory control.


##### PART 2 — R2C AND C2R TRANSFORMS: EXPLOITING HERMITIAN SYMMETRY

### The Hermitian Symmetry Property

    For a REAL-VALUED input signal x[n] ∈ ℝ:
        X[N-k] = conj(X[k])    for k = 1, ..., N/2-1

    This is the HERMITIAN SYMMETRY of the DFT for real inputs.
    PROOF:
        X[N-k] = Σ_n x[n] × W_N^{n(N-k)}
               = Σ_n x[n] × W_N^{nN} × W_N^{-nk}
               = Σ_n x[n] × 1 × W_N^{-nk}    (since W_N^{nN} = exp(-2πi×n) = 1)
               = conj(Σ_n x[n] × W_N^{nk})   (since x[n] is real)
               = conj(X[k])

    CONSEQUENCE: only N/2+1 unique complex values in X (the "non-redundant" half).
        X[0]:       real-valued (DC component)
        X[1..N/2-1]: complex pairs
        X[N/2]:     real-valued (Nyquist frequency, only when N is even)

### R2C Transform

    INPUT:  N real float32 values → OUTPUT: N/2+1 complex float32 values.

    MEMORY:
        Input:  N × 4 bytes
        Output: (N/2 + 1) × 8 bytes
        Ratio:  output / input = (N/2 + 1) / N × 2 ≈ 1.0 (same total bytes, approximately)

    WORK:
        Compute only N/2+1 outputs instead of N.
        Internal trick: treat the real N-point signal as a complex N/2-point signal,
        compute the complex FFT, then extract the N/2+1 unique real-input DFT outputs.
        Work ≈ half of C2C for the same N.

    CUFT API:
        cufftExecR2C(plan, (cufftReal*)d_in, (cufftComplex*)d_out)
        Plan type: CUFFT_R2C
        Plan creation: cufftPlan1d(&plan, N, CUFFT_R2C, 1)

### C2R Transform (Inverse R2C)

    INPUT:  N/2+1 complex float32 values → OUTPUT: N real float32 values.
    This is the IDFT for a signal known to have real-valued output.

    NORMALISATION:
        cuFFT (like FFTW) does NOT normalise the inverse transform.
        After C2R: you must divide by N: output = cufft_output / N.
        This is a design choice — avoids the division in cases where it
        cancels with a subsequent operation.

    PIPELINE:
        R2C(x) → [frequency domain processing] → C2R(X) / N → x_processed

    CUFT API:
        cufftExecC2R(plan, (cufftComplex*)d_freq, (cufftReal*)d_out)
        Plan type: CUFFT_C2R

### C2C Transform

    INPUT:  N complex float32 values → OUTPUT: N complex float32 values.

    DIRECTIONS:
        CUFFT_FORWARD  (-1): X[k] = Σ x[n] × exp(-2πi × nk/N)
        CUFFT_INVERSE  (+1): X[k] = Σ x[n] × exp(+2πi × nk/N)

    Note: CUFFT_INVERSE does NOT divide by N — caller must normalise.

    MEMORY: input and output are both N complex values.
    The output CAN be in-place: same pointer for input and output.
    In-place C2C is fully supported by cuFFT.

    CUFT API:
        cufftExecC2C(plan, d_in, d_out, CUFFT_FORWARD)
        Plan type: CUFFT_C2C

### 2D and 3D Transforms

    cuFFT supports multi-dimensional transforms with separable decomposition:
        2D DFT = 1D DFT along rows × 1D DFT along columns.

    CUFFT 2D API:
        cufftPlan2d(&plan, Ny, Nx, CUFFT_C2C);
        // Nx = fastest-varying dimension (columns), Ny = rows
        cufftExecC2C(plan, d_in, d_out, CUFFT_FORWARD);

    For 2D R2C: output has size Ny × (Nx/2 + 1) complex values.

    3D transforms: cufftPlan3d.
    Multi-GPU 2D transforms: cuFFTMp (multi-process cuFFT via MPI).


##### PART 3 — FFT-BASED CONVOLUTION: THE CONVOLUTION THEOREM

### The Convolution Theorem

    For signals x and h of length N:
        DFT(x * h) = DFT(x) × DFT(h)    [pointwise complex multiply]

    where * denotes CIRCULAR convolution and × is element-wise multiplication.

    CIRCULAR vs LINEAR CONVOLUTION:
        Circular:  (x *_N h)[n] = Σ_{m=0}^{N-1} x[m] × h[(n-m) mod N]
        Linear:    (x * h)[n]   = Σ_m x[m] × h[n-m]  (full, length N_x + N_h - 1)

    To compute LINEAR convolution via FFT:
        1. Choose M = next_pow2(N_x + N_h - 1)   (zero-pad both signals)
        2. Compute X = FFT(x, M) and H = FFT(h, M)
        3. Y = X × H  (pointwise)
        4. y = IFFT(Y) / M
        5. Result is y[0..N_x+N_h-2] (the linear convolution)

### Complexity Analysis: FFT vs Direct Convolution

    DIRECT CONVOLUTION:
        For signal length N_x and kernel length N_h:
        Operations: N_x × N_h real multiplications.
        Typical kernel sizes in CNNs: N_h = 3, 5, 7 (spatial conv).
        For 1D audio: N_h = 256–4096 (FIR filters).

    FFT CONVOLUTION:
        Padded size M = N_x + N_h - 1 (round up to power of 2).
        Operations: 3 × M × log₂(M) (two FFTs + one IFFT) + M (multiply).
        Asymptotic: O(M log M).

    CROSSOVER POINT (where FFT becomes faster):
        Direct: N_x × N_h
        FFT:    3M log₂(M) + M ≈ 3M log₂(M)
        Break-even: N_h ≈ 3 log₂(M)
        For M = 1024: N_h ≈ 30. For M = 4096: N_h ≈ 36.

    ON GPU:
        GPU parallelism makes short convolutions efficient via CUDA kernels.
        FFT convolution becomes faster at smaller N_h on GPU than on CPU.
        cuDNN uses direct convolution for N_h ≤ 5 and FFT convolution for N_h > 7.
        For 1D signals with long kernels (N_h > 32): FFT is always preferred.

### The OLA and OLS Methods for Very Long Signals

    PROBLEM: FFT convolution requires both x and h to fit in GPU memory.
    For streaming audio (gigabytes) with a 4096-tap FIR filter:
        Cannot FFT the entire signal at once.

    OVERLAP-ADD (OLA):
        Split x into non-overlapping blocks of size L.
        Convolve each block with h (length N_h) → result length L + N_h - 1.
        Sum the overlapping tails with adjacent results.
        FFT size per block: M = next_pow2(L + N_h - 1).
        Choose L to maximise efficiency: typically L = 3× to 8× N_h.

    OVERLAP-SAVE (OLS):
        Split x into overlapping blocks of size M (overlap = N_h - 1 samples).
        FFT each block, multiply by H (pre-computed FFT of filter), IFFT.
        Discard the first N_h - 1 samples of each block (corrupted by circular wrap).
        Concatenate the valid portions.
        OLS is often slightly more efficient than OLA (no tail summation needed).

### FFT Convolution for Neural Network Kernels

    In deep learning, FFT convolution appears in:
        1. LONG 1D AUDIO CONVOLUTIONS: WaveNet, HiFi-GAN (kernel sizes 256–4096).
        2. GLOBAL RECEPTIVE FIELD 2D CONVOLUTIONS: ConvNeXt-like large kernels.
        3. SSM (State Space Models): Mamba, S4 use structured convolutions
           that are implemented as frequency-domain multiplications.
        4. FLOP-EFFICIENT ATTENTION APPROXIMATIONS: linear attention via
           polynomial approximations of softmax, sometimes computed via FFT.

    For S4/Mamba:
        The convolution kernel is computed analytically and converted to
        a frequency-domain filter once. Each forward pass: one R2C on input,
        one pointwise multiply, one C2R. Total: O(N log N) per sequence.


##### PART 4 — BATCHED FFT: PARALLELISM ACROSS INDEPENDENT TRANSFORMS

### Why Batching Matters for GPU Efficiency

    A single 1D FFT of length N=1024 has limited parallelism:
        Each Cooley-Tukey stage: N/2 = 512 independent butterflies.
        512 threads occupy only 4 warps (16 warps per SM typical).
        On A100 (108 SMs): 4 warps occupy 1/432 of the GPU.
        GPU utilisation ≈ 0.23% for a single N=1024 FFT.

    With batch size B:
        Total independent butterflies: B × N/2 per stage.
        For B=32: 16384 butterflies → 128 warps → still only 3 SMs.
        For B=128: 65536 butterflies → 512 warps → 12 SMs.
        For B=512: 262144 butterflies → 2048 warps → 47 SMs (44% utilisation).
        For B=2048: 1048576 butterflies → 8192 warps → fills all 108 SMs.

    CONCLUSION: to saturate the GPU, use B ≥ 2048 / (N/32) transforms.
    For N=1024: B ≥ 64 to approach full utilisation.
    For N=16384: B ≥ 4 sufficient (large N provides its own parallelism).

### cuFFT Batched API: cufftPlanMany

    cufftPlanMany handles the most general batched case:

        cufftPlanMany(
            cufftHandle *plan,
            int rank,           // 1 for 1D, 2 for 2D, 3 for 3D
            int *n,             // [N] for 1D; [Ny, Nx] for 2D
            int *inembed,       // input embedding (NULL = no padding)
            int istride,        // spacing between consecutive elements
            int idist,          // distance between consecutive batches
            int *onembed,       // output embedding
            int ostride,        // output stride
            int odist,          // output distance between batches
            cufftType type,     // CUFFT_R2C, CUFFT_C2C, etc.
            int batch           // number of transforms
        );

    FOR CONTIGUOUS BATCH (most common):
        inembed = NULL, istride = 1, idist = N
        onembed = NULL, ostride = 1, odist = N/2+1 (for R2C) or N (for C2C)
        Signals are packed: [signal_0 | signal_1 | ... | signal_{B-1}]

    FOR STRIDED BATCH (e.g., rows of a 2D tensor):
        istride = B (signals are column-major, stride across batch)
        idist   = 1 (adjacent elements in a signal are B apart)

    FOR PADDED BATCH (zero-padding to power-of-2 size):
        inembed = [N_padded], n = [N_actual]
        inembed > n tells cuFFT the input is embedded in a larger buffer
        with zeros at positions N_actual+1 to N_padded.

### Advanced: cufftXtExec and Multiple GPUs

    For very large batches or extremely long transforms:
        cufftXtMakePlanMany: multi-GPU plan.
        cufftXtExecDescriptorC2C: execute on multiple GPUs simultaneously.
        Each GPU handles a subset of the batch (or a sub-volume for 3D).

    Use case: 3D FFTs in weather simulation, seismic imaging, MD simulation.
    cuFFTMp (available separately): distributes 3D FFT across MPI ranks,
    each owning a slab of the 3D volume.


##### PART 5 — cuFFT MEMORY ACCESS PATTERNS AND PERFORMANCE MODEL

### The Butterfly Memory Access Pattern

    Each stage of the Cooley-Tukey algorithm accesses elements in a
    specific stride pattern. In the naive (bit-reversed) layout:

    STAGE 0 (stride=1):   elements 0,1 — 2,3 — 4,5 — ... (contiguous pairs)
    STAGE 1 (stride=2):   elements 0,2 — 1,3 — 4,6 — 5,7 — ...
    STAGE 2 (stride=4):   elements 0,4 — 1,5 — 2,6 — 3,7 — ...
    ...
    STAGE log₂(N)-1:      elements 0,N/2 — 1,N/2+1 — ...

    MEMORY COALESCING CONCERN:
        Early stages (small stride): accesses are coalesced (adjacent threads
        access adjacent memory). HBM bandwidth utilised efficiently.
        Late stages (large stride): accesses stride across the entire array.
        Stride = N/2 means alternating cache lines → poor L1 utilisation.

    cuFFT SOLUTION — SMEM TILING:
        Divide the N-point FFT into segments of size TILE (e.g., 1024).
        Load each tile into SMEM (512 complex = 16 KB for FP32).
        Perform all butterfly stages within the tile in SMEM (zero HBM traffic).
        Write back to HBM once per tile.
        For stages crossing tile boundaries: one global transpose pass.

### Arithmetic Intensity of cuFFT

    For a 1D C2C FFT of N complex FP32 values:
        HBM reads:  2 × N × 8 bytes (read input, write output)
        FLOPs:      5 × N × log₂(N)  (5 per butterfly: 2 mul + 3 add, complex)

    AI = FLOPs / Bytes = 5 × N × log₂(N) / (2 × N × 8)
       = 5 log₂(N) / 16

    For N=1024:   AI = 5 × 10 / 16 = 3.1 FLOPs/Byte
    For N=16384:  AI = 5 × 14 / 16 = 4.4 FLOPs/Byte
    For N=1M:     AI = 5 × 20 / 16 = 6.3 FLOPs/Byte

    H100 ridge point: 989 TFLOP/s / 3.35 TB/s = 295 FLOPs/Byte.
    cuFFT is ALWAYS memory-bandwidth-limited, regardless of N.

    PRACTICAL IMPLICATION:
        cuFFT performance scales linearly with HBM bandwidth.
        H100 FFT throughput ≈ 3× A100 (bandwidth ratio ≈ 3.35/2.0 = 1.67×,
        but also better L2 cache and SMEM bandwidth contribute).
        No benefit from tensor cores — FFT is not matmul.

### The cuFFT Roofline at Different Sizes and Batch Counts

    Peak achievable cuFFT throughput (signals per second):
        Peak = HBM_BW / bytes_per_signal = HBM_BW / (2 × N × 8)

    For H100 (3.35 TB/s), N=1024:
        Peak = 3.35e12 / (2 × 1024 × 8) = 204 million signals/second
        At batch=2048: 204M / (some_util_factor) ≈ realistic ~100M sig/s.

    For N=16384: Peak = 3.35e12 / (2 × 16384 × 8) ≈ 12.8M signals/second.

    For real-world applications the achieved throughput is 60–80% of peak
    due to:
        - Twiddle factor computation overhead
        - SMEM bank conflicts in certain tile sizes
        - Kernel launch overhead for multi-stage transforms
        - Sub-optimal batch count (GPU not fully saturated)

### cuFFT vs vendor alternatives

    ROCFFT (AMD): equivalent API on ROCm GPUs.
    VkFFT: open-source, runs on Vulkan/CUDA/ROCm, often faster for small N.
    cuFFT:  NVIDIA's official library, best for large N and production use.
    PyTorch torch.fft: uses cuFFT under the hood on CUDA tensors.
        torch.fft.rfft → cuFFT R2C
        torch.fft.fft  → cuFFT C2C
        torch.fft.irfft → cuFFT C2R
        All planning is handled automatically; plans are cached internally.


##### PART 6 — TWIDDLE FACTORS, BIT-REVERSAL, AND THE SMEM BUTTERFLY KERNEL

### Twiddle Factors

    W_N^k = exp(-2πik/N) = cos(2πk/N) - i × sin(2πk/N)

    Computing trigonometric functions at runtime is expensive.
    cuFFT strategy:
        Small N (≤ 4096): store all N twiddle factors in constant/texture memory.
        Large N (> 4096): compute on-the-fly using trigonometric identities or
        precomputed tables for the prime-factor components.

    BIT-REVERSAL PERMUTATION:
        The Cooley-Tukey DIT (decimation-in-time) FFT requires inputs in
        bit-reversed order. For N=8:
            Normal:     0,1,2,3,4,5,6,7
            Bit-reversed: 0,4,2,6,1,5,3,7  (bit-reverse each 3-bit index)
        cuFFT absorbs bit-reversal into the first stage kernel.
        Alternatively: use DIF (decimation-in-frequency) which bit-reverses output.

### The SMEM Butterfly Kernel (Simplified)

    // 1D FFT of TILE=1024 complex points using SMEM
    // One CUDA block handles the full 1024-point FFT

    __global__ void fft_1024_kernel(cufftComplex* data) {
        __shared__ cufftComplex smem[1024];
        int tid = threadIdx.x;

        // Load with bit-reversal
        int rev_tid = bit_reverse(tid, 10);  // 10 = log2(1024)
        smem[tid] = data[rev_tid + blockIdx.x * 1024];
        __syncthreads();

        // log2(1024) = 10 butterfly stages
        for (int stage = 0; stage < 10; stage++) {
            int half_stride = 1 << stage;     // 1, 2, 4, 8, ...
            int stride      = half_stride * 2; // 2, 4, 8, 16, ...

            int butterfly_group = tid / half_stride;
            int butterfly_index = tid % half_stride;

            // Twiddle factor index
            int k = butterfly_index * (1024 / stride);
            cufftComplex W = twiddle_table[k];  // from constant memory

            cufftComplex even = smem[butterfly_group * stride + butterfly_index];
            cufftComplex odd  = smem[butterfly_group * stride + butterfly_index + half_stride];
            cufftComplex t    = complex_mul(W, odd);

            __syncthreads();   // all threads must complete before updating smem

            smem[butterfly_group * stride + butterfly_index]              = even + t;
            smem[butterfly_group * stride + butterfly_index + half_stride] = even - t;
            __syncthreads();
        }

        // Write back
        data[tid + blockIdx.x * 1024] = smem[tid];
    }

    KEY DETAILS:
        __syncthreads() at each stage: all 1024 threads must complete the
        reads before any thread writes the results.
        With 1024 threads: 16 warps per block. Each warp handles 32 butterflies.
        SMEM bank conflicts: at stage where stride = 32, threads 0 and 16 access
        the same SMEM bank. cuFFT avoids this via padding: SMEM[1024+extra_padding].

### The Global Transpose for Large N

    For N > TILE (e.g., N = 65536 with TILE = 1024):
        Phase 1: Compute 64 in-tile FFTs of 1024 points each.
        Global transpose: rearrange data for inter-tile butterfly passes.
        Phase 2: Compute the inter-tile butterflies.
        Phase 3 (if needed): another in-tile pass after a second transpose.

    The GLOBAL TRANSPOSE is a MATRIX TRANSPOSE of a B × TILE array.
    cuFFT uses a coalesced transpose kernel (processes 32×32 tiles into SMEM).
    This is similar to the shared memory matrix transpose from Module 20.
    The transpose itself is a pure bandwidth operation: AI = 1 FLOPs/Byte.


##### PART 7 — PRACTICAL cuFFT USAGE: API PATTERNS AND PYTORCH INTEGRATION

### Minimal cuFFT C2C Program

    #include <cufft.h>
    #include <cuda_runtime.h>

    int main() {
        const int N = 1024;
        const int BATCH = 32;

        // Allocate device memory
        cufftComplex *d_data;
        cudaMalloc(&d_data, sizeof(cufftComplex) * N * BATCH);

        // Create plan
        cufftHandle plan;
        cufftPlan1d(&plan, N, CUFFT_C2C, BATCH);

        // Execute forward FFT (in-place)
        cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);
        cudaDeviceSynchronize();

        // Execute inverse FFT (in-place)
        cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE);
        cudaDeviceSynchronize();
        // Remember: output is NOT normalised — divide by N*BATCH if needed.

        // Cleanup
        cufftDestroy(plan);
        cudaFree(d_data);
        return 0;
    }

### R2C Convolution Pattern

    // Convolve signal x (length N_x) with filter h (length N_h)
    // M = next power-of-2 >= N_x + N_h - 1

    cufftHandle plan_r2c, plan_c2r;
    cufftPlan1d(&plan_r2c, M, CUFFT_R2C, BATCH);
    cufftPlan1d(&plan_c2r, M, CUFFT_C2R, BATCH);

    // 1. Pad and FFT signal x
    cufftExecR2C(plan_r2c, d_x_padded, d_X);   // d_X: M/2+1 complex

    // 2. Pre-compute FFT of filter h (done once, reused for all batches)
    cufftExecR2C(plan_r2c, d_h_padded, d_H);

    // 3. Pointwise multiply in frequency domain
    pointwise_multiply<<<grid, block>>>(d_X, d_H, d_Y, M/2+1);

    // 4. Inverse FFT
    cufftExecC2R(plan_c2r, d_Y, d_y_padded);   // d_y_padded: M real values

    // 5. Normalise (NOT done by cuFFT)
    scale_kernel<<<grid, block>>>(d_y_padded, 1.0f/M, M);

    // 6. Extract valid output: d_y[0..N_x+N_h-2]

### PyTorch FFT Interface

    import torch
    import torch.fft

    # 1D R2C:
    x = torch.randn(1024).cuda()
    X = torch.fft.rfft(x)             # Returns complex64 of length 513 (N/2+1)
    x_back = torch.fft.irfft(X, n=1024)  # Must specify n for reconstruction

    # 1D C2C:
    z = torch.randn(1024, dtype=torch.complex64).cuda()
    Z = torch.fft.fft(z)
    z_back = torch.fft.ifft(Z)        # Normalised (divides by N)

    # Batched R2C (last dimension transformed):
    x_batch = torch.randn(32, 1024).cuda()
    X_batch = torch.fft.rfft(x_batch)         # (32, 513)

    # 2D R2C (last 2 dimensions):
    img = torch.randn(8, 256, 256).cuda()
    IMG = torch.fft.rfft2(img)                 # (8, 256, 129)

    # FFT-based convolution:
    def fft_conv1d(x, h):
        N = x.shape[-1] + h.shape[-1] - 1
        M = 1 << (N - 1).bit_length()
        X = torch.fft.rfft(x, n=M)
        H = torch.fft.rfft(h, n=M)
        return torch.fft.irfft(X * H, n=M)[..., :N]

### Memory Layout for Strided and Non-Contiguous Tensors

    cuFFT (via PyTorch) requires CONTIGUOUS input tensors.
    If input is non-contiguous (e.g., after a transpose), PyTorch calls
    .contiguous() first — an extra copy.

    For performance-critical FFT pipelines:
        Ensure all input tensors are contiguous before calling torch.fft.*.
        Use torch.as_strided sparingly — it can trigger unexpected copies.

    For cuFFT directly (C API with strides):
        Use cufftPlanMany with explicit istride/ostride/idist/odist to
        handle non-contiguous layouts without a copy.

"""

# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ── 1 ─────────────────────────────────────────────────────────────────────
    "1 · Cooley-Tukey Butterfly — DFT, FFT Stages & Complexity": {
        "description": (
            "Implement the Cooley-Tukey radix-2 DIT FFT from scratch. Show the "
            "bit-reversal permutation, the butterfly structure at each stage, and "
            "the twiddle factor pattern. Trace all stages for N=8 with exact "
            "intermediate values. Compare FLOPs against the naïve DFT at every "
            "power of two from N=8 to N=1M. Show the GPU parallelism structure: "
            "how many independent butterflies exist per stage."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math

print("=" * 68)
print("  COOLEY-TUKEY FFT — Butterfly Stages, Bit-Reversal & Complexity")
print("=" * 68)
print()


# ─────────────────────────────────────────────────────────────────────
# Core FFT implementation
# ─────────────────────────────────────────────────────────────────────

def bit_reverse(x, n_bits):
    """Reverse the binary representation of x (n_bits wide)."""
    result = 0
    for _ in range(n_bits):
        result = (result << 1) | (x & 1)
        x >>= 1
    return result

def bit_reverse_permutation(arr):
    """Reorder arr in bit-reversed index order."""
    n      = len(arr)
    n_bits = int(math.log2(n))
    out    = np.zeros(n, dtype=complex)
    for i in range(n):
        out[bit_reverse(i, n_bits)] = arr[i]
    return out

def fft_radix2(x, trace=False):
    """
    Cooley-Tukey radix-2 DIT FFT.
    Returns the DFT of x (length must be power of 2).
    If trace=True, prints intermediate values at each stage.
    """
    N      = len(x)
    assert N > 0 and (N & (N-1)) == 0, "N must be a power of 2"
    n_bits = int(math.log2(N))

    # Bit-reverse permutation
    X = bit_reverse_permutation(np.array(x, dtype=complex))

    if trace:
        print(f"  After bit-reversal: {X.round(3)}")
        print()

    # Butterfly stages
    stage_results = []
    for stage in range(n_bits):
        half_stride = 1 << stage           # 1, 2, 4, ...
        stride      = half_stride * 2      # 2, 4, 8, ...

        # W_N^k twiddle factors for this stage
        W_stage = np.exp(-2j * np.pi * np.arange(half_stride) / stride)

        for group_start in range(0, N, stride):
            for k in range(half_stride):
                even_idx = group_start + k
                odd_idx  = group_start + k + half_stride
                t = W_stage[k] * X[odd_idx]
                X[even_idx], X[odd_idx] = X[even_idx] + t, X[even_idx] - t

        stage_results.append(X.copy())
        if trace:
            vals = [f"({v.real:.2f}{'+' if v.imag>=0 else ''}{v.imag:.2f}j)"
                    for v in X]
            print(f"  Stage {stage+1:2d} (stride={stride:4d}): {' '.join(vals[:min(8,N)])}")

    return X, stage_results


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: N=8 trace — all stages
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — N=8 Full Butterfly Trace (DIT Radix-2)")
print("━" * 68)
print()

x8 = np.array([1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0])
print(f"  Input:     {x8.tolist()}")

# Bit-reversal mapping
n_bits8 = 3
print(f"  Bit-reversal mapping (3-bit indices):")
for i in range(8):
    rev = bit_reverse(i, n_bits8)
    print(f"    x[{i}] → position {rev}  "
          f"({i:03b} → {rev:03b})")
print()

X8, stages = fft_radix2(x8, trace=True)
X8_ref     = np.fft.fft(x8)

print()
print(f"  Final FFT output:")
for k in range(8):
    print(f"    X[{k}] = {X8[k].real:+.4f} {'+' if X8[k].imag>=0 else ''}{X8[k].imag:.4f}j")
print()
print(f"  NumPy reference match: "
      f"{'✅' if np.allclose(X8, X8_ref, atol=1e-10) else '❌'}")
print()

# Twiddle factors for N=8
print("  Twiddle factors W_8^k = exp(-2πik/8):")
for k in range(5):
    w = np.exp(-2j*np.pi*k/8)
    print(f"    W_8^{k} = cos({k}×45°) - i×sin({k}×45°) = "
          f"{w.real:+.4f} {'+' if w.imag>=0 else ''}{w.imag:.4f}j")


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: Hermitian symmetry of real input
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 2 — Hermitian Symmetry: Why R2C Saves Half the Work")
print("━" * 68)
print()

print("  For real input x[n], the DFT satisfies X[N-k] = conj(X[k]).")
print()
print(f"  Verifying for x = {x8.tolist()}:")
print()
print(f"  {'k':>4}  {'X[k]':>28}  {'X[N-k]':>28}  {'conj(X[k])':>28}  Match")
print("  " + "─" * 90)
N8 = 8
for k in range(1, N8//2):
    xk  = X8[k]
    xnk = X8[N8-k]
    cjk = np.conj(xk)
    match = np.isclose(xnk, cjk, atol=1e-10)
    print(f"  {k:>4}  {xk.real:+.3f}{'+' if xk.imag>=0 else ''}{xk.imag:.3f}j  "
          f"{' ':>12}{xnk.real:+.3f}{'+' if xnk.imag>=0 else ''}{xnk.imag:.3f}j  "
          f"{' ':>12}{cjk.real:+.3f}{'+' if cjk.imag>=0 else ''}{cjk.imag:.3f}j  "
          f"{'✅' if match else '❌'}")

print()
print(f"  R2C only computes: X[0..{N8//2}] = {N8//2+1} values instead of {N8}.")
print(f"  Memory reduction: {N8//2+1}/{N8} = {(N8//2+1)/N8*100:.0f}% of complex storage.")
print(f"  Work reduction:   approximately ×2 fewer operations.")


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: FLOPs — FFT vs DFT across sizes
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 3 — FLOPs: FFT vs Naïve DFT Across N")
print("━" * 68)
print()

print(f"  {'N':>10}  {'DFT FLOPs':>14}  {'FFT FLOPs':>14}  "
      f"{'Speedup':>9}  {'GPU butterflies/stage'}")
print("  " + "─" * 62)

for exp in range(3, 21):
    N   = 1 << exp
    dft = 2 * N * N         # N² complex multiply-adds (real: ~2N² operations)
    fft = 5 * N * exp       # 5 per butterfly (complex): 4 adds + 1 mul × N/2 × stages
    speedup = dft / fft
    butterflies = N // 2    # independent per stage
    print(f"  {N:>10,}  {dft:>14,}  {fft:>14,}  {speedup:>9.1f}×  "
          f"{butterflies:>22,}")


# ─────────────────────────────────────────────────────────────────────
# SECTION 4: GPU parallelism per stage
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 4 — GPU Parallelism: Butterflies per Stage vs SM Saturation")
print("━" * 68)
print()

H100_SMs       = 132
WARPS_PER_SM   = 64
THREADS_PER_WARP = 32
H100_MAX_THREADS = H100_SMs * WARPS_PER_SM * THREADS_PER_WARP

print(f"  H100: {H100_SMs} SMs × {WARPS_PER_SM} warps × {THREADS_PER_WARP} threads "
      f"= {H100_MAX_THREADS:,} max threads")
print()
print(f"  Parallelism = N/2 butterflies per stage (all independent).")
print(f"  One butterfly = 2 threads (handles 2 elements).")
print()
print(f"  {'N':>10}  {'log₂N':>7}  {'N/2 (butterflies)':>20}  "
      f"{'Threads needed':>16}  {'SM utilisation':>16}")
print("  " + "─" * 70)

for exp in range(3, 22):
    N          = 1 << exp
    n_bf       = N // 2
    n_threads  = N          # one thread per element in typical impl
    sm_util    = min(n_threads / H100_MAX_THREADS, 1.0) * 100
    print(f"  {N:>10,}  {exp:>7}  {n_bf:>20,}  {n_threads:>16,}  "
          f"{sm_util:>15.2f}%")

print()
print("  For N < 16M: a single FFT under-utilises H100.")
print("  SOLUTION: batch B transforms to multiply parallelism by B.")
''',
    },

    # ── 2 ─────────────────────────────────────────────────────────────────────
    "2 · R2C vs C2C — Hermitian Symmetry, Memory Layout & Transform Types": {
        "description": (
            "Implement and compare R2C, C2C forward, and C2C inverse transforms. "
            "Show the exact output array sizes and memory layouts. Verify the "
            "normalisation convention (cuFFT unnormalised inverse). Demonstrate "
            "Hermitian symmetry by checking all conjugate pairs in R2C output. "
            "Show how to recover real-valued time-domain signals with C2R. "
            "Compute and compare the arithmetic intensity of R2C vs C2C."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math

print("=" * 68)
print("  R2C vs C2C — Hermitian Symmetry, Memory Layout & Transform Types")
print("=" * 68)
print()


# ─────────────────────────────────────────────────────────────────────
# Transform type implementations (reference via numpy.fft)
# ─────────────────────────────────────────────────────────────────────

def r2c_transform(x):
    """Real-to-Complex DFT. Returns only N/2+1 unique outputs."""
    return np.fft.rfft(x)   # numpy.fft.rfft = R2C

def c2c_transform(z, direction='forward'):
    """Complex-to-Complex DFT."""
    if direction == 'forward':
        return np.fft.fft(z)
    else:
        return np.fft.ifft(z) * len(z)   # unnormalised (cuFFT convention)

def c2r_transform(X, N):
    """Complex-to-Real IDFT (unnormalised, cuFFT convention)."""
    return np.fft.irfft(X, n=N) * N   # irfft normalises; undo that


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: Transform types, output sizes, and memory
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — Transform Types: Output Sizes and Memory Footprint")
print("━" * 68)
print()

for N in [8, 16, 64, 1024, 4096, 16384]:
    input_r_bytes  = N * 4               # real float32
    input_c_bytes  = N * 8               # complex float32 (2× float)
    output_r2c_bytes = (N//2 + 1) * 8   # N/2+1 complex
    output_c2c_bytes = N * 8             # N complex
    output_c2r_bytes = N * 4             # N real

    print(f"  N = {N:>6}:")
    print(f"    R2C:  input={input_r_bytes:>6} B real    → output={output_r2c_bytes:>6} B "
          f"({N//2+1} complex)  ratio={output_r2c_bytes/input_r_bytes:.2f}×")
    print(f"    C2C:  input={input_c_bytes:>6} B complex → output={output_c2c_bytes:>6} B "
          f"({N} complex)    ratio={output_c2c_bytes/input_c_bytes:.2f}×")
    print(f"    C2R:  input={(N//2+1)*8:>6} B complex → output={output_c2r_bytes:>6} B "
          f"({N} real)       ratio={output_c2r_bytes/((N//2+1)*8):.2f}×")
print()
print("  R2C output: N/2+1 values (Hermitian symmetry → other half redundant).")
print("  C2C: input and output both N complex values (can be in-place).")


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: Hermitian symmetry verification
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 2 — Hermitian Symmetry: All Conjugate Pairs Verified")
print("━" * 68)
print()

np.random.seed(42)
N_h = 16
x_real = np.random.randn(N_h).astype(np.float32)
X_c2c  = np.fft.fft(x_real)     # Full N-point DFT

print(f"  Real input x[n], N={N_h}. DFT X[k] via C2C (full {N_h} outputs).")
print()
print(f"  {'k':>4}  {'X[k]':>28}  {'X[N-k]':>28}  {'conj(X[k])':>28}  {'Match?'}")
print("  " + "─" * 88)

all_match = True
for k in range(1, N_h//2):
    xk  = X_c2c[k]
    xnk = X_c2c[N_h-k]
    cjk = np.conj(xk)
    ok  = np.isclose(xnk, cjk, atol=1e-5)
    if not ok:
        all_match = False
    def fmt(z):
        return f"{z.real:+8.4f}{'+' if z.imag>=0 else ''}{z.imag:.4f}j"
    print(f"  {k:>4}  {fmt(xk)}  {fmt(xnk)}  {fmt(cjk)}  {'✅' if ok else '❌'}")

print()
print(f"  X[0]   = {X_c2c[0].real:.4f}  (DC, always real)")
print(f"  X[N/2] = {X_c2c[N_h//2].real:.4f}  (Nyquist, always real for even N)")
print(f"  All conjugate pairs verified: {'✅' if all_match else '❌'}")
print()
print(f"  R2C output contains only: X[0..{N_h//2}] = {N_h//2+1} values.")
print(f"  Missing: X[{N_h//2+1}..{N_h-1}] = {N_h//2-1} values (redundant conjugates).")


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: Round-trip verification with cuFFT normalisation convention
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 3 — Round-Trip: cuFFT Normalisation Convention")
print("━" * 68)
print()

print("  cuFFT (and FFTW) do NOT normalise the inverse transform.")
print("  After IFFT: output is N times the original signal.")
print("  Caller must divide by N for correct reconstruction.")
print()

N_rt = 32
x_rt = np.random.randn(N_rt).astype(np.float32)

# Simulate cuFFT R2C → C2R pipeline
X_r2c = np.fft.rfft(x_rt.astype(np.float64))        # R2C
x_c2r_unnorm = np.fft.irfft(X_r2c, n=N_rt) * N_rt   # C2R unnormalised (cuFFT)
x_c2r_norm   = np.fft.irfft(X_r2c, n=N_rt)           # C2R normalised (numpy)

err_unnorm = np.abs(x_c2r_unnorm - x_rt).max()
err_norm   = np.abs(x_c2r_norm   - x_rt).max()

print(f"  Input x (first 8): {x_rt[:8].round(3).tolist()}")
print(f"  R2C then C2R unnormalised (cuFFT output): {x_c2r_unnorm[:8].round(3).tolist()}")
print(f"  R2C then C2R normalised (/N):             {x_c2r_norm[:8].round(3).tolist()}")
print()
print(f"  Max error (unnormalised vs original): {err_unnorm:.4f}  ← wrong!")
print(f"  Max error (normalised /N vs original): {err_norm:.2e}  ← correct ✅")
print()
print(f"  RULE: after cuFFT C2R (or CUFFT_INVERSE C2C):")
print(f"    output_correct = cufft_output / N")


# ─────────────────────────────────────────────────────────────────────
# SECTION 4: Arithmetic intensity of R2C vs C2C
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 4 — Arithmetic Intensity: R2C vs C2C vs Batched")
print("━" * 68)
print()

H100_BW_GBS  = 3350.0
H100_PEAK    = 989.0    # TFLOP/s for FP16 (but FFT uses FP32 typically)
H100_PEAK_FP32 = 67.0   # TFLOP/s for scalar FP32

def fft_flops_per_transform(N, is_real=False):
    """Complex multiply-adds per FFT transform (5 FLOPs per butterfly)."""
    return 5 * N * math.log2(N) * (0.5 if is_real else 1.0)

def fft_bytes(N, is_real=False, batch=1):
    """HBM bytes for FFT (read input + write output)."""
    if is_real:
        return (N * 4 + (N//2+1) * 8) * batch   # float32 in + complex out
    else:
        return 2 * N * 8 * batch                 # complex in + complex out

print(f"  {'Transform':>14}  {'N':>8}  {'Batch':>6}  {'FLOPs':>14}  "
      f"{'HBM bytes':>12}  {'AI':>8}  {'BW-limited time µs'}")
print("  " + "─" * 76)

for is_real, label in [(False, "C2C FP32"), (True, "R2C FP32")]:
    for N in [1024, 4096, 16384, 65536]:
        for batch in [1, 64, 1024]:
            flops  = fft_flops_per_transform(N, is_real) * batch
            hbm    = fft_bytes(N, is_real, batch)
            ai     = flops / hbm
            t_us   = hbm / (H100_BW_GBS * 1e9) * 1e6
            if batch == 1 and N in [1024, 16384]:
                print(f"  {label:>14}  {N:>8,}  {batch:>6}  {flops/1e6:>12.2f}M  "
                      f"{hbm/1e3:>10.1f}KB  {ai:>8.2f}  {t_us:>10.3f}")

print()
print(f"  All FFT AI values << H100 ridge ({H100_PEAK/H100_BW_GBS:.0f} FLOPs/Byte for FP16).")
print(f"  FFT is ALWAYS bandwidth-limited. Throughput ∝ HBM bandwidth.")
print(f"  Implication: H100 (3.35 TB/s) is {3350/2000:.7f}× faster than A100 (2.0 TB/s) for FFT.")
''',
    },

    # ── 3 ─────────────────────────────────────────────────────────────────────
    "3 · FFT Convolution — Theorem, Zero-Padding & Crossover Analysis": {
        "description": (
            "Implement FFT-based linear convolution via the convolution theorem: "
            "zero-pad to power-of-2 size, R2C both signals, pointwise multiply, "
            "C2R, normalise. Verify output against direct convolution. Compute "
            "the FLOPs crossover: at what kernel length N_h does FFT convolution "
            "beat direct convolution? Show the crossover dependence on N_x. "
            "Implement the Overlap-Add method for streaming convolution."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math

print("=" * 68)
print("  FFT CONVOLUTION — Theorem, Zero-Padding & Crossover Analysis")
print("=" * 68)
print()

np.random.seed(7)


# ─────────────────────────────────────────────────────────────────────
# Convolution implementations
# ─────────────────────────────────────────────────────────────────────

def next_pow2(n):
    """Smallest power of 2 >= n."""
    return 1 << (n - 1).bit_length()

def direct_conv(x, h):
    """Direct (naïve) linear convolution. O(N_x × N_h)."""
    N_x, N_h = len(x), len(h)
    N_out     = N_x + N_h - 1
    y         = np.zeros(N_out)
    for n in range(N_out):
        for m in range(N_h):
            if 0 <= n-m < N_x:
                y[n] += x[n-m] * h[m]
    return y

def fft_conv(x, h):
    """
    FFT-based linear convolution via the convolution theorem.
    Steps: zero-pad → R2C both → pointwise multiply → C2R → normalise.
    """
    N_x, N_h = len(x), len(h)
    N_lin     = N_x + N_h - 1       # required linear convolution length
    M         = next_pow2(N_lin)     # FFT size (pad to power of 2)

    # Zero-pad
    x_pad = np.zeros(M); x_pad[:N_x] = x
    h_pad = np.zeros(M); h_pad[:N_h] = h

    # R2C transforms
    X = np.fft.rfft(x_pad)
    H = np.fft.rfft(h_pad)

    # Pointwise multiply in frequency domain
    Y = X * H

    # C2R and normalise (numpy irfft already normalises; no need to /M)
    y = np.fft.irfft(Y, n=M)

    return y[:N_lin]   # trim to linear convolution length

def fft_conv_flops(N_x, N_h):
    """FLOPs for FFT convolution (2 FFTs + 1 IFFT + multiply)."""
    N_lin = N_x + N_h - 1
    M     = next_pow2(N_lin)
    fft_f = 5 * M * math.log2(M)   # per FFT (complex, FP32)
    mul_f = 8 * (M//2 + 1)          # pointwise complex multiply = 6 mul + 2 add each
    return 3 * fft_f + mul_f        # 2 forward + 1 inverse + multiply

def direct_conv_flops(N_x, N_h):
    """FLOPs for direct convolution (multiplications only)."""
    return 2 * N_x * N_h           # N_x × N_h multiply-adds


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: Correctness verification
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — Correctness: FFT Conv vs Direct Conv")
print("━" * 68)
print()

test_cases = [
    (np.array([1.0, 2.0, 3.0, 4.0]), np.array([1.0, 1.0, 1.0])),
    (np.random.randn(64),             np.random.randn(8)),
    (np.random.randn(512),            np.random.randn(32)),
    (np.random.randn(4096),           np.random.randn(256)),
]

print(f"  {'N_x':>6}  {'N_h':>6}  {'M (FFT size)':>14}  "
      f"{'Max error':>12}  {'Correct?'}")
print("  " + "─" * 52)

for x_t, h_t in test_cases:
    N_x, N_h = len(x_t), len(h_t)
    M        = next_pow2(N_x + N_h - 1)

    y_direct = direct_conv(x_t, h_t) if N_x * N_h < 200000 else np.convolve(x_t, h_t)
    y_fft    = fft_conv(x_t, h_t)

    max_err  = np.abs(y_direct - y_fft).max()
    correct  = max_err < 1e-5
    print(f"  {N_x:>6}  {N_h:>6}  {M:>14,}  {max_err:>12.2e}  "
          f"{'✅' if correct else '❌'}")


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: FLOPs crossover analysis
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 2 — FLOPs Crossover: When FFT Conv Beats Direct Conv")
print("━" * 68)
print()

print("  Fixed signal length N_x = 4096. Varying kernel length N_h.")
print()
N_x = 4096

print(f"  {'N_h':>8}  {'Direct FLOPs':>14}  {'FFT FLOPs':>14}  "
      f"{'FFT/Direct':>12}  {'FFT faster?'}")
print("  " + "─" * 56)

crossover_nh = None
for N_h in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]:
    d_flops = direct_conv_flops(N_x, N_h)
    f_flops = fft_conv_flops(N_x, N_h)
    ratio   = f_flops / d_flops
    faster  = ratio < 1.0
    if faster and crossover_nh is None:
        crossover_nh = N_h
    print(f"  {N_h:>8}  {d_flops:>14,}  {f_flops:>14,}  "
          f"{ratio:>11.2f}×  {'✅ FFT' if faster else '  direct'}")

print()
print(f"  Crossover point (N_x={N_x:,}): N_h ≈ {crossover_nh}")
print(f"  For N_h > {crossover_nh}: FFT convolution uses fewer FLOPs.")
print()

# Show crossover as function of N_x
print("  Crossover N_h as function of signal length N_x:")
print()
print(f"  {'N_x':>10}  {'Crossover N_h':>16}  {'Comment'}")
print("  " + "─" * 46)

for N_x_test in [256, 512, 1024, 4096, 16384, 65536]:
    co = None
    for nh in range(1, N_x_test+1):
        if fft_conv_flops(N_x_test, nh) < direct_conv_flops(N_x_test, nh):
            co = nh
            break
    comment = ("short kernels OK" if co and co < 50
               else "FFT always better" if co and co < 5
               else "")
    print(f"  {N_x_test:>10,}  {str(co) + ' tokens':>16}  {comment}")


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: Overlap-Add for streaming convolution
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 3 — Overlap-Add: Streaming Convolution for Long Signals")
print("━" * 68)
print()

print("  Problem: signal x is very long (millions of samples). Cannot FFT all at once.")
print("  OLA solution: process x in blocks of size L, collect overlapping tails.")
print()

def overlap_add(x, h, L):
    """
    Overlap-Add convolution of x with h.
    L = block size (number of input samples per FFT block).
    """
    N_h  = len(h)
    M    = next_pow2(L + N_h - 1)   # FFT size per block
    H    = np.fft.rfft(np.pad(h, (0, M - N_h)))   # pre-compute H once

    N_out = len(x) + N_h - 1
    y     = np.zeros(N_out)

    for block_start in range(0, len(x), L):
        block = x[block_start:block_start+L]
        # Zero-pad block to M
        block_pad = np.zeros(M); block_pad[:len(block)] = block
        # FFT, multiply, IFFT
        Y_block = np.fft.rfft(block_pad) * H
        y_block = np.fft.irfft(Y_block, n=M)
        # Add (overlap) into output
        out_start = block_start
        out_end   = min(out_start + M, N_out)
        y[out_start:out_end] += y_block[:out_end-out_start]

    return y

# Verify OLA
np.random.seed(11)
x_stream = np.random.randn(2048)
h_fir    = np.random.randn(128)    # 128-tap FIR filter

y_ref   = np.convolve(x_stream, h_fir)
y_ola_L256 = overlap_add(x_stream, h_fir, L=256)
y_ola_L512 = overlap_add(x_stream, h_fir, L=512)

err_256 = np.abs(y_ref - y_ola_L256).max()
err_512 = np.abs(y_ref - y_ola_L512).max()

print(f"  Signal: N_x={len(x_stream)}, filter: N_h={len(h_fir)}")
print()
print(f"  {'Method':<28}  {'L (block)':>10}  {'FFT size M':>12}  {'Error':>10}")
print("  " + "─" * 54)
print(f"  {'Direct (np.convolve)':<28}  {'—':>10}  {'—':>12}  {'—':>10}")
print(f"  {'Overlap-Add':<28}  {256:>10}  {next_pow2(256+128-1):>12}  {err_256:>10.2e}")
print(f"  {'Overlap-Add':<28}  {512:>10}  {next_pow2(512+128-1):>12}  {err_512:>10.2e}")
print()

# Efficiency of OLA
print("  OLA efficiency: larger L → fewer FFTs per input sample.")
print()
print(f"  {'L (block)':>12}  {'M (FFT size)':>14}  {'FFTs per L':>12}  "
      f"{'Work per sample (FLOPs)':>24}")
print("  " + "─" * 62)
N_h_ola = 256
for L_ola in [128, 256, 512, 1024, 2048, 4096]:
    M_ola      = next_pow2(L_ola + N_h_ola - 1)
    ffts_per_L = 3          # 2 forward (x & H pre-computed) + 1 inverse
    # Only 1 new forward FFT per block (H pre-computed once)
    flops_per_sample = (5 * M_ola * math.log2(M_ola) * 2 + 6 * (M_ola//2+1)) / L_ola
    print(f"  {L_ola:>12,}  {M_ola:>14,}  {ffts_per_L:>12}  {flops_per_sample:>24.1f}")

print()
print("  Optimal L ≈ 4× to 8× N_h: balances FFT overhead vs block processing.")
print(f"  For N_h={N_h_ola}: optimal L ≈ {4*N_h_ola}–{8*N_h_ola}.")
''',
    },

    # ── 4 ─────────────────────────────────────────────────────────────────────
    "4 · Batched FFT — Parallelism Model, Plan Parameters & Throughput": {
        "description": (
            "Model the throughput scaling of batched 1D FFTs as batch size increases "
            "from 1 to 65536. Show SM utilisation and effective HBM throughput "
            "for different N values. Implement the cufftPlanMany parameter "
            "derivation for contiguous, strided, and padded batches. Compute "
            "the optimal batch size to saturate H100. Compare R2C batched "
            "throughput against C2C batched throughput."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math

print("=" * 68)
print("  BATCHED FFT — Parallelism Model, Plan Parameters & Throughput")
print("=" * 68)
print()

# H100 constants
H100_SMs         = 132
H100_WARPS_SM    = 64
H100_THREADS_SM  = H100_WARPS_SM * 32
H100_MAX_THREADS = H100_SMs * H100_THREADS_SM
H100_BW_GBS      = 3350.0


def fft_bytes_per_signal(N, is_r2c=False):
    """HBM bytes to read input and write output for one signal."""
    if is_r2c:
        return N * 4 + (N//2 + 1) * 8   # real in + complex out
    else:
        return N * 8 * 2                  # complex in + complex out

def fft_threads_per_signal(N):
    """Number of CUDA threads used to compute one N-point FFT (typical assignment)."""
    # cuFFT assigns ~N threads per batch element (one per output bin)
    return N

def sm_utilisation(N, batch):
    """Fraction of H100 SMs that are active for this batched FFT."""
    total_threads = fft_threads_per_signal(N) * batch
    # Threads fill SMs; ceil(total/max) waves
    waves = math.ceil(total_threads / H100_MAX_THREADS)
    util  = total_threads / (waves * H100_MAX_THREADS)
    return util

def peak_throughput_signals_per_sec(N, batch, is_r2c=False):
    """
    Bandwidth-limited throughput for B signals of length N.
    Returns signals per second.
    """
    total_bytes = fft_bytes_per_signal(N, is_r2c) * batch
    time_s      = total_bytes / (H100_BW_GBS * 1e9)
    return batch / time_s


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: Throughput scaling vs batch size
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — Throughput Scaling vs Batch Size (N=1024, C2C FP32)")
print("━" * 68)
print()

N_bench = 1024

print(f"  N={N_bench}, C2C FP32, H100 ({H100_SMs} SMs, {H100_BW_GBS} GB/s)")
print()
print(f"  {'Batch':>8}  {'Threads':>12}  {'SM util':>9}  "
      f"{'HBM bytes':>12}  {'BW-limited t (µs)':>20}  {'Signals/sec'}")
print("  " + "─" * 72)

for batch in [1, 4, 16, 64, 256, 1024, 4096, 16384, 65536]:
    threads    = fft_threads_per_signal(N_bench) * batch
    sm_util    = sm_utilisation(N_bench, batch) * 100
    hbm_bytes  = fft_bytes_per_signal(N_bench) * batch
    t_us       = hbm_bytes / (H100_BW_GBS * 1e9) * 1e6
    sigs_per_s = peak_throughput_signals_per_sec(N_bench, batch)

    print(f"  {batch:>8,}  {threads:>12,}  {sm_util:>8.1f}%  "
          f"{hbm_bytes/1e6:>9.2f}MB  {t_us:>20.3f}  {sigs_per_s/1e6:>8.1f}M/s")

print()
print("  Throughput in signals/sec is CONSTANT regardless of batch size")
print("  (bandwidth-limited: total bytes ∝ batch, time ∝ batch).")
print("  SM utilisation saturates at ~B = 64 for N=1024 (fills all SMs).")
print("  Below saturation: some SMs idle → sub-optimal bandwidth usage.")


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: Minimum batch to saturate H100
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 2 — Minimum Batch to Saturate H100 (>= 90% SM utilisation)")
print("━" * 68)
print()

print(f"  H100: {H100_SMs} SMs × {H100_THREADS_SM:,} max threads/SM = {H100_MAX_THREADS:,} max threads")
print()
print(f"  {'N':>10}  {'Min batch for 90%':>20}  {'Threads at min batch':>22}  {'SM util'}")
print("  " + "─" * 60)

for N_sat in [64, 128, 256, 512, 1024, 4096, 16384, 65536, 262144]:
    # Find minimum batch where SM util >= 90%
    for b_try in range(1, 100000):
        if sm_utilisation(N_sat, b_try) >= 0.90:
            min_batch = b_try
            break
    else:
        min_batch = -1

    util_at_min = sm_utilisation(N_sat, min_batch) * 100
    threads_min = N_sat * min_batch
    print(f"  {N_sat:>10,}  {min_batch:>20,}  {threads_min:>22,}  {util_at_min:>7.1f}%")

print()
print("  Large N is self-saturating (parallelism = N/2 per stage).")
print("  Small N requires large batch to utilise all SMs.")
print("  Rule of thumb: batch ≥ H100_max_threads / N to fully saturate.")


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: cufftPlanMany parameter derivation
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 3 — cufftPlanMany Parameter Derivation for 3 Layouts")
print("━" * 68)
print()

print("  cufftPlanMany(plan, rank, n, inembed, istride, idist,")
print("                        onembed, ostride, odist, type, batch)")
print()

layouts = [
    ("Contiguous (standard)",
     "Signals are packed: [sig0 | sig1 | ... | sigB-1]",
     "inembed=NULL, istride=1, idist=N",
     "onembed=NULL, ostride=1, odist=N/2+1 (R2C) or N (C2C)"),

    ("Column-strided (rows of 2D tensor)",
     "Signals are rows of a [N × B] matrix (row-major → col-major strided)",
     "inembed=NULL, istride=B, idist=1",
     "onembed=NULL, ostride=B, odist=1"),

    ("Padded (zero-pad input to power-of-2)",
     "Input signal length N_in stored in buffer of length M (M > N_in)",
     "inembed=[M], n=[N_in], istride=1, idist=M",
     "onembed=NULL, ostride=1, odist=N_in/2+1 (R2C)"),
]

N_ex, B_ex, N_in_ex, M_ex = 512, 32, 400, 512

for i, (name, desc, input_params, output_params) in enumerate(layouts):
    print(f"  LAYOUT {i+1}: {name}")
    print(f"  Scenario: {desc}")
    print()
    print(f"    Input:  {input_params}")
    print(f"    Output: {output_params}")
    print()
    if i == 0:
        print(f"    For N={N_ex}, B={B_ex}, R2C:")
        print(f"      Input buffer:  {N_ex * B_ex} real floats = {N_ex*B_ex*4} bytes")
        print(f"      Output buffer: {(N_ex//2+1) * B_ex} complex floats = {(N_ex//2+1)*B_ex*8} bytes")
    elif i == 1:
        print(f"    For N={N_ex}, B={B_ex} (2D tensor [{N_ex}×{B_ex}]):")
        print(f"      Signal 0: elements at indices 0, B, 2B, ..., (N-1)B")
        print(f"      Signal k: elements at indices k, k+B, k+2B, ..., k+(N-1)B")
    elif i == 2:
        print(f"    For N_in={N_in_ex}, M={M_ex}, B={B_ex} (zero-padded R2C):")
        print(f"      Input:  {M_ex * B_ex} real floats (includes zero padding {M_ex-N_in_ex} zeros each)")
        print(f"      Output: {(N_in_ex//2+1)*B_ex} complex values")
    print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 4: R2C vs C2C batched throughput comparison
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 4 — R2C vs C2C Throughput: Memory and FLOPs Comparison")
print("━" * 68)
print()

print(f"  Batch=2048, varying N. H100 bandwidth-limited throughput.")
print()
print(f"  {'N':>10}  {'R2C bytes/sig':>16}  {'C2C bytes/sig':>16}  "
      f"{'R2C Gsig/s':>12}  {'C2C Gsig/s':>12}  {'R2C/C2C ratio'}")
print("  " + "─" * 74)

BATCH_COMPARE = 2048
for N_cmp in [128, 256, 512, 1024, 2048, 4096, 8192, 16384]:
    b_r2c = fft_bytes_per_signal(N_cmp, is_r2c=True)
    b_c2c = fft_bytes_per_signal(N_cmp, is_r2c=False)

    tps_r2c = peak_throughput_signals_per_sec(N_cmp, BATCH_COMPARE, is_r2c=True)  / 1e9
    tps_c2c = peak_throughput_signals_per_sec(N_cmp, BATCH_COMPARE, is_r2c=False) / 1e9
    ratio   = tps_r2c / tps_c2c

    print(f"  {N_cmp:>10,}  {b_r2c/1024:>14.2f}KB  {b_c2c/1024:>14.2f}KB  "
          f"{tps_r2c:>12.3f}  {tps_c2c:>12.3f}  {ratio:>7.2f}×")

print()
print("  R2C is faster because input is real (N×4 bytes) vs C2C (N×8 bytes).")
print("  But output of R2C is N/2+1 complex values (N×8 bytes) — comparable.")
print("  Net: R2C saves ≈ (N×4) / (N×8 + N×8) ≈ 20% less HBM traffic → 1.25× faster.")
''',
    },

    # ── 5 ─────────────────────────────────────────────────────────────────────
    "5 · FFT Performance Model — HBM Roofline, N Padding & Plan Selection": {
        "description": (
            "Build the complete cuFFT performance model on H100 and A100: "
            "HBM bandwidth limits, arithmetic intensity across N and batch, "
            "and the optimal plan type selection. Show the power-of-2 padding "
            "benefit by computing FFT cost for original N vs padded N. Measure "
            "the cost of bad factorisation (prime N). Provide an FFT size "
            "selection guide for audio, ML, and scientific computing workloads."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math
from functools import lru_cache

print("=" * 68)
print("  FFT PERFORMANCE MODEL — Roofline, N Padding & Plan Selection")
print("=" * 68)
print()

H100_BW_GBS   = 3350.0
A100_BW_GBS   = 2000.0
A10G_BW_GBS   =  600.0

GPUS = {
    "H100 SXM5": H100_BW_GBS,
    "A100 SXM4": A100_BW_GBS,
    "A10G":      A10G_BW_GBS,
}


@lru_cache(maxsize=None)
def smallest_prime_factor(n):
    if n < 2: return 1
    for p in [2, 3, 5, 7, 11, 13]:
        if n % p == 0:
            return p
    # Check remaining odd divisors
    i = 17
    while i * i <= n:
        if n % i == 0:
            return i
        i += 2
    return n  # n is prime

def factorize(n):
    """Return prime factorization as {prime: exponent}."""
    factors = {}
    while n > 1:
        p = smallest_prime_factor(n)
        factors[p] = factors.get(p, 0) + 1
        n //= p
    return factors

def is_highly_composite(n):
    """True if all prime factors of n are 2, 3, 5, or 7 (5-smooth or 7-smooth)."""
    facts = factorize(n)
    return all(p in {2, 3, 5, 7} for p in facts)

def fft_cost_factor(n):
    """
    Approximate cuFFT cost multiplier based on factorisation.
    1.0 = pure radix-2 (optimal). Larger = more expensive.
    """
    if n == 1: return 1.0
    facts = factorize(n)
    cost  = 1.0
    for p, e in facts.items():
        if p == 2:   cost *= 1.0 * e
        elif p == 3: cost *= 1.3 * e
        elif p == 5: cost *= 1.6 * e
        elif p == 7: cost *= 2.0 * e
        else:        cost *= (p * 1.0) * e  # large primes very expensive
    return cost

def next_good_size(n, max_prime=7):
    """Find the smallest N >= n whose prime factors are all <= max_prime."""
    while not is_highly_composite(n):
        n += 1
    return n

def fft_bandwidth_time_us(N, batch, bw_gbs, is_r2c=False):
    """Bandwidth-limited time in µs for a batched FFT."""
    if is_r2c:
        bytes_per_sig = N * 4 + (N//2+1) * 8
    else:
        bytes_per_sig = N * 8 * 2
    total_bytes = bytes_per_sig * batch
    return total_bytes / (bw_gbs * 1e9) * 1e6


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: Factorisation quality and FFT cost
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — FFT Plan Quality: Factorisation and Cost Factor")
print("━" * 68)
print()

print("  cuFFT decomposes N into prime factors. Small factors (2,3,5,7) are fast.")
print("  Large prime factors → slow mixed-radix or Bluestein algorithm.")
print()
print(f"  {'N':>8}  {'Factorisation':>20}  {'Highly composite?':>20}  "
      f"{'Cost factor':>12}  {'Recommended'}")
print("  " + "─" * 68)

test_ns = [512, 600, 1000, 1024, 1080, 1200, 1331, 2048, 3000, 4096,
           5000, 6000, 7919, 8192, 9000, 16384, 32768]

for n_test in test_ns:
    facts = factorize(n_test)
    fact_str = '×'.join(f'{p}^{e}' if e>1 else str(p)
                         for p,e in sorted(facts.items()))
    hc       = is_highly_composite(n_test)
    cf       = fft_cost_factor(n_test)
    rec      = "✅ good" if hc and cf < 10 else ("⚠ ok" if cf < 20 else "❌ avoid")
    print(f"  {n_test:>8,}  {fact_str:>20}  {str(hc):>20}  "
          f"{cf:>12.1f}  {rec}")


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: Padding benefit — next good size
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 2 — Padding to Good FFT Size: Cost vs Waste")
print("━" * 68)
print()

print("  Zero-padding to the next 7-smooth number often outweighs the extra data.")
print()
print(f"  {'Original N':>12}  {'Padded N':>10}  {'Waste%':>8}  "
      f"{'Cost factor orig':>18}  {'Cost factor pad':>17}  {'Pad faster?'}")
print("  " + "─" * 74)

for n_orig in [513, 600, 800, 900, 1000, 1500, 2000, 3000, 5000, 7919, 10000]:
    n_pad    = next_good_size(n_orig)
    waste    = (n_pad - n_orig) / n_orig * 100
    cf_orig  = fft_cost_factor(n_orig)
    cf_pad   = fft_cost_factor(n_pad)

    # Estimate: padded FFT is faster if (cf_orig * n_orig) > (cf_pad * n_pad)
    work_orig = cf_orig * n_orig * math.log2(n_orig)
    work_pad  = cf_pad  * n_pad  * math.log2(n_pad)
    pad_faster = work_pad < work_orig

    print(f"  {n_orig:>12,}  {n_pad:>10,}  {waste:>7.1f}%  "
          f"{cf_orig:>18.1f}  {cf_pad:>17.1f}  "
          f"{'✅ yes' if pad_faster else '  no '}")

print()
print("  Padding is strongly recommended for N with large prime factors.")
print("  For N=7919 (prime!): padding to 8000 = 2^6 × 5^3 is ~1000× faster.")


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: Roofline at different GPU tiers
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 3 — cuFFT Roofline: H100 vs A100 vs A10G")
print("━" * 68)
print()

BATCH_RL = 2048

print(f"  Batch={BATCH_RL}, C2C FP32, bandwidth-limited estimates")
print()
print(f"  {'N':>8}", end="")
for gname in GPUS:
    print(f"  {gname:>16}", end="")
print()
print(f"  {'':>8}", end="")
for _ in GPUS:
    print(f"  {'(µs per batch)':>16}", end="")
print()
print("  " + "─" * (10 + 18 * len(GPUS)))

for N_rl in [256, 512, 1024, 4096, 16384, 65536, 262144]:
    print(f"  {N_rl:>8,}", end="")
    for gname, bw in GPUS.items():
        t_us = fft_bandwidth_time_us(N_rl, BATCH_RL, bw, is_r2c=False)
        print(f"  {t_us:>16.3f}", end="")
    print()

print()
first_bw = list(GPUS.values())[0]
second_bw = list(GPUS.values())[1]
print(f"  H100/A100 speedup ratio: {first_bw/second_bw:.2f}× (matches HBM bandwidth ratio)")
print(f"  cuFFT performance is PURELY bandwidth-limited: no tensor core benefit.")


# ─────────────────────────────────────────────────────────────────────
# SECTION 4: FFT size selection guide
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 4 — FFT Size Selection Guide for Common Workloads")
print("━" * 68)
print()

workloads = [
    ("Audio, 44.1 kHz, 1024-tap FIR",
     "Signal chunks: N = 4096 (L=3072 + 1024 OLA overhead)",
     "M = 4096 = 2^12. Fully optimal. Good choice."),
    ("Audio, 44.1 kHz, 4096-tap FIR",
     "Signal chunks: N = 16384 (L=12288 + 4096 OLA overhead)",
     "M = 16384 = 2^14. Still optimal."),
    ("ML 1D sequence, length 1000",
     "Pad to next good size: 1000 → 1024 = 2^10",
     "Cost factor 1000: 6.4 (has 5^3 × 8). Cost factor 1024: 10.0 (2^10). Both ok."),
    ("SSM (Mamba), N=2048 context",
     "FFT over full sequence: N = 2048 = 2^11",
     "M = 2048. Optimal. FFT conv cost: 5×2048×11 ≈ 112K FLOPs."),
    ("2D Image 1920×1080",
     "Next good 2D size: 1920 = 2^7×3×5, 1080 = 2^3×3^3×5",
     "Both 7-smooth. cuFFT handles well. No padding needed."),
    ("Radio astronomy, N = 10007",
     "N = 10007 is prime! Pad to N = 10080 = 2^5×3^2×5×7",
     "10080 is 7-smooth. Huge speedup vs running N=10007."),
    ("Seismic, N = 1e6",
     "N = 1,000,000 = 2^6×5^6. 5-smooth. Good.",
     "Should be fast. Alternatively: 1,048,576 = 2^20 is even better."),
]

for name, scenario, recommendation in workloads:
    print(f"  {name}:")
    print(f"    Scenario:       {scenario}")
    print(f"    Recommendation: {recommendation}")
    print()
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