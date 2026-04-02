"""
PTX & SASS — Inline PTX in CUDA C, Reading SASS (cuobjdump) & Register Pressure
==================================================================================

Every CUDA kernel you write travels through a two-stage compilation pipeline
before a single instruction executes on the GPU:

    CUDA C / C++ → PTX (Parallel Thread eXecution) → SASS (Shader ASSembly)

PTX is NVIDIA's virtual ISA — an assembly language for an idealised GPU with
an unlimited number of registers and a regular instruction set. The CUDA
compiler (nvcc / NVVM) compiles your kernel to PTX, which NVIDIA ships with
the binary so that future GPU architectures can recompile it.

SASS is the real machine code — architecture-specific binary instructions for
a particular compute capability (sm_80 for A100, sm_90 for H100, etc.). The
PTX assembler (ptxas) lowers PTX to SASS, performing register allocation,
instruction scheduling, and hardware-specific optimisations along the way.

Why should a GPU programmer understand PTX and SASS?

    INLINE PTX: sometimes the compiler cannot generate the optimal instruction
    sequence. Inline assembly (`asm volatile("...")`) lets you insert specific
    PTX instructions directly into your CUDA kernel — for atomics with unusual
    semantics, for warp-level synchronisation primitives, for tensor core WMMA
    instructions, or for latency-critical cache-bypass loads.

    READING SASS: `cuobjdump --dump-sass` disassembles a compiled .cubin into
    human-readable SASS. Reading SASS tells you: how many registers the compiler
    allocated, whether your memory accesses were coalesced (LDG.E.128 vs multiple
    LDG.E.32), whether your loops were unrolled, and whether instruction dual-issue
    is happening. It's the ground truth about what the GPU actually executes.

    REGISTER PRESSURE: every thread has a private register file (256 KB per SM
    on A100, shared among all resident threads). Using too many registers per
    thread forces the compiler to SPILL registers to local memory (HBM), causing
    dramatic throughput drops. Using too few means poor latency hiding. The sweet
    spot — typically 32–64 registers per thread — requires understanding both
    PTX register semantics and SASS register allocation.

"""

import textwrap, re, math
import numpy as np

TOPIC_NAME   = "PTX & SASS — Inline PTX in CUDA C, Reading SASS (cuobjdump) & Register Pressure"
DISPLAY_NAME = "20 · PTX & SASS"
ICON         = "🔩"
SUBTITLE     = "PTX ISA · SASS Disassembly · cuobjdump · Inline PTX asm · Register Pressure · ILP"

THEORY = """

##### PART 1 — THE COMPILATION PIPELINE: FROM CUDA C TO SASS

### The Two-Stage Pipeline

    STAGE 1 — CUDA C/C++ to PTX (offline, at build time):
        nvcc  -arch=compute_80  -ptx  kernel.cu  -o  kernel.ptx
        The NVVM compiler (LLVM-based) lowers CUDA C to PTX.
        PTX is architecture-agnostic — "portable assembly" for NVIDIA GPUs.
        Output: human-readable .ptx text file.

    STAGE 2 — PTX to SASS (online, at first kernel launch or cubin generation):
        ptxas  --gpu-name sm_80  kernel.ptx  -o  kernel.cubin
        OR: nvcc  -arch=sm_80  kernel.cu  -o  kernel.cubin
        The PTX assembler performs register allocation and scheduling.
        Output: binary .cubin (or embedded in the .fatbin within the executable).

    FATBINARY: a compiled CUDA executable contains a .fatbin section with:
        - PTX for forward compatibility (JIT-compiled at first launch on new arch).
        - SASS cubins for target architectures (immediate execution, no JIT).

    VIEWING THE INTERMEDIATE FORMS:
        nvcc -ptx kernel.cu         → kernel.ptx (human-readable PTX)
        nvcc -cubin kernel.cu       → kernel.cubin (binary SASS)
        cuobjdump --dump-ptx  kernel.cubin  → print embedded PTX
        cuobjdump --dump-sass kernel.cubin  → print SASS disassembly
        nvdisasm kernel.cubin               → alternative SASS disassembler

### PTX: The Virtual ISA

    PTX has a RISC-like instruction set with:
        TYPED REGISTERS: %r (32-bit), %rd (64-bit), %f (float32), %fd (float64),
                         %p (predicate, 1-bit), %rs (16-bit).
        VIRTUAL REGISTERS: unlimited number of registers in PTX.
                           ptxas resolves these to physical registers.
        SSA-LIKE FORM: each register written exactly once (approximate SSA).
        EXPLICIT TYPES: every instruction encodes operand types in the opcode.

    EXAMPLE — PTX for  y = x * x + 1.0f:
        .reg .f32 %f<3>;           // declare 3 float32 registers
        ld.global.f32 %f0, [ptr];  // load x from global memory
        mul.f32 %f1, %f0, %f0;     // f1 = x * x
        add.f32 %f2, %f1, 0f3F800000;  // f2 = f1 + 1.0f (hex float)
        st.global.f32 [out], %f2;  // store result

    PTX STATE SPACES:
        .global:  main HBM (device memory). Accessed with ld.global / st.global.
        .shared:  SMEM. ld.shared / st.shared. Very fast within thread block.
        .local:   per-thread local memory (spill storage). Slow — goes to HBM.
        .const:   constant memory (64 KB, cached, broadcast). ld.const.
        .param:   kernel parameters. Accessible at entry only.
        .reg:     register file.

### SASS: The Real Machine Code

    SASS is architecture-specific binary assembly. Each GPU arch has its own:
        Volta (sm_70): 64-bit wide instructions, 128-bit predicates.
        Ampere (sm_80): 128-bit wide control words, wgmma not yet (that's Hopper).
        Hopper (sm_90): 128-bit GMMA (wgmma), TMA, new barrier instructions.

    SASS INSTRUCTION FORMAT (Ampere example):
        OPCODE.MODIFIER dest, src1, src2, src3 ;   CONTROL_CODE
        LDG.E.128 R4, [R2] ;                       // 128-bit global load into R4-R7
        FFMA R8, R4, R4, R2 ;                       // R8 = R4*R4 + R2 (fused MLA)
        STG.E [R10], R8 ;                           // store R8 to global

    CONTROL CODE (the SASS "instruction word" prefix):
        The upper bits of a SASS instruction encode:
            - Stall count: how many cycles to wait after this instruction.
            - Yield bit: hint to switch to another warp this cycle.
            - Write barrier: which write-dependency barrier index.
            - Read barrier: which read-dependency barrier to wait on.
        This is the SCOREBOARD / dependency tracking metadata used by the
        hardware to schedule independent instructions without interlocks.

### cuobjdump Output Format

    cuobjdump --dump-sass kernel.cubin  produces:

        Fatbin elf code:
        ================
        arch = sm_80
        code for sm_80
            Function : _Z9my_kernelPfS_i
            .headerflags    @"EF_CUDA_TEXMODE_UNIFIED EF_CUDA_64BIT_ADDRESS"
            /*0000*/     IMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x28] ;    /* 0x00000a00ff017624 */
            /*0010*/     S2R R0, SR_CTAID.X ;                         /* 0x0000000000007919 */
            /*0020*/     S2R R2, SR_TID.X ;                           /* 0x0000000000027919 */
            /*0030*/     IMAD R0, R0, c[0x0][0x0], R2 ;               /* 0x0000000000007a24 */
            ...

    READING THE OUTPUT:
        /*0000*/ : byte offset of this instruction in the kernel binary.
        IMAD.MOV.U32 : opcode (Integer MAD, MOV variant, unsigned 32-bit).
        R1 : destination register.
        c[0x0][0x28] : constant memory bank 0, offset 0x28 (kernel parameter).
        /* hex */ : the 64-bit raw instruction word (for cross-referencing docs).


##### PART 2 — PTX INSTRUCTION REFERENCE: THE MOST IMPORTANT INSTRUCTIONS

### Load and Store Instructions

    ld.global.f32   %f0, [%rd0];          // load 32-bit float from global mem
    ld.global.v4.f32 {%f0,%f1,%f2,%f3}, [%rd0]; // vectorised 128-bit load (4×f32)
    ld.global.nc.f32 %f0, [%rd0];         // non-coherent (L1 cache bypass)
    ld.shared.f32   %f0, [%r0];           // load from shared memory
    ld.const.f32    %f0, [%rd0+0];        // load from constant memory

    st.global.f32   [%rd0], %f0;          // store 32-bit float to global
    st.global.v4.f32 [%rd0], {%f0,%f1,%f2,%f3}; // vectorised store

    CACHE CONTROL MODIFIERS:
        .ca : cache at all levels (L1 + L2). Default for loads.
        .cg : cache at L2 only (bypass L1). Good for streaming writes.
        .cs : cache streaming (evict-first in L1). For once-accessed data.
        .lu : last use (mark data for early eviction). Saves L2 capacity.
        .cv : volatile (always reload from HBM — for synchronisation).
        .nc : non-coherent (use texture cache). Good for read-only broadcast.

### Arithmetic Instructions

    add.f32  %f2, %f0, %f1;   // float add
    fma.rn.f32 %f3, %f0, %f1, %f2;  // fused multiply-add, round-to-nearest
    mad.lo.u32 %r3, %r0, %r1, %r2;  // integer multiply-add (low 32 bits)
    shl.b32  %r1, %r0, 2;            // shift left by 2
    and.b32  %r2, %r0, 0xf;          // bitwise AND with 0xf

    IMPORTANT: fma.rn is the key instruction — it maps directly to the GPU's
    FMA hardware unit and achieves maximum throughput (2× single instructions).

### Control Flow

    setp.lt.s32 %p0, %r0, %r1;  // predicate: p0 = (r0 < r1)
    @%p0 bra TARGET;              // conditional branch (predicated)
    @!%p0 bra TARGET;             // branch if predicate is FALSE

    PREDICATED EXECUTION: instead of branches, single instructions can be
    predicated:
        @%p0 add.f32 %f2, %f0, %f1;  // execute add only if p0 is true
    This avoids warp divergence for short conditional sequences.

### Special Registers

    %tid.x, %tid.y, %tid.z    : thread index within block
    %ntid.x, %ntid.y, %ntid.z : block dimensions
    %ctaid.x, %ctaid.y         : block index in grid
    %nctaid.x                  : grid dimension
    %laneid                    : warp lane index (0..31)
    %warpid                    : warp index within SM
    %smid                      : SM index (useful for persistent kernels)
    %clock64                   : 64-bit SM clock counter (for benchmarking)
    %lanemask_lt               : bitmask of lanes with lower lane IDs

    In PTX:
        mov.u32 %r0, %tid.x;         // load thread X index
        mov.u64 %rd0, %clock64;       // read SM clock

### Warp Shuffle (PTX shfl)

    shfl.sync.idx.b32  %r2, %r1, %r0, 0x1f, 0xffffffff;
    // Broadcast %r1 from lane %r0 to all lanes in mask 0xffffffff
    // Width = 0x1f + 1 = 32 (full warp)

    shfl.sync.down.b32 %r2, %r1, 1, 0x1f, 0xffffffff;
    // Shift down by 1: lane i receives value from lane i+1

    This is the PTX encoding of CUDA's __shfl_sync intrinsics.

### Atomic Instructions

    atom.global.add.f32 %f0, [%rd0], %f1;   // atomic add to global memory
    atom.global.cas.b32 %r0, [%rd1], %r1, %r2;  // compare-and-swap
    red.global.add.f32 [%rd0], %f1;         // reduction (no return value, faster)

    red is faster than atom when you don't need the old value.


##### PART 3 — INLINE PTX IN CUDA C: ASM VOLATILE SYNTAX

### Basic Inline PTX Syntax

    asm volatile (
        "ptx_instruction_string"
        : output_constraints     // "=r"(c_var)
        : input_constraints      // "r"(c_var)
        : clobber_list           // "memory", "cc"
    );

    CONSTRAINT CODES:
        "r"  : 32-bit integer register (.u32 or .s32 or .b32)
        "l"  : 64-bit integer register (.u64)
        "f"  : 32-bit float register (.f32)
        "d"  : 64-bit float register (.f64)
        "h"  : 16-bit register (.u16)
        "=r" : output 32-bit integer register (written by asm)
        "=f" : output float register

### Practical Examples

    EXAMPLE 1 — Read SM clock counter:
        uint64_t clock;
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(clock));

    EXAMPLE 2 — Cache-bypass load (streaming):
        float val;
        asm volatile("ld.global.cs.f32 %0, [%1];"
                     : "=f"(val)
                     : "l"(ptr));

    EXAMPLE 3 — Vectorised 128-bit load (4 floats at once):
        float4 v;
        asm volatile("ld.global.v4.f32 {%0,%1,%2,%3}, [%4];"
                     : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
                     : "l"(ptr));

    EXAMPLE 4 — Warp lane ID:
        uint32_t lane;
        asm volatile("mov.u32 %0, %%laneid;" : "=r"(lane));

    EXAMPLE 5 — Atomic add with release semantics:
        float old;
        asm volatile("atom.global.add.f32 %0, [%1], %2;"
                     : "=f"(old)
                     : "l"(ptr), "f"(val));

    EXAMPLE 6 — Predicated store (write only if condition):
        asm volatile(
            "{ .reg .pred %p; setp.ne.s32 %p, %1, 0; @%p st.global.f32 [%2], %3; }"
            :
            : "r"(cond), "l"(ptr), "f"(val)
        );

### When to Use Inline PTX

    JUSTIFIED USE CASES:
        1. Accessing PTX instructions not yet exposed by CUDA intrinsics.
           (New instructions in each SM generation before compiler support.)
        2. Precise cache control modifiers (.cs, .lu, .cv) for prefetch hints.
        3. Reading special registers (%clock64, %smid, %laneid).
        4. Tensor core WMMA when using raw mma.sync instructions.
        5. Achieving specific instruction ordering the compiler reorders away.
        6. Custom atomic semantics not available via CUDA intrinsics.

    WHEN NOT TO USE:
        Most CUDA intrinsics (__shfl_sync, __ldg, atomicAdd) compile to the
        correct PTX. Inline PTX is needed only when the intrinsic doesn't exist
        or when you need exact control over instruction modifiers.
        Overuse makes kernels fragile — PTX syntax changes between toolkit versions.

### The volatile Keyword

    asm volatile prevents the compiler from:
        - Removing the instruction if the output appears unused.
        - Moving the instruction past other volatile instructions.
        - Merging multiple identical inline asm blocks.
    Always use volatile for: I/O operations, timing, side effects.


##### PART 4 — READING SASS WITH cuobjdump

### Key SASS Instructions to Recognise

    MEMORY ACCESS INSTRUCTIONS:
        LDG.E.32   R4, [R2]           // load 32-bit from global (R2 = pointer)
        LDG.E.128  R4, [R2]           // 128-bit vectorised load → R4, R5, R6, R7
        LDS.32     R4, [R2]           // load 32-bit from shared memory
        STG.E [R2], R4                // store to global
        STS [R2], R4                  // store to shared

    VECTORISED LOADS (sign of good coalescing):
        LDG.E.128 loads 16 bytes = 4 floats per thread.
        A warp of 32 threads × 16 bytes = 512 bytes = one L2 cache line (512 B).
        If you see LDG.E.32 instead of LDG.E.128: potential uncoalesced access.

    COMPUTE INSTRUCTIONS:
        FFMA R8, R4, R5, R6           // fused float multiply-add: R8 = R4*R5+R6
        FADD R4, R4, R5               // float add
        FMUL R4, R4, R5               // float multiply
        IMAD R4, R5, R6, R7           // integer multiply-add
        IADD3 R4, R3, R2, R1          // 3-way integer add

    CONTROL FLOW:
        BRA target                     // unconditional branch
        BSSY target                    // branch synchronisation (for divergence)
        SYNC                           // warp sync (barrier within warp)
        BAR.SYNC 0                     // thread block barrier

    SPECIAL:
        S2R R0, SR_TID.X               // special register to register (thread ID)
        CS2R R0, SRZ                   // constant to register (zero register)

### Interpreting SASS for Performance

    RECOGNISING GOOD CODE:
        ✅ LDG.E.128: 128-bit vectorised load (optimal bandwidth utilisation).
        ✅ FFMA: fused multiply-add (compiler fused separate mul+add).
        ✅ Interleaved LDG and FFMA: the compiler is hiding load latency.
        ✅ Loop body with no BAR.SYNC: warp-level code with no synchronisation.

    RECOGNISING BAD CODE:
        ❌ LDG.E.32 alone (not 128): loads are not 128-bit aligned or vectorised.
        ❌ Local memory accesses (LDL / STL): register spill to local memory!
        ❌ Many IMAD for indexing in the hot loop: expensive index computation.
        ❌ BRA with adjacent BSSY/SYNC: warp divergence handling.
        ❌ ST.E32 to .local[] space: compiler spilled a variable to L-mem.

    REGISTER SPILL DETECTION:
        In SASS: LDL and STL instructions access local memory.
        In PTX: ld.local and st.local instructions appear.
        To avoid: reduce register usage (--maxrregcount flag), restructure kernel.
        Spilled load latency: ~200 cycles (same as HBM) if not L1 cached.

### Annotated SASS Walkthrough — Simple Add Kernel

    // CUDA C source:
    // __global__ void add(float* a, float* b, float* c, int N) {
    //     int i = blockIdx.x * blockDim.x + threadIdx.x;
    //     if (i < N) c[i] = a[i] + b[i];
    // }

    // SASS (sm_80, simplified):
    /*0000*/ IMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x28]  // R1 = kernel param: a_ptr lo
    /*0010*/ S2R R0, SR_CTAID.X                        // R0 = blockIdx.x
    /*0020*/ S2R R2, SR_TID.X                          // R2 = threadIdx.x
    /*0030*/ IMAD R0, R0, c[0x0][0x0], R2              // R0 = blockIdx.x * blockDim.x + threadIdx.x
    /*0040*/ ISETP.GE.AND P0, PT, R0, c[0x0][0x18], PT // P0 = (i >= N) → bounds check
    /*0050*/ @P0 EXIT                                   // if i >= N: exit thread
    /*0060*/ IMAD.WIDE R2, R0, 4, c[0x0][0x20]        // R3:R2 = &a[i] (ptr + i*4)
    /*0070*/ IMAD.WIDE R4, R0, 4, c[0x0][0x28]        // R5:R4 = &b[i]
    /*0080*/ LDG.E R2, [R2]                             // R2 = a[i]
    /*0090*/ LDG.E R4, [R4]                             // R4 = b[i]
    /*00a0*/ FADD R2, R2, R4                            // R2 = a[i] + b[i]
    /*00b0*/ IMAD.WIDE R4, R0, 4, c[0x0][0x30]        // R5:R4 = &c[i]
    /*00c0*/ STG.E [R4], R2                             // c[i] = result
    /*00d0*/ EXIT

    READING THIS:
        c[0x0][0x28] = constant memory bank 0, offset 0x28 = kernel arg a_ptr lo.
        IMAD.WIDE: 32×32→64 bit multiply-add — efficient address computation.
        Two separate LDG.E.32 loads: could be improved to LDG.E.128 if
        pointers were 128-bit aligned and the kernel processed 4 elements/thread.


##### PART 5 — REGISTER PRESSURE: OCCUPANCY AND THE REGISTER FILE

### The GPU Register File

    A100 register file: 256 KB per SM.
        = 65,536 registers × 4 bytes/register = 256 KB.
    These 65,536 registers are SHARED among all resident warps on the SM.

    If a kernel uses 64 registers/thread:
        64 regs/thread × 32 threads/warp × N warps = 65,536 total
        → Maximum N warps on this SM: 65,536 / (64 × 32) = 32 warps.
        A100 has 64 warps max per SM → register occupancy = 32/64 = 50%.

    If a kernel uses 128 registers/thread:
        128 × 32 × N = 65,536
        → N = 16 warps → occupancy = 16/64 = 25%.

    If a kernel uses 32 registers/thread:
        32 × 32 × N = 65,536
        → N = 64 warps → occupancy = 64/64 = 100%.

### Occupancy vs Performance: The Latency Hiding Argument

    OCCUPANCY is not directly proportional to performance. The reason to
    maximise occupancy is LATENCY HIDING:

        GPU throughput model: while one warp waits for a memory load (~200
        cycles), the SM can switch to another ready warp (zero overhead).
        If there are many resident warps, the SM is always running something.

    MINIMUM OCCUPANCY for full latency hiding:
        Memory latency: ~200 cycles.
        Issue rate: 4 instructions/cycle per warp (on A100).
        Warp slots needed to hide latency: 200 / 4 = 50 warps (theoretical).
        In practice: 16–32 warps often sufficient (not all instructions are loads).

    DIMINISHING RETURNS:
        Going from 16→32 warps: significant improvement (latency hiding).
        Going from 32→48 warps: moderate improvement.
        Going from 48→64 warps: minimal improvement.
        Below 16 warps: SM frequently idle, large performance loss.

### Controlling Register Usage

    NVCC FLAG — Limit registers per thread:
        nvcc --maxrregcount=64 kernel.cu
        Forces the compiler to use at most 64 registers per thread.
        If the kernel NEEDS more: excess variables SPILL to local memory.

    LAUNCH BOUNDS — per-kernel hint to the compiler:
        __global__ __launch_bounds__(maxThreadsPerBlock, minBlocksPerSM)
        void my_kernel(...) { ... }

        maxThreadsPerBlock: helps compiler allocate minimum registers for
        this block size, improving occupancy.
        minBlocksPerSM: forces the compiler to target N blocks per SM minimum,
        which in turn limits register usage to achieve that occupancy.

    PRAGMA — per-function register limit (CUDA 11.5+):
        #pragma nv_diag_suppress 174   // suppress "unused variable" for pragma
        #pragma maxrregcount 64
        __device__ void helper() { ... }

    REGISTER PRESSURE TRACING — view register count:
        nvcc --ptxas-options=-v kernel.cu
        Prints: "ptxas info: Function '_Z6kernelPf' uses 32 registers,
                 0 bytes smem, 0 bytes lmem, 48 bytes cmem[0]"
        lmem > 0 means SPILLING.

### Minimising Register Usage: Practical Techniques

    TECHNIQUE 1 — Break long dependency chains:
        Instead of:
            float a = x * y;
            float b = a * z;     // b depends on a (long dep chain, a lives long)
            float c = b + w;     // c depends on b
        Write:
            float c = (x * y) * z + w;   // one expression: compiler may collapse

    TECHNIQUE 2 — Avoid storing intermediate results:
        // BAD: stores 8 intermediate values (8 registers alive at once)
        float r[8]; for (int i=0; i<8; i++) r[i] = load(i); sum = r[0]+...+r[7];
        // GOOD: accumulate directly (2 registers live at once)
        float sum = 0; for (int i=0; i<8; i++) sum += load(i);

    TECHNIQUE 3 — Use __restrict__ for pointer aliasing:
        void kernel(float* __restrict__ a, float* __restrict__ b) { ... }
        Tells the compiler a and b never overlap → enables register reuse.

    TECHNIQUE 4 — Reduce tiling factor:
        In a tiled kernel, each additional tiling factor (processing 4 elements
        per thread instead of 1) requires 4× the registers for live tile values.
        Trade-off: fewer launches vs more register pressure.


##### PART 6 — INSTRUCTION SCHEDULING AND ILP

### The Scoreboard and Out-of-Order Execution

    NVIDIA GPUs execute instructions IN-ORDER within a warp.
    (No OOO hardware like a CPU superscalar core.)
    However: the SM can SWITCH to another warp when the current one stalls.

    THE SCOREBOARD tracks outstanding memory operations:
        Each load (LDG.E) is assigned a "read dependency barrier" index.
        A subsequent instruction that reads the loaded register must wait
        until the scoreboard for that barrier is cleared.
        In SASS: the control code encodes which barrier each instruction
        waits on and which barrier each instruction clears.

    WAIT CODES in SASS CONTROL WORD:
        The 23-bit control code per instruction encodes:
            - stall_cycles (4 bits): how many cycles after issue before the
              NEXT instruction can issue. 0 = issue every cycle.
            - yield (1 bit): suggest warp switch to the scheduler.
            - write barrier index (3 bits): which write-dependency barrier.
            - read barrier index (3 bits): which read-barrier to wait for.

### Instruction-Level Parallelism (ILP) Within a Warp

    Even without warp switching, a single warp can achieve ILP by having
    multiple INDEPENDENT instructions that can issue to different pipelines:

    DUAL ISSUE: A100 can issue 2 independent instructions per cycle per warp:
        Cycle N:   FFMA R0, R1, R2, R3   (goes to FMA pipe)
        Cycle N:   LDG.E R4, [R5]         (goes to load pipe, same cycle)
        → Both issue simultaneously.

    PIPELINE UNITS (A100):
        FP32 FMA: 64 ops/cycle per SM (2 units × 32 threads/warp).
        INT32:    16 ops/cycle per SM.
        FP64:     32 ops/cycle per SM.
        Load:     128 bytes/cycle per SM.
        Store:    64 bytes/cycle per SM.

    ACHIEVING DUAL ISSUE — keep independent operations interleaved:
        // Poor: dependent chain (FFMA output feeds next FFMA)
        FFMA R0, R1, R2, R3;   // stall: must wait for R0
        FFMA R4, R0, R5, R6;   // stall 6 cycles for R0

        // Good: independent chain (different output registers)
        FFMA R0, R1, R2, R3;
        FFMA R4, R5, R6, R7;   // R4 doesn't depend on R0 → dual-issue!
        FFMA R8, R9, R10, R11; // triple (if another pipe available)

### Software Pipelining: Load Ahead of Compute

    The key SASS pattern for memory-bound kernels is LOAD-COMPUTE OVERLAP:

        LOOP:
            LDG.E R4, [next_ptr] ;  // ISSUE: load next batch (async)
            FFMA R8, R4, R5, R6 ;   // USE: compute with current batch
            // ... compute on current ...
            DEPBAR ;                 // WAIT: ensure R4 is ready
            STG.E [out_ptr], R8 ;    // STORE: write result
            // next_ptr advances, current becomes next

    The LDG.E and FFMA overlap: the load unit and compute unit run simultaneously.
    DEPBAR (dependency barrier) stalls only when the loaded data is actually needed.
    This is the hardware implementation of the double-buffer pattern from Module 38.


##### PART 7 — PROFILING WITH NCU: CONNECTING SASS TO HARDWARE METRICS

### NCU Source-Level Attribution

    Nsight Compute (ncu) can attribute performance metrics to SOURCE LINES
    and SASS INSTRUCTIONS:
        ncu --set detailed --source-level 2 ./kernel_binary
        Opens in the GUI with per-instruction metrics:
            - Cycles stalled per instruction.
            - Memory throughput per LDG instruction.
            - Issue rate per FMA instruction.

    KEY NCU METRICS FOR REGISTER ANALYSIS:
        sm__inst_executed_op_generic_st.sum.per_cycle_elapsed
            → stores per cycle; should be near peak for store-heavy kernels.
        l1tex__data_pipe_lsu_wavefronts_mem_local.sum
            → LOCAL MEMORY WAVEFRONTS — if non-zero: REGISTER SPILL detected!
        smsp__warp_issue_stalled_long_scoreboard_pct
            → fraction of cycles stalled waiting for scoreboard (memory latency).
        smsp__warp_issue_stalled_short_scoreboard_pct
            → stalled waiting for short-latency dependencies (math result).

### Connecting NCU to SASS

    NCU's "Source Counters" view shows SASS instructions with annotations:
        FFMA R8, R4, R5, R6  [Issue: 1.00 per cycle, Stall: 0.00]
        LDG.E R4, [R2]       [Issue: 1.00 per cycle, Stall: 0.23 after]
    The 0.23 stall after LDG means: 23% of the time, the warp stalls for
    23% of cycles after this load waiting for the data.

    Seeing high stalls after LDG: either the memory system is saturated
    or there aren't enough warps to hide the latency.

### --ptxas-options=-v: Compile-Time Register Report

    The single most useful flag for understanding register usage:
        nvcc --ptxas-options=-v kernel.cu
        Output per kernel:
            ptxas info: Function '_Z9my_kernelPfS_i':
                .reg = 48,  .lmem = 0,  .smem = 8192,  .cmem[0] = 64
        .reg:   registers per thread. Target: 32–64 for good occupancy.
        .lmem:  local memory bytes per thread. Target: 0. Any > 0 = spill!
        .smem:  shared memory bytes per block.
        .cmem:  constant memory bytes used.

"""

OPERATIONS = {

    "1 · PTX Instruction Model — Registers, State Spaces & Typed Ops": {
        "description": (
            "Simulate PTX's register model and typed instruction set in Python. "
            "Show how PTX virtual registers map to types (.f32, .u32, .u64, .pred). "
            "Implement a PTX interpreter for key instructions: ld.global, st.global, "
            "fma, mad, setp, predicated branch. Trace PTX execution for a simple "
            "kernel (y = ax + b). Show how PTX state spaces (.global, .shared, "
            ".local, .const) differ in bandwidth and latency."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math

print("=" * 68)
print("  PTX INSTRUCTION MODEL — Registers, State Spaces & Typed Ops")
print("=" * 68)
print()

np.random.seed(42)


class PTXRegisters:
    """Simulate PTX virtual register file with typed registers."""

    def __init__(self):
        self.regs   = {}       # register name → value
        self.preds  = {}       # predicate register name → bool
        self.global_mem = {}   # address → value (simulated global memory)
        self.shared_mem = {}   # address → value (simulated SMEM)
        self.local_mem  = {}   # address → value (spill storage)
        self.const_mem  = {}   # const bank → value

    def set(self, name, val):
        self.regs[name] = val

    def get(self, name):
        if name in self.regs:
            return self.regs[name]
        raise ValueError(f"Register {name} not found")

    def set_pred(self, name, val):
        self.preds[name] = bool(val)

    def get_pred(self, name):
        return self.preds.get(name, False)


class PTXInterpreter:
    """
    Minimal PTX interpreter for educational simulation.
    Supports: ld.global, st.global, ld.shared, fma.rn, add, mul, mad,
              setp, predicated branch, mov.
    """
    def __init__(self):
        self.regs     = PTXRegisters()
        self.pc       = 0
        self.trace    = []

    def ld_global_f32(self, dst, addr):
        val = self.regs.global_mem.get(addr, 0.0)
        self.regs.set(dst, np.float32(val))
        self.trace.append(f"ld.global.f32 {dst}, [{addr}] → {val:.4f}")

    def st_global_f32(self, addr, src):
        val = self.regs.get(src)
        self.regs.global_mem[addr] = val
        self.trace.append(f"st.global.f32 [{addr}], {src} ← {val:.4f}")

    def ld_shared_f32(self, dst, addr):
        val = self.regs.shared_mem.get(addr, 0.0)
        self.regs.set(dst, np.float32(val))
        self.trace.append(f"ld.shared.f32 {dst}, [{addr}] → {val:.4f}")

    def st_shared_f32(self, addr, src):
        val = self.regs.get(src)
        self.regs.shared_mem[addr] = val
        self.trace.append(f"st.shared.f32 [{addr}], {src} ← {val:.4f}")

    def ld_local_f32(self, dst, addr):
        """Simulates register spill read — expensive."""
        val = self.regs.local_mem.get(addr, 0.0)
        self.regs.set(dst, np.float32(val))
        self.trace.append(f"ld.local.f32 {dst}, [{addr}] → {val:.4f}  ⚠ SPILL READ")

    def st_local_f32(self, addr, src):
        """Simulates register spill write — expensive."""
        val = self.regs.get(src)
        self.regs.local_mem[addr] = val
        self.trace.append(f"st.local.f32 [{addr}], {src} ← {val:.4f}  ⚠ SPILL WRITE")

    def fma_rn(self, dst, a, b, c):
        """fma.rn.f32 dst, a, b, c  →  dst = a*b + c"""
        va, vb, vc = self.regs.get(a), self.regs.get(b), self.regs.get(c)
        result = np.float32(np.float64(va) * np.float64(vb) + np.float64(vc))
        self.regs.set(dst, result)
        self.trace.append(f"fma.rn.f32 {dst}, {a}, {b}, {c} → {result:.4f}")

    def add_f32(self, dst, a, b):
        va = a if not isinstance(a, str) else self.regs.get(a)
        vb = b if not isinstance(b, str) else self.regs.get(b)
        result = np.float32(va + vb)
        self.regs.set(dst, result)
        self.trace.append(f"add.f32 {dst}, {a}, {b} → {result:.4f}")

    def mul_f32(self, dst, a, b):
        va = a if not isinstance(a, str) else self.regs.get(a)
        vb = b if not isinstance(b, str) else self.regs.get(b)
        result = np.float32(va * vb)
        self.regs.set(dst, result)
        self.trace.append(f"mul.f32 {dst}, {a}, {b} → {result:.4f}")

    def mov_f32(self, dst, val):
        self.regs.set(dst, np.float32(val))
        self.trace.append(f"mov.f32 {dst}, {val} → {np.float32(val):.4f}")

    def setp_lt_f32(self, pred, a, b):
        va = self.regs.get(a)
        vb = b if not isinstance(b, str) else self.regs.get(b)
        result = va < vb
        self.regs.set_pred(pred, result)
        self.trace.append(f"setp.lt.f32 {pred}, {a}, {b} → {result}")

    def setp_ge_s32(self, pred, a, b_imm):
        va = self.regs.get(a)
        result = int(va) >= b_imm
        self.regs.set_pred(pred, result)
        self.trace.append(f"setp.ge.s32 {pred}, {a}, {b_imm} → {result}")

    def print_trace(self):
        for step in self.trace:
            print(f"    {step}")


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: Trace PTX for y = a*x + b
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — PTX Trace: y = a*x + b  (SAXPY one element)")
print("━" * 68)
print()

ptx = PTXInterpreter()

# Simulate kernel parameters (from .param space / constant memory)
# a=2.5, b=1.0, x stored at address 100, result at address 200
ptx.regs.global_mem[100] = 3.0   # x = 3.0
ptx.regs.global_mem[200] = 0.0   # y = (output)

print("  PTX pseudocode for  y = a*x + b  (a=2.5, b=1.0, x=3.0):")
print()
print("    .reg .f32 %f<4>;")
print("    ld.global.f32 %f0, [x_ptr];    // load x")
print("    mov.f32 %f1, 2.5;              // a = 2.5")
print("    mov.f32 %f2, 1.0;              // b = 1.0")
print("    fma.rn.f32 %f3, %f0, %f1, %f2;// %f3 = x*a + b")
print("    st.global.f32 [y_ptr], %f3;    // store y")
print()
print("  Execution trace:")
print()

ptx.ld_global_f32("%f0", 100)
ptx.mov_f32("%f1", 2.5)
ptx.mov_f32("%f2", 1.0)
ptx.fma_rn("%f3", "%f0", "%f1", "%f2")
ptx.st_global_f32(200, "%f3")
ptx.print_trace()

print()
expected = 3.0 * 2.5 + 1.0
result   = ptx.regs.global_mem[200]
print(f"  Expected y = 3.0 × 2.5 + 1.0 = {expected}")
print(f"  Computed y = {result:.4f}  {'✅' if abs(result - expected) < 1e-4 else '❌'}")


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: State space bandwidth model
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 2 — PTX State Spaces: Bandwidth & Latency Model")
print("━" * 68)
print()

state_spaces = [
    ("ld.global",  "HBM (global)",   3350,  200,  "Main device memory"),
    ("ld.shared",  "SMEM",           19000, 20,   "Per-block scratchpad (~19 TB/s sm_80)"),
    ("ld.const",   "Const cache",    3350,  5,    "64KB broadcast cache (all threads same addr)"),
    ("ld.local",   "Local (spill)",  3350,  200,  "⚠ REGISTER SPILL — same as global!"),
    ("ld.global.nc","Texture cache", 3350,  80,   "Read-only non-coherent, L1 cached"),
]

print(f"  {'Instruction':<18}  {'Storage':<26}  {'BW (GB/s)':>10}  "
      f"{'Latency (cyc)':>15}  {'Notes'}")
print("  " + "─" * 80)

for instr, storage, bw, lat, note in state_spaces:
    print(f"  {instr:<18}  {storage:<26}  {bw:>10,}  "
          f"{lat:>15}  {note}")

print()
print("  ⚠  ld.local is EXACTLY as slow as ld.global — it goes to HBM!")
print("  Register spills are catastrophic: eliminate them with --maxrregcount.")


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: Predicated execution trace
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 3 — Predicated Execution: Avoiding Warp Divergence")
print("━" * 68)
print()

print("  CONDITIONAL: if (x > 0) y = x * 2; else y = -x;")
print()
print("  BRANCHING approach (causes warp divergence):")
print("    setp.le.f32 %p0, %f0, 0.0;")
print("    @%p0 bra ELSE;              // diverges here if warp has mixed signs")
print("    mul.f32 %f1, %f0, 2.0;")
print("    bra DONE;")
print("  ELSE:")
print("    neg.f32 %f1, %f0;")
print("  DONE:")
print()
print("  PREDICATED approach (no divergence — both paths execute in parallel):")
print("    setp.gt.f32 %p0, %f0, 0.0; // p0 = (x > 0)")
print("    mul.f32 %f1, %f0, 2.0;     // f1 = x*2 (always computed)")
print("    neg.f32 %f2, %f0;           // f2 = -x  (always computed)")
print("    selp.f32 %f3, %f1, %f2, %p0; // f3 = p0 ? f1 : f2")
print()
print("  selp (select predicate): a single instruction, no branch, no divergence.")
print("  Best for: short conditionals where both paths are cheap to compute.")
print("  Branching wins: when one path is rare and expensive to compute.")
print()

# Demonstrate numerically
ptx2 = PTXInterpreter()
for x_val in [2.0, -1.5, 0.0, 3.5, -4.0]:
    ptx2.regs.set("%f0", np.float32(x_val))
    ptx2.setp_lt_f32("%p0", "%f0", np.float32(0.0))

    # Predicated result: if x < 0 → neg; else → mul by 2
    # (simulate selp by Python conditional on predicate)
    is_neg = ptx2.regs.get_pred("%p0")
    if is_neg:
        ptx2.mul_f32("%f1", "%f0", np.float32(-1.0))
    else:
        ptx2.mul_f32("%f1", "%f0", np.float32(2.0))

    expected_v = (-x_val) if x_val < 0 else (2.0 * x_val)
    result_v   = ptx2.regs.get("%f1")
    print(f"  x={x_val:>5.1f}: p0={is_neg}, %f1 = {result_v:.4f}  "
          f"(expected {expected_v:.4f})  "
          f"{'✅' if abs(result_v - expected_v) < 1e-4 else '❌'}")
    ptx2.trace.clear()
''',
    },

    "2 · SASS Instruction Anatomy — Opcodes, Modifiers & Dependency Codes": {
        "description": (
            "Build a SASS instruction parser that decodes opcodes, operand modifiers, "
            "and control words from cuobjdump output. Show how to identify memory "
            "access width (LDG.E.32 vs LDG.E.128), detect register spills (LDL/STL), "
            "spot compute vs memory instructions. Trace a complete SASS basic block "
            "for a simple reduce kernel. Explain the control code fields: stall cycles, "
            "yield bit, and dependency barriers."
        ),
        "language": "python",
        "code": '''
import numpy as np
import re

print("=" * 68)
print("  SASS INSTRUCTION ANATOMY — Opcodes, Modifiers & Dependency Codes")
print("=" * 68)
print()


# ─────────────────────────────────────────────────────────────────────
# SASS instruction database (representative subset)
# ─────────────────────────────────────────────────────────────────────

SASS_OPCODES = {
    # Memory — loads
    "LDG":    {"class": "mem",     "unit": "load",    "latency": 200,
               "desc": "Load from Global memory"},
    "LDS":    {"class": "mem",     "unit": "load",    "latency": 20,
               "desc": "Load from Shared memory"},
    "LDC":    {"class": "mem",     "unit": "load",    "latency": 5,
               "desc": "Load from Constant memory"},
    "LDL":    {"class": "mem",     "unit": "load",    "latency": 200,
               "desc": "Load from Local (spill) — EXPENSIVE"},
    "LDSM":   {"class": "mem",     "unit": "load",    "latency": 20,
               "desc": "Load from Shared Matrix (for TC)"},
    # Memory — stores
    "STG":    {"class": "mem",     "unit": "store",   "latency": 0,
               "desc": "Store to Global"},
    "STS":    {"class": "mem",     "unit": "store",   "latency": 0,
               "desc": "Store to Shared"},
    "STL":    {"class": "mem",     "unit": "store",   "latency": 0,
               "desc": "Store to Local (spill) — EXPENSIVE"},
    # Float compute
    "FFMA":   {"class": "compute", "unit": "fma",     "latency": 4,
               "desc": "Float Fused Multiply-Add"},
    "FADD":   {"class": "compute", "unit": "fadd",    "latency": 4,
               "desc": "Float Add"},
    "FMUL":   {"class": "compute", "unit": "fmul",    "latency": 4,
               "desc": "Float Multiply"},
    "MUFU":   {"class": "compute", "unit": "sfu",     "latency": 8,
               "desc": "Multi-Function Unit (rcp, sqrt, sin, cos)"},
    # Integer compute
    "IMAD":   {"class": "compute", "unit": "ialu",    "latency": 4,
               "desc": "Integer Multiply-Add"},
    "IADD3":  {"class": "compute", "unit": "ialu",    "latency": 4,
               "desc": "3-way Integer Add"},
    "ISETP":  {"class": "compute", "unit": "ialu",    "latency": 4,
               "desc": "Integer Set Predicate"},
    "FSETP":  {"class": "compute", "unit": "fcomp",   "latency": 4,
               "desc": "Float Set Predicate"},
    "SHL":    {"class": "compute", "unit": "ialu",    "latency": 4,
               "desc": "Shift Left"},
    "SHR":    {"class": "compute", "unit": "ialu",    "latency": 4,
               "desc": "Shift Right"},
    # Control
    "BRA":    {"class": "control", "unit": "branch",  "latency": 0,
               "desc": "Branch"},
    "EXIT":   {"class": "control", "unit": "branch",  "latency": 0,
               "desc": "Thread Exit"},
    "BAR":    {"class": "control", "unit": "barrier", "latency": 0,
               "desc": "Thread Block Barrier"},
    "DEPBAR": {"class": "control", "unit": "barrier", "latency": 0,
               "desc": "Dependency Barrier — wait for outstanding loads"},
    # Special
    "S2R":    {"class": "special", "unit": "s2r",     "latency": 20,
               "desc": "Special Register to Register (thread IDs, etc.)"},
    "ATOM":   {"class": "mem",     "unit": "atom",    "latency": 200,
               "desc": "Atomic Memory Operation"},
    "RED":    {"class": "mem",     "unit": "atom",    "latency": 0,
               "desc": "Reduction (no return value, faster than ATOM)"},
    "SHFL":   {"class": "compute", "unit": "shfl",    "latency": 4,
               "desc": "Warp Shuffle"},
    "HMMA":   {"class": "compute", "unit": "tc",      "latency": 16,
               "desc": "Half-precision Matrix Multiply-Accumulate (Tensor Core)"},
}

SASS_MODIFIERS = {
    "E":      "Even (aligned memory access)",
    "128":    "128-bit width (4× float32 per thread)",
    "64":     "64-bit width (2× float32 per thread)",
    "32":     "32-bit width (default)",
    "16":     "16-bit width",
    "WIDE":   "Wide multiply (32×32→64 result)",
    "MOV":    "Move variant (no actual multiply)",
    "AND":    "Logical AND mask",
    "U32":    "Unsigned 32-bit",
    "S32":    "Signed 32-bit",
    "RN":     "Round to Nearest Even",
    "SYNC":   "Synchronised version",
}


def parse_sass_instruction(line):
    """Parse a cuobjdump SASS line into components."""
    line = line.strip()

    # Extract byte offset: /*0000*/
    offset_match = re.match(r'/\*([0-9a-fA-F]+)\*/', line)
    offset = int(offset_match.group(1), 16) if offset_match else None

    # Remove offset prefix and hex suffix
    instr_part = re.sub(r'/\*[^*]+\*/', '', line).strip()
    instr_part = re.sub(r';.*$', '', instr_part).strip()

    # Split opcode (may have dot-separated modifiers) from operands
    parts   = instr_part.split(None, 1)
    if not parts:
        return None
    opcode_full = parts[0]
    operands    = parts[1].strip() if len(parts) > 1 else ""

    # Split opcode into base + modifiers
    tokens = opcode_full.split('.')
    base   = tokens[0]
    mods   = tokens[1:]

    info   = SASS_OPCODES.get(base, {"class": "unknown", "unit": "?",
                                      "latency": 0, "desc": "Unknown instruction"})

    return {
        "offset":    offset,
        "opcode":    base,
        "modifiers": mods,
        "operands":  operands,
        "class":     info["class"],
        "unit":      info["unit"],
        "latency":   info["latency"],
        "desc":      info["desc"],
        "is_spill":  base in ("LDL", "STL"),
        "is_tc":     base in ("HMMA", "IMMA", "WMMA"),
        "mem_width": next((int(m) for m in mods if m.isdigit()), 32),
    }


def analyze_sass_block(sass_lines):
    """Analyse a list of SASS instructions and report key metrics."""
    instructions = [parse_sass_instruction(l) for l in sass_lines
                    if l.strip() and not l.strip().startswith('.')]
    instructions = [i for i in instructions if i]

    stats = {
        "total": len(instructions),
        "compute": sum(1 for i in instructions if i["class"] == "compute"),
        "memory":  sum(1 for i in instructions if i["class"] == "mem"),
        "control": sum(1 for i in instructions if i["class"] == "control"),
        "spills":  sum(1 for i in instructions if i["is_spill"]),
        "ldg_128": sum(1 for i in instructions
                       if i["opcode"] == "LDG" and i["mem_width"] == 128),
        "ldg_32":  sum(1 for i in instructions
                       if i["opcode"] == "LDG" and i["mem_width"] <= 32),
    }
    return instructions, stats


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: Decode representative SASS instructions
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — SASS Instruction Decoder: Opcode + Modifier Analysis")
print("━" * 68)
print()

sample_sass = [
    "/*0000*/ LDG.E.128 R4, [R2] ;",
    "/*0010*/ LDG.E.32 R8, [R6] ;",
    "/*0020*/ STL [R10+0x10], R4 ;",
    "/*0030*/ FFMA R12, R4, R5, R6 ;",
    "/*0040*/ IMAD.WIDE R2, R0, 0x4, c[0x0][0x20] ;",
    "/*0050*/ ISETP.GE.AND P0, PT, R0, c[0x0][0x18], PT ;",
    "/*0060*/ S2R R0, SR_TID.X ;",
    "/*0070*/ DEPBAR.LE SB0, 0x1 ;",
    "/*0080*/ HMMA.884.F32 R20, R8, R16, RZ ;",
    "/*0090*/ SHFL.BFLY PT, R4, R4, 0x10, 0x1f ;",
]

print(f"  {'Offset':>8}  {'Opcode':<10}  {'Modifiers':<16}  "
      f"{'Class':<10}  {'Unit':<10}  {'Lat':>5}  {'Flags'}")
print("  " + "─" * 74)

for line in sample_sass:
    i = parse_sass_instruction(line)
    if i is None:
        continue
    flags = []
    if i["is_spill"]:  flags.append("⚠SPILL")
    if i["is_tc"]:     flags.append("TC")
    if i["mem_width"] == 128: flags.append("128b✅")
    elif i["class"] == "mem" and i["mem_width"] == 32: flags.append("32b⚠")
    flag_str = " ".join(flags)
    mods_str = ".".join(i["modifiers"])
    print(f"  {i['offset']:>#8x}  {i['opcode']:<10}  {mods_str:<16}  "
          f"{i['class']:<10}  {i['unit']:<10}  {i['latency']:>5}  {flag_str}")


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: Annotated SASS walkthrough — add kernel
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 2 — Annotated SASS: add_kernel (c[i] = a[i] + b[i])")
print("━" * 68)
print()

add_kernel_sass = """
// Source: __global__ void add(float* a, float* b, float* c, int N)
// Compiled for sm_80 (A100), -O3
/*0000*/ IMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x28]  // R1 = a_ptr_lo (from const param)
/*0010*/ S2R R0, SR_CTAID.X                        // R0 = blockIdx.x
/*0020*/ S2R R2, SR_TID.X                          // R2 = threadIdx.x
/*0030*/ IMAD R0, R0, c[0x0][0x0], R2              // R0 = blockIdx.x*blockDim.x + threadIdx.x
/*0040*/ ISETP.GE.AND P0, PT, R0, c[0x0][0x18], PT // P0 = (i >= N)
/*0050*/ @P0 EXIT                                   // if out-of-bounds: exit
/*0060*/ IMAD.WIDE R2, R0, 4, c[0x0][0x20]        // R3:R2 = a + i*4  (64-bit addr)
/*0070*/ IMAD.WIDE R4, R0, 4, c[0x0][0x28]        // R5:R4 = b + i*4
/*0080*/ LDG.E R2, [R2]                             // R2 = a[i]  (32-bit load)
/*0090*/ LDG.E R4, [R4]                             // R4 = b[i]
/*00a0*/ IMAD.WIDE R6, R0, 4, c[0x0][0x30]        // R7:R6 = c + i*4
/*00b0*/ FADD R2, R2, R4                            // R2 = a[i] + b[i]
/*00c0*/ STG.E [R6], R2                             // c[i] = result
/*00d0*/ EXIT
"""

print("  Source: __global__ void add(float* a, float* b, float* c, int N)")
print()

annotations = [
    ("IMAD.MOV.U32",  "Load kernel arg (a_ptr) from constant mem c[0][0x28]"),
    ("S2R SR_CTAID.X","Read blockIdx.x into R0"),
    ("S2R SR_TID.X",  "Read threadIdx.x into R2"),
    ("IMAD (R0)",     "i = blockIdx.x * blockDim.x + threadIdx.x"),
    ("ISETP.GE",      "Compute bounds check predicate P0 = (i >= N)"),
    ("@P0 EXIT",      "Predicated exit for out-of-bounds threads"),
    ("IMAD.WIDE (R2)","64-bit addr: &a[i] = a_ptr + i*4"),
    ("IMAD.WIDE (R4)","64-bit addr: &b[i] = b_ptr + i*4"),
    ("LDG.E (R2)",    "Load a[i] — 32-bit ⚠ (could be 128-bit with vectorisation)"),
    ("LDG.E (R4)",    "Load b[i] — 32-bit"),
    ("IMAD.WIDE (R6)","64-bit addr: &c[i] = c_ptr + i*4"),
    ("FADD",          "float add: R2 = a[i] + b[i]"),
    ("STG.E",         "Store result to c[i]"),
    ("EXIT",          "Thread exits"),
]

for i, (instr, note) in enumerate(annotations):
    print(f"  {i+1:>3}. {instr:<22} // {note}")

print()
print("  ⚠ OPTIMISATION OPPORTUNITY:")
print("  LDG.E.32 (32-bit) instead of LDG.E.128 (128-bit).")
print("  With 128-bit vectorised loads: process 4 elements per thread.")
print("  This halves the instruction count and doubles effective load bandwidth.")


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: Control code (scoreboard) analysis
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 3 — SASS Control Codes: Stall Cycles & Dependency Barriers")
print("━" * 68)
print()

print("  Each SASS instruction has a 23-bit CONTROL WORD encoding:")
print()
print("    Bits [3:0]   = stall_cycles:  cycles to wait before NEXT issue")
print("    Bit  [4]     = yield:         suggest warp switch to scheduler")
print("    Bits [8:5]   = write barrier: which write-dep barrier to set")
print("    Bits [17:15] = read barrier:  which read-dep barrier to wait on")
print("    Bits [14:9]  = use barriers:  which barriers this instruction uses")
print()

ctrl_examples = [
    (0b00000000_00000000_0000, "0x00000", "Issue next cycle, no deps"),
    (0b00000000_00000000_0110, "0x00006", "Stall 6 cycles after this (FP result ready)"),
    (0b00000000_00000001_0000, "0x00010", "Yield bit set — suggest warp switch"),
    (0b00000000_00010000_0000, "0x01000", "Set write barrier SB0 (scoreboard 0)"),
    (0b00000000_10000000_0000, "0x08000", "Wait on read barrier SB3 before issue"),
]

print(f"  {'Control code':>14}  {'Stall':>7}  {'Yield':>7}  {'Meaning'}")
print("  " + "─" * 60)
for ctrl, hex_str, meaning in ctrl_examples:
    stall = ctrl & 0xF
    yield_bit = (ctrl >> 4) & 1
    print(f"  {hex_str:>14}  {stall:>7}  {yield_bit:>7}  {meaning}")

print()
print("  HOW DEPBAR WORKS in practice:")
print("    LDG.E R4, [R2]  — sets SB0 (outstanding load barrier 0)")
print("    FFMA R8, R6, R6, R7  — independent, issues next cycle")
print("    DEPBAR.LE SB0, 0x1   — WAIT: stall until SB0 clears (R4 is ready)")
print("    FFMA R9, R4, R4, R8  — now safe to use R4")
print()
print("  The DEPBAR position is chosen by ptxas to maximise overlap between")
print("  the load latency and independent compute instructions.")
''',
    },

    "3 · Register Pressure — Occupancy Model, Spill Detection & --maxrregcount": {
        "description": (
            "Build the complete GPU occupancy model as a function of register "
            "count per thread. Show how register usage limits resident warps on "
            "the SM. Compute the occupancy for A100 and H100 at different register "
            "counts. Show the spill threshold and its performance impact. Model "
            "the tradeoff between ILP (more registers = more live values) and "
            "occupancy. Calculate the optimal register count for bandwidth-bound "
            "vs compute-bound kernels."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math

print("=" * 68)
print("  REGISTER PRESSURE — Occupancy Model, Spill Detection & --maxrregcount")
print("=" * 68)
print()

# Hardware constants
GPU_CONFIGS = {
    "A100 (sm_80)": {
        "regs_per_sm":       65536,   # 256 KB / 4 bytes
        "max_warps_per_sm":  64,
        "max_threads_per_sm":2048,
        "smem_per_sm_kb":    192,     # configurable up to 192 KB
        "max_blocks_per_sm": 32,
    },
    "H100 (sm_90)": {
        "regs_per_sm":       65536,   # same register file
        "max_warps_per_sm":  64,
        "max_threads_per_sm":2048,
        "smem_per_sm_kb":    228,
        "max_blocks_per_sm": 32,
    },
    "V100 (sm_70)": {
        "regs_per_sm":       65536,
        "max_warps_per_sm":  64,
        "max_threads_per_sm":2048,
        "smem_per_sm_kb":    96,
        "max_blocks_per_sm": 32,
    },
}


def compute_occupancy(gpu_name, regs_per_thread, block_size=256, smem_bytes=0):
    """
    Compute SM occupancy based on register usage.
    Returns: (active_warps, max_warps, occupancy_pct, limiting_factor)
    """
    g = GPU_CONFIGS[gpu_name]
    warps_per_block = block_size // 32

    # Register constraint: how many blocks fit?
    regs_per_block = regs_per_thread * block_size
    # Round up regs_per_block to 256 (ptxas allocates in 256-register granularity)
    regs_per_block = math.ceil(regs_per_block / 256) * 256
    blocks_by_regs = g["regs_per_sm"] // regs_per_block if regs_per_block > 0 else 999

    # SMEM constraint
    smem_per_sm = g["smem_per_sm_kb"] * 1024
    blocks_by_smem = smem_per_sm // smem_bytes if smem_bytes > 0 else 999

    # Block count constraint
    blocks_by_count = g["max_blocks_per_sm"]

    # Active blocks = minimum of all constraints
    active_blocks = min(blocks_by_regs, blocks_by_smem, blocks_by_count,
                        g["max_threads_per_sm"] // block_size)

    active_warps = active_blocks * warps_per_block
    max_warps    = g["max_warps_per_sm"]
    occupancy    = active_warps / max_warps * 100

    # Determine limiting factor
    limiting = "register"
    if blocks_by_smem < blocks_by_regs and smem_bytes > 0:
        limiting = "smem"
    if g["max_blocks_per_sm"] <= min(blocks_by_regs, blocks_by_smem):
        limiting = "block_count"

    return active_warps, max_warps, occupancy, limiting, active_blocks


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: Occupancy table vs register count
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — Occupancy vs Register Count (block_size=256, no SMEM)")
print("━" * 68)
print()

print(f"  {'Regs/thread':>12}  {'A100 warps':>12}  {'A100 occ':>10}  "
      f"{'H100 warps':>12}  {'H100 occ':>10}  {'Limiting'}")
print("  " + "─" * 62)

for regs in [16, 24, 32, 40, 48, 56, 64, 80, 96, 128, 160, 192, 255]:
    wa, mwa, oa, la, _ = compute_occupancy("A100 (sm_80)", regs)
    wh, mwh, oh, lh, _ = compute_occupancy("H100 (sm_90)", regs)
    print(f"  {regs:>12}  {wa:>8}/{mwa:<3}  {oa:>9.0f}%  "
          f"{wh:>8}/{mwh:<3}  {oh:>9.0f}%  {la}")

print()
print("  32 regs: 100% occupancy on A100 (64 warps, 2048 threads).")
print("  64 regs: 50% occupancy (32 warps) — still good for latency hiding.")
print("  128 regs: 25% occupancy — performance likely limited below 256-wide kernels.")


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: Performance model — occupancy vs throughput
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 2 — Occupancy to Throughput: When Does It Matter?")
print("━" * 68)
print()

H100_HBM_BW_GBS = 3350.0
H100_TC_TFLOPS  = 989.0
H100_WARPS_PER_SM = 64
MEM_LATENCY_CYCS  = 200   # cycles for HBM latency
ISSUE_RATE        = 4.0   # instructions issued per cycle (when active)

# Minimum warps to hide memory latency
# Assumes each warp issues 1 instruction every 4 cycles
# To keep the SM busy: need enough warps to fill the 200-cycle latency
min_warps_for_hiding = math.ceil(MEM_LATENCY_CYCS / ISSUE_RATE)

print(f"  HBM latency: {MEM_LATENCY_CYCS} cycles. Issue rate: {ISSUE_RATE} instr/cycle/warp.")
print(f"  Minimum warps for full latency hiding: {min_warps_for_hiding}")
print()

print(f"  {'Occupancy':>12}  {'Warps':>8}  {'Lat. hiding':>14}  "
      f"{'BW-bound perf%':>16}  {'Compute-bound perf%'}")
print("  " + "─" * 68)

for occ_pct in [6, 12, 25, 37, 50, 62, 75, 87, 100]:
    warps  = int(H100_WARPS_PER_SM * occ_pct / 100)
    # For memory-bound kernels: throughput scales with occupancy up to hiding limit
    hiding_ratio = min(warps / min_warps_for_hiding, 1.0)
    bw_perf_pct  = hiding_ratio * 100
    # For compute-bound kernels: tensor cores utilisation scales with block-level
    # occupancy. Diminishing returns past ~50% occupancy.
    cmp_perf_pct  = min(100.0, math.log(warps + 1) / math.log(H100_WARPS_PER_SM + 1) * 115)
    cmp_perf_pct  = min(cmp_perf_pct, 100.0)
    hiding_str    = "✅ full" if hiding_ratio >= 1.0 else f"⚠ {hiding_ratio:.0%}"
    print(f"  {occ_pct:>11}%  {warps:>8}  {hiding_str:>14}  "
          f"{bw_perf_pct:>15.0f}%  {cmp_perf_pct:>18.0f}%")

print()
print(f"  Memory-bound kernels need ≥{min_warps_for_hiding} warps for full latency hiding.")
print("  Compute-bound kernels (matmul): less sensitive to occupancy (TC keeps unit busy).")


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: Spill cost model
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 3 — Register Spill: Detection & Cost Model")
print("━" * 68)
print()

print("  Register spill = compiler writes live register values to local memory")
print("  (per-thread HBM allocation) when register count exceeds the hardware limit.")
print()
print("  DETECTION COMMANDS:")
print("    nvcc --ptxas-options=-v kernel.cu  → look for '.lmem = N' (N > 0 = spill)")
print("    cuobjdump --dump-sass kernel.cubin → look for LDL/STL instructions")
print("    ncu --metric l1tex__data_pipe_lsu_wavefronts_mem_local.sum  → local mem wfronts")
print()

spill_scenarios = [
    (0,    "No spill",    100.0, "Optimal — all live vars in registers"),
    (64,   "Light spill", 80.0,  "1-2 vars spilled; still mostly in regs"),
    (256,  "Mod. spill",  60.0,  "Several temps spilled per inner loop iter"),
    (1024, "Heavy spill", 35.0,  "Hot path variable spilled; poor performance"),
    (4096, "Severe spill",15.0,  "Kernel has >3× too many live vars; rewrite needed"),
]

print(f"  {'lmem/thread (B)':>18}  {'Rel. throughput':>17}  {'Diagnosis'}")
print("  " + "─" * 60)
for lmem, label, perf_pct, diag in spill_scenarios:
    bar = "█" * int(perf_pct / 5)
    print(f"  {lmem:>18}  {bar:<20} {perf_pct:>4.0f}%  {label}: {diag}")

print()
print("  SPILL COST FORMULA:")
print("    Each spilled load/store: 200-cycle latency (same as global memory).")
print("    If inner loop spills N variables per iteration:")
print("      Extra latency = N × 200 cycles (can't be hidden if loop-carried dependency).")
print("    FIX: --maxrregcount=64, or restructure kernel to reduce live variables.")


# ─────────────────────────────────────────────────────────────────────
# SECTION 4: --maxrregcount tradeoff analysis
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 4 — --maxrregcount Tradeoff: Spill vs Occupancy")
print("━" * 68)
print()

# Simulate a kernel that naturally uses 80 registers
NATURAL_REGS = 80
BLOCK_SIZE   = 256

print(f"  Kernel naturally uses {NATURAL_REGS} registers.")
print(f"  Block size: {BLOCK_SIZE} threads.")
print()
print(f"  {'--maxrregcount':>16}  {'Regs used':>10}  {'lmem (B)':>10}  "
      f"{'Occupancy':>11}  {'Estimated perf%':>16}")
print("  " + "─" * 68)

# Model: reducing regs below natural usage causes proportional spill
for max_regs in [255, 128, 96, 80, 64, 56, 48, 40, 32]:
    actual_regs = min(NATURAL_REGS, max_regs)
    spilled_regs = max(0, NATURAL_REGS - max_regs)
    # Each spilled register → roughly 4 bytes of lmem per thread
    lmem = spilled_regs * 4

    # Occupancy improvement from fewer registers
    _, _, occ, _, _ = compute_occupancy("A100 (sm_80)", actual_regs, BLOCK_SIZE)

    # Performance model:
    # +occupancy benefit (latency hiding) vs -spill cost (extra HBM traffic)
    hiding_ratio  = min(int(H100_WARPS_PER_SM * occ / 100) / min_warps_for_hiding, 1.0)
    spill_penalty = max(0, 1 - spilled_regs * 0.02)   # ~2% perf per spilled reg
    perf_pct = hiding_ratio * spill_penalty * 100

    limit_str = "✅ no spill" if spilled_regs == 0 else f"⚠ spill {spilled_regs}r"
    print(f"  {max_regs:>16}  {actual_regs:>10}  {lmem:>10}  "
          f"{occ:>10.0f}%  {perf_pct:>13.0f}%  {limit_str}")

print()
print("  Sweet spot: --maxrregcount=64 (50% occ, no spill on most kernels).")
print("  Danger zone: --maxrregcount=32 forces spill → performance collapses.")
print("  Rule: profile with ncu before tuning --maxrregcount.")
''',
    },

    "4 · Instruction Scheduling & ILP — Pipeline Analysis & Dual Issue": {
        "description": (
            "Model the SASS instruction pipeline on A100: compute unit latencies, "
            "the scoreboard, and dual-issue opportunities. Simulate instruction "
            "scheduling for a sequence of dependent and independent FMA operations. "
            "Show how the compiler schedules loads ahead of compute to hide latency. "
            "Compute IPC (instructions per cycle) for different instruction mixes. "
            "Show the effect of dependency chains on pipeline utilisation."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math
from collections import defaultdict

print("=" * 68)
print("  INSTRUCTION SCHEDULING & ILP — Pipeline Analysis & Dual Issue")
print("=" * 68)
print()

# A100 pipeline units and their latencies (cycles)
PIPELINE_UNITS = {
    "fma":    {"throughput": 1, "latency": 4,   "desc": "FP32/FP16 FMA"},
    "fadd":   {"throughput": 1, "latency": 4,   "desc": "FP32 add/cmp"},
    "ialu":   {"throughput": 1, "latency": 4,   "desc": "INT32 add/multiply"},
    "load":   {"throughput": 1, "latency": 200, "desc": "Global memory load"},
    "store":  {"throughput": 1, "latency": 1,   "desc": "Global memory store (no dep)"},
    "smem_l": {"throughput": 1, "latency": 20,  "desc": "Shared memory load"},
    "smem_s": {"throughput": 1, "latency": 1,   "desc": "Shared memory store"},
    "sfu":    {"throughput": 4, "latency": 8,   "desc": "SFU: rcp, sqrt (1/4 throughput)"},
    "tc":     {"throughput": 1, "latency": 16,  "desc": "Tensor core HMMA"},
    "shfl":   {"throughput": 1, "latency": 4,   "desc": "Warp shuffle"},
    "branch": {"throughput": 1, "latency": 16,  "desc": "Branch (misprediction: 16cy)"},
}

class InstructionScheduler:
    """
    Simulate in-order instruction scheduling for a single warp.
    Tracks register readiness times and stall cycles.
    """
    def __init__(self):
        self.cycle       = 0
        self.reg_ready   = {}   # reg_name → cycle when its value is ready
        self.issue_log   = []

    def _ready_cycle(self, regs_in):
        """Earliest cycle we can read inputs."""
        return max((self.reg_ready.get(r, 0) for r in regs_in), default=0)

    def issue(self, opcode, regs_out, regs_in, unit="fma"):
        info = PIPELINE_UNITS.get(unit, {"latency": 4, "throughput": 1})

        # Stall until all inputs are ready
        earliest   = self._ready_cycle(regs_in)
        issue_cycle = max(self.cycle, earliest)
        stall_cycs  = issue_cycle - self.cycle

        # Mark outputs as ready after latency
        for r in regs_out:
            self.reg_ready[r] = issue_cycle + info["latency"]

        self.issue_log.append({
            "opcode":      opcode,
            "issue_cycle": issue_cycle,
            "stall":       stall_cycs,
            "latency":     info["latency"],
            "unit":        unit,
            "regs_in":     regs_in,
            "regs_out":    regs_out,
        })
        self.cycle = issue_cycle + 1   # issue takes 1 cycle slot
        return issue_cycle

    def report(self):
        total_stalls = sum(e["stall"] for e in self.issue_log)
        total_cycles = (self.issue_log[-1]["issue_cycle"] + 1
                        if self.issue_log else 0)
        n_instrs     = len(self.issue_log)
        ipc          = n_instrs / total_cycles if total_cycles > 0 else 0
        return total_stalls, total_cycles, ipc

    def print_schedule(self, title):
        print(f"  {title}:")
        print(f"  {'#':>4}  {'Opcode':<18}  {'Issue@':>7}  "
              f"{'Stall':>7}  {'Lat':>5}  {'Inputs→Outputs'}")
        print("  " + "─" * 64)
        for i, e in enumerate(self.issue_log):
            inputs_ready = self._ready_cycle(e["regs_in"]) if i > 0 else 0
            ins_str  = ",".join(e["regs_in"])
            outs_str = ",".join(e["regs_out"])
            stall_str = f"!{e['stall']}" if e["stall"] > 0 else "  —"
            print(f"  {i+1:>4}  {e['opcode']:<18}  {e['issue_cycle']:>7}  "
                  f"{stall_str:>7}  {e['latency']:>5}  {ins_str} → {outs_str}")

        stalls, cycles, ipc = self.report()
        print(f"  Total: {len(self.issue_log)} instrs, {cycles} cycles, "
              f"{stalls} stall cycles, IPC = {ipc:.2f}")
        print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: Dependent chain vs independent chain
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — Dependency Chain: FMA Throughput vs Latency")
print("━" * 68)
print()

# DEPENDENT CHAIN: R0 = R0*R1+R2; R0 = R0*R3+R4; ...
# Each FMA depends on the previous output → stalls
dep = InstructionScheduler()
for i in range(6):
    dep.issue(f"FFMA R0←{i+1}", ["R0"], ["R0", f"R{i+1}", f"R{i+2}"], "fma")
dep.print_schedule("DEPENDENT chain (R0 = R0 * Ri + Ri+1, six iterations)")

# INDEPENDENT CHAIN: R0=..; R4=..; R8=..; each uses different output regs
ind = InstructionScheduler()
for i in range(6):
    ri = i * 4
    ind.issue(f"FFMA R{ri}", [f"R{ri}"], [f"R{ri+1}", f"R{ri+2}", f"R{ri+3}"], "fma")
ind.print_schedule("INDEPENDENT chain (each FFMA writes different register)")

# ─────────────────────────────────────────────────────────────────────
# SECTION 2: Load-compute overlap (software pipeline)
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 2 — Load-Compute Overlap: Hiding 200-Cycle Memory Latency")
print("━" * 68)
print()

# WITHOUT OVERLAP: issue load, wait, compute
no_ol = InstructionScheduler()
no_ol.issue("LDG.E  R4,[ptr]",  ["R4"], ["ptr"], "load")
no_ol.issue("FFMA R8,R4,R4,R0", ["R8"], ["R4", "R0"], "fma")   # stalls 199 cycles!
no_ol.issue("LDG.E  R4,[ptr2]", ["R4"], ["ptr2"], "load")
no_ol.issue("FFMA R9,R4,R4,R1", ["R9"], ["R4", "R1"], "fma")
no_ol.print_schedule("WITHOUT overlap (load then immediate use)")

# WITH OVERLAP: issue load early, compute independent work, then use
ol = InstructionScheduler()
ol.issue("LDG.E  R4,[ptr]",   ["R4"],  ["ptr"],     "load")    # issue load early
ol.issue("FFMA R8,R0,R1,R2",  ["R8"],  ["R0","R1","R2"], "fma") # independent: no stall
ol.issue("FFMA R9,R3,R5,R6",  ["R9"],  ["R3","R5","R6"], "fma") # independent
ol.issue("FFMA R10,R7,R11,R12",["R10"],["R7","R11","R12"],"fma")# independent
ol.issue("LDG.E  R4b,[ptr2]", ["R4b"], ["ptr2"],    "load")    # prefetch next
# After many independent FMAs, DEPBAR would fire — now use R4
ol.issue("FFMA R13,R4,R4,R8", ["R13"], ["R4","R8"], "fma")     # R4 should be ready
ol.print_schedule("WITH overlap (load early, compute independent work)")

# ─────────────────────────────────────────────────────────────────────
# SECTION 3: Mixed instruction IPC analysis
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 3 — IPC Analysis for Different Instruction Mixes")
print("━" * 68)
print()

print("  Theoretical peak IPC on A100: 1.0 (single issue) to 4.0+ (with dual issue).")
print("  Practical IPC depends on: dependency chains, instruction mix, latency hiding.")
print()

mixes = [
    ("Dense FMA (compute-bound)",
     [("FFMA R0,R1,R2,R3","fma","R1 R2 R3","R0"),
      ("FFMA R4,R5,R6,R7","fma","R5 R6 R7","R4"),
      ("FFMA R8,R9,R10,R11","fma","R9 R10 R11","R8"),
      ("FFMA R12,R13,R14,R15","fma","R13 R14 R15","R12"),]),
    ("Load then FMA (memory-compute interleaved)",
     [("LDG.E R0,[R1]","load","R1","R0"),
      ("FFMA R4,R5,R6,R7","fma","R5 R6 R7","R4"),   # independent
      ("FFMA R8,R9,R10,R11","fma","R9 R10 R11","R8"), # independent
      ("FFMA R0b,R0,R0,R4","fma","R0 R4","R0b"),]),  # uses loaded R0
    ("All loads (pure memory)",
     [("LDG.E R0,[R1]","load","R1","R0"),
      ("LDG.E R2,[R3]","load","R3","R2"),
      ("LDG.E R4,[R5]","load","R5","R4"),
      ("LDG.E R6,[R7]","load","R7","R6"),]),
]

for mix_name, instrs in mixes:
    sched = InstructionScheduler()
    for opcode, unit, ins_str, outs_str in instrs:
        ins  = ins_str.split()
        outs = outs_str.split()
        sched.issue(opcode, outs, ins, unit)
    stalls, cycles, ipc = sched.report()
    print(f"  {mix_name}:")
    print(f"    {len(instrs)} instructions, {cycles} cycles, "
          f"{stalls} stall cycles, IPC = {ipc:.2f}")
    print()

# ─────────────────────────────────────────────────────────────────────
# SECTION 4: Warp switching to hide latency
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 4 — Warp Switching: How Multiple Warps Hide Memory Latency")
print("━" * 68)
print()

MEM_LATENCY    = 200   # cycles
FMA_THROUGHPUT = 1     # cycle per FMA
N_WARPS_OPTIONS = [1, 2, 4, 8, 16, 32, 48, 64]

print("  Each warp: LDG (200cy latency) → 4 FMAs → repeat.")
print("  SM schedules ready warps while others wait for loads.")
print()
print(f"  {'N warps':>8}  {'SM busy%':>10}  {'Effective IPC':>15}  "
      f"{'Load hidden?':>14}  {'Throughput vs peak'}")
print("  " + "─" * 60)

# Model: with N warps, SM can issue from another warp during load latency
# Warp issues 1 load + 4 FMAs per iteration (5 instructions, ~204 cycles)
# With N warps: SM stays busy if N * 4 FMAs >= 200 cycles of load latency
fma_per_iteration = 4
instrs_per_iter   = 1 + fma_per_iteration

for n_warps in N_WARPS_OPTIONS:
    total_work_cycles = fma_per_iteration  # FMAs per warp per iteration
    # With N warps round-robining: effective SM cycles per iteration
    # = max(load_latency, N * fma_cycles_per_warp_per_iter)
    # When N warps cover the load latency:
    effective_busy = min(1.0, n_warps * total_work_cycles / MEM_LATENCY)
    ipc = effective_busy * instrs_per_iter / instrs_per_iter   # norm to 1.0

    hidden = "✅ fully" if n_warps * fma_per_iteration >= MEM_LATENCY else "❌ partial"
    tput_pct = effective_busy * 100
    print(f"  {n_warps:>8}  {tput_pct:>9.0f}%  {effective_busy:>15.2f}  "
          f"{hidden:>14}  {tput_pct:>10.0f}%")

print()
min_warps_needed = math.ceil(MEM_LATENCY / fma_per_iteration)
print(f"  Minimum warps to fully hide {MEM_LATENCY}-cycle load: {min_warps_needed} warps.")
print(f"  = ceil({MEM_LATENCY} latency / {fma_per_iteration} FMAs per iteration).")
print(f"  This matches the register occupancy calculation: at 50 warps or more,")
print(f"  the SM never idles waiting for memory.")
''',
    },

    "5 · Inline PTX Patterns — asm volatile, Special Registers & Vectorised Loads": {
        "description": (
            "Implement and test common inline PTX patterns: reading the SM clock "
            "counter, loading the lane ID, vectorised 128-bit loads, non-coherent "
            "cache loads, atomic operations. Show the constraint syntax for each. "
            "Demonstrate a performance-critical kernel loop that requires inline PTX "
            "for precise control. Simulate the output of --ptxas-options=-v for "
            "different kernels and show how to interpret register and lmem counts."
        ),
        "language": "python",
        "code": '''
import numpy as np
import math

print("=" * 68)
print("  INLINE PTX PATTERNS — asm volatile, Special Registers & Vectorised Loads")
print("=" * 68)
print()

np.random.seed(42)


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: Common inline PTX patterns with annotations
# ─────────────────────────────────────────────────────────────────────
print("━" * 68)
print("  SECTION 1 — Inline PTX Patterns: Code + Constraint + Use Case")
print("━" * 68)
print()

patterns = [
    {
        "name": "Read SM clock (timing)",
        "code": 'uint64_t t;\\nasm volatile("mov.u64 %0, %%clock64;" : "=l"(t));',
        "constraint": '"=l"(t): output to 64-bit int t',
        "use": "GPU-side micro-benchmarking; profiling inner loop cycles",
        "note": "clock64 ticks at SM frequency (~1.41 GHz on A100)",
    },
    {
        "name": "Read warp lane ID",
        "code": 'uint32_t lane;\\nasm volatile("mov.u32 %0, %%laneid;" : "=r"(lane));',
        "constraint": '"=r"(lane): output to 32-bit register',
        "use": "Custom warp-level reductions without __lane_id() overhead",
        "note": "Returns 0-31. Alternative: __builtin_ia32_rdtsc() does NOT work on GPU",
    },
    {
        "name": "Vectorised 128-bit load",
        "code": 'float4 v;\\nasm volatile("ld.global.v4.f32 {%0,%1,%2,%3}, [%4];"\\n  : "=f"(v.x),"=f"(v.y),"=f"(v.z),"=f"(v.w)\\n  : "l"(ptr));',
        "constraint": '"=f" × 4: four 32-bit float outputs; "l": 64-bit address',
        "use": "Load 4 floats in one 128-bit transaction = optimal coalescing",
        "note": "ptr must be 16-byte aligned. Equivalent to: *((float4*)ptr)",
    },
    {
        "name": "Non-coherent (NC) load",
        "code": 'float val;\\nasm volatile("ld.global.nc.f32 %0, [%1];"\\n  : "=f"(val) : "l"(ptr));',
        "constraint": '"=f"(val): float output; "l"(ptr): 64-bit ptr',
        "use": "Read-only data accessed by all threads uniformly (broadcasts well)",
        "note": "Uses texture L1 cache — higher effective BW for uniform access",
    },
    {
        "name": "Streaming store (L2 bypass)",
        "code": 'asm volatile("st.global.cs.f32 [%0], %1;"\\n  : : "l"(ptr), "f"(val));',
        "constraint": '"l": output ptr; "f": float value to store',
        "use": "Write-once output that should not pollute L1/L2 caches",
        "note": ".cs = cache streaming: evict-first policy in L2",
    },
    {
        "name": "Warp vote (any/all)",
        "code": 'uint32_t mask;\\nasm volatile("vote.any.b32 %0, %0, 0xffffffff;"\\n  : "+r"(mask));',
        "constraint": '"+r": read-write 32-bit register (input and output)',
        "use": "Check if any thread in warp has a condition true",
        "note": "Equivalent to __ballot_sync but with explicit mask control",
    },
    {
        "name": "Atomic add with return",
        "code": 'float old;\\nasm volatile("atom.global.add.f32 %0, [%1], %2;"\\n  : "=f"(old) : "l"(ptr), "f"(val));',
        "constraint": '"=f": old value returned; "l"+"f": address and value',
        "use": "Float atomic add — equivalent to atomicAdd but with explicit PTX",
        "note": "Use red.global.add.f32 if old value not needed (faster)",
    },
    {
        "name": "Predicated store",
        "code": 'asm volatile(\\n  "{ .reg .pred %p;\\\n"\\n  "  setp.ne.s32 %p, %1, 0;\\\n"\\n  "  @%p st.global.f32 [%2], %3; }"\\n  : : "r"(cond), "l"(ptr), "f"(val));',
        "constraint": '"r": int32 condition; "l": ptr; "f": value',
        "use": "Store only if condition is true — avoids branch divergence",
        "note": "The {} block is a PTX compound statement with local .reg",
    },
]

for pat in patterns:
    print(f"  ─── {pat['name']} ───")
    print(f"  Code:")
    for line in pat["code"].split("\\n"):
        print(f"    {line}")
    print(f"  Constraints: {pat['constraint']}")
    print(f"  Use case:    {pat['use']}")
    print(f"  Note:        {pat['note']}")
    print()


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: --ptxas-options=-v output interpretation
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 2 — ptxas -v Output: Register & lmem Analysis")
print("━" * 68)
print()

print("  nvcc --ptxas-options=-v kernel.cu compiles and prints per-kernel stats.")
print()

ptxas_outputs = [
    {
        "kernel":  "_Z10add_kernelPfS_S_i",
        "regs":    24,  "lmem": 0,    "smem": 0,      "cmem": 64,
        "verdict": "✅ Excellent. Low reg count, no spill.",
        "occ":     "100% on A100 (64 warps)",
    },
    {
        "kernel":  "_Z15matmul_kernel_v1PfS_S_ii",
        "regs":    64,  "lmem": 0,    "smem": 32768,  "cmem": 64,
        "verdict": "✅ Good. High reg for ILP, SMEM for tiles, no spill.",
        "occ":     "50% on A100 (32 warps) — acceptable for compute-bound",
    },
    {
        "kernel":  "_Z12reduce_badlyPfS_i",
        "regs":    96,  "lmem": 128,  "smem": 1024,   "cmem": 64,
        "verdict": "⚠ WARNING: lmem=128 bytes → REGISTER SPILL detected!",
        "occ":     "25% on A100 (16 warps) — AND spilling",
    },
    {
        "kernel":  "_Z16attention_forwardPfS_S_ii",
        "regs":    128, "lmem": 512,  "smem": 65536,  "cmem": 128,
        "verdict": "❌ CRITICAL: 512B lmem spill. Consider splitting into smaller kernels.",
        "occ":     "12% on A100 (8 warps) — very poor",
    },
    {
        "kernel":  "_Z16softmax_triton_v2PfS_i",
        "regs":    48,  "lmem": 0,    "smem": 4096,   "cmem": 64,
        "verdict": "✅ Good. Triton-style, no spill, moderate reg.",
        "occ":     "75% on A100 (48 warps)",
    },
]

print(f"  {'Kernel':<42}  {'Regs':>5}  {'lmem':>6}  "
      f"{'smem':>7}  {'cmem':>5}")
print("  " + "─" * 70)
for p in ptxas_outputs:
    name_short = p["kernel"][-40:]
    print(f"  {name_short:<42}  {p['regs']:>5}  {p['lmem']:>6}  "
          f"{p['smem']:>7}  {p['cmem']:>5}")

print()
print("  DETAILED ANALYSIS:")
for p in ptxas_outputs:
    print(f"  {p['kernel']}:")
    print(f"    .reg={p['regs']}, .lmem={p['lmem']}, .smem={p['smem']}, .cmem={p['cmem']}")
    print(f"    {p['verdict']}")
    print(f"    Occupancy: {p['occ']}")


# ─────────────────────────────────────────────────────────────────────
# SECTION 3: Vectorised load correctness simulation
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 3 — Vectorised Load: LDG.E.128 vs LDG.E.32 Behaviour")
print("━" * 68)
print()

print("  LDG.E.32: load 4 bytes. Thread 0 loads A[0], thread 1 loads A[1], ...")
print("  LDG.E.128: load 16 bytes. Thread 0 loads A[0:4], thread 1 loads A[4:8], ...")
print()

# Simulate both access patterns
N_threads = 4  # small example
data = np.arange(16, dtype=np.float32) * 1.5  # 16 float values

print("  Source data: A = [0.0, 1.5, 3.0, 4.5, 6.0, 7.5, 9.0, ...]")
print()
print("  LDG.E.32 (scalar load, 1 float per thread):")
for tid in range(N_threads):
    val = data[tid]
    print(f"    Thread {tid}: loads A[{tid}] = {val:.1f}  (4 bytes)")

print()
print("  LDG.E.128 (vectorised load, 4 floats per thread):")
for tid in range(N_threads):
    vals = data[tid*4:(tid+1)*4]
    print(f"    Thread {tid}: loads A[{tid*4}:{tid*4+4}] = {vals.tolist()}  (16 bytes)")

print()
print("  Bandwidth comparison (32 threads × 128 floats/step):")
print(f"    LDG.E.32:  32 threads × 4B  = 128 bytes per instruction")
print(f"    LDG.E.128: 32 threads × 16B = 512 bytes per instruction  (4× more!)")
print(f"    512 bytes = exactly one L2 cache line → perfect coalescing ✅")
print()

# Show when 128-bit loads are possible
print("  REQUIREMENTS for LDG.E.128:")
requirements = [
    ("128-bit alignment", "Base pointer must be 16-byte aligned (ptr % 16 == 0)"),
    ("Contiguous access", "Thread i accesses address: base + i * 16 (stride of 4 floats)"),
    ("4 consecutive elements", "Kernel processes 4 floats per thread (not 1)"),
    ("float4 data type",  "CUDA: float4* ptr; val = *ptr; or struct with 4 floats"),
]
for req, desc in requirements:
    print(f"    {req:<28}: {desc}")


# ─────────────────────────────────────────────────────────────────────
# SECTION 4: Inline PTX for GPU benchmarking (clock64)
# ─────────────────────────────────────────────────────────────────────
print()
print("━" * 68)
print("  SECTION 4 — GPU Clock Counter: Timing Inner Loops with PTX")
print("━" * 68)
print()

print("  Use case: measure exact cycle count for a kernel region.")
print()
print("  CORRECT PATTERN:")
print()
print("  __global__ void benchmark_kernel(float* out, uint64_t* timer) {")
print("      // Start timer")
print("      uint64_t t_start;")
print('      asm volatile("mov.u64 %0, %%clock64;" : "=l"(t_start));')
print()
print("      // Work to benchmark")
print("      float acc = 0.0f;")
print("      for (int i = 0; i < 1024; i++) {")
print("          acc = fmaf(acc, acc, 1.0f);  // FMA loop")
print("      }")
print()
print("      // Stop timer")
print("      uint64_t t_stop;")
print('      asm volatile("mov.u64 %0, %%clock64;" : "=l"(t_stop));')
print()
print("      // Store result to prevent dead-code elimination")
print("      *out   = acc;")
print("      *timer = t_stop - t_start;   // cycles elapsed")
print("  }")
print()
print("  INTERPRETING RESULTS:")

A100_SM_FREQ_GHZ = 1.41
cycles_per_fma    = 1   # 1024 FMAs / (1024 / 4 stages) assuming ILP=4

sample_measurements = [
    ("1024 FMAs (4 independent chains)", 260, 1024),
    ("1024 FMAs (1 dependent chain)",    4100, 1024),
    ("1024 global loads",                204800, 1024),
    ("1024 SMEM loads",                  20480,  1024),
]

print(f"  {'Operation':<40}  {'Cycles':>10}  {'Instr':>7}  "
      f"{'IPC':>6}  {'Time (µs)'}")
print("  " + "─" * 68)
for name, cycs, n_instr in sample_measurements:
    ipc   = n_instr / cycs
    t_us  = cycs / (A100_SM_FREQ_GHZ * 1e3)  # µs
    print(f"  {name:<40}  {cycs:>10,}  {n_instr:>7}  "
          f"{ipc:>6.3f}  {t_us:>9.3f}")

print()
print(f"  A100 SM runs at {A100_SM_FREQ_GHZ} GHz.")
print(f"  4 independent FMA chains: IPC ≈ 4 (near peak for FP32 FMA unit).")
print(f"  1 dependent FMA chain: IPC ≈ 0.25 (every instruction stalls 4 cycles).")
print(f"  Global loads: 200 cycles latency each → IPC = 1/200 = 0.005 — terrible!")
print(f"  SMEM loads: 20 cycles → IPC = 1/20 = 0.05 — still poor for isolated loads.")
''',
    },
}

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