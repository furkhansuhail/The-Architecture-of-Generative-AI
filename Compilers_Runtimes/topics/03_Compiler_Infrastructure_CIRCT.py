"""
CIRCT — Circuit IR Compilers and Tools
========================================

CIRCT (Circuit IR Compilers and Tools) is a hardware compiler infrastructure
project that extends MLIR into the domain of digital hardware design.
Where MLIR bridges ML frameworks down to LLVM IR, CIRCT bridges hardware
description languages (Verilog, FIRRTL, SystemC) down to gate-level netlists
and silicon.

The problem CIRCT solves: hardware tooling is decades behind software compilers.
Traditional EDA (Electronic Design Automation) tools are monolithic, proprietary,
and built on 1980s-era data formats. There is no LLVM for hardware — no shared,
extensible IR infrastructure that hardware teams can build on and contribute to.

CIRCT's insight: hardware compilation is just another multi-level lowering problem.
The same MLIR infrastructure that lowers tf.Conv2D → linalg → affine → LLVM IR
can lower a hardware module description → RTL → gate-level netlist → silicon.

In the connected compiler stack:
    LLVM  (module 1)    ← CIRCT uses LLVM for software-side code generation
    MLIR  (module 2)    ← CIRCT IS an MLIR extension; reuses all infrastructure
    CIRCT (this module) ← HW-oriented MLIR dialects, FIRRTL, Verilog emission
    XLA   (module 4)    ← XLA accelerator targets (TPUs) are compiled via CIRCT-like paths
    TVM   (module 5)    ← TVM FPGA/custom accelerator backends use RTL generation

CIRCT's role in AI/ML specifically:
    Custom AI accelerators (TPUs, NPUs, FPGAs) require hardware design tooling.
    ML frameworks (XLA, TVM) generate compute graphs; those graphs must eventually
    become silicon or FPGA bitstreams. CIRCT provides the compiler infrastructure
    for that final step: taking a high-level hardware description and lowering it
    all the way to Verilog, which EDA synthesis tools consume.

"""

import textwrap
import re

TOPIC_NAME   = "CIRCT — Circuit IR Compilers and Tools"
DISPLAY_NAME = "03 · CIRCT"
ICON         = "⚙️"
SUBTITLE     = "HW-Oriented MLIR Dialects, FIRRTL, and Silicon-Level Lowering"


# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """

##### PART 1 — WHY CIRCT EXISTS: THE HARDWARE TOOLING CRISIS

### The State of Hardware Design Tooling (Pre-CIRCT)

    Hardware design relies on tools built in the 1980s and 1990s:
        Verilog (1984):       the dominant hardware description language
        VHDL (1987):          the formal alternative; verbose, rarely loved
        SystemVerilog (2005): adds OOP features but not compiler-friendliness
        Proprietary EDA:      Synopsys, Cadence, Mentor dominate; tools are
                              $500k+/year licences, black boxes, no APIs

    The fundamental problem: there is no shared IR for hardware.
        Every EDA tool speaks its own internal representation.
        Synopsys DC has one netlist IR.
        Vivado (Xilinx FPGA) has another.
        Verilator (open-source simulator) has yet another.
        None of them are extensible. None share passes. None share analyses.

    Compare to software compilers after LLVM:
        clang, rustc, swiftc, kotlin — ALL lower into LLVM IR.
        Any pass written once (loop-vectorize, instcombine) works for all.
        Hardware has no equivalent. Every EDA vendor reinvents the wheel.

### The AI Accelerator Pressure

    The rise of ML hardware (2015–present) made this crisis acute:
        Google TPU v1 (2016): first custom ML ASIC at scale
        Amazon Inferentia:    AWS custom inference chip
        Cerebras, Graphcore, SambaNova, Tenstorrent: startup ASICs
        NVIDIA GPU → H100:    increasingly programmable, custom data paths
        FPGAs for inference:  Xilinx Alveo, Intel Stratix used for low-latency

    Each of these requires:
        1. A hardware description (what does the chip do?)
        2. RTL generation (write the logic in Verilog/VHDL)
        3. Logic synthesis (convert RTL to gate-level netlist)
        4. Place and route (map gates to physical silicon locations)
        5. Timing verification (does it meet clock constraints?)

    Steps 1–2 are increasingly software-generated (high-level synthesis).
    The compiler infrastructure for those steps was broken.
    CIRCT is the fix.

### CIRCT's Founding Vision (2020)

    CIRCT was started by Google, LLVM Foundation, and academic collaborators.
    Core insight:

        ┌──────────────────────────────────────────────────────────────┐
        │  MLIR already solved the "multi-level IR framework" problem  │
        │  for ML compilers. The SAME infrastructure works for         │
        │  hardware compilers.                                         │
        │                                                              │
        │  A hardware compiler is just another progressive lowering:   │
        │  SystemC / FIRRTL (module level)                             │
        │      ↓  lower to RTL                                         │
        │  HW + Comb + Seq dialects (register-transfer level)          │
        │      ↓  lower to logic                                       │
        │  SV dialect (structural SystemVerilog)                       │
        │      ↓  export                                               │
        │  Verilog/FIRRTL files (EDA tool input)                       │
        │      ↓  synthesis (Yosys / Synopsys DC)                      │
        │  Gate-level netlist → silicon or FPGA bitstream              │
        └──────────────────────────────────────────────────────────────┘

    CIRCT provides the shared IR infrastructure for ALL steps above.
    Hardware teams get the same benefits ML teams got from MLIR:
        - Shared pass manager for hardware transformations
        - Shared pattern rewriting for hardware optimisations
        - Shared type system (hardware bit widths, clock domains)
        - Extensible dialect mechanism for new hardware concepts
        - Open source, composable, no vendor lock-in


##### PART 2 — CIRCT'S DIALECT HIERARCHY

### What CIRCT Dialects Do

    Like MLIR's ML dialects (tosa → linalg → affine → llvm), CIRCT defines
    a hierarchy of hardware-domain dialects that progressively lower from
    high-level circuit descriptions to synthesisable RTL.

    Each dialect represents one level of hardware abstraction:
        High level:   design intent, modules, types, connectivity
        Mid level:    register-transfer logic (combinational + sequential)
        Low level:    structural Verilog/SystemVerilog for EDA tools

### The Core CIRCT Dialects

    ┌──────────────────────────────────────────────────────────────────────┐
    │  FIRRTL dialect  (Flexible Internal Representation for RTL Logic)    │
    │  Source: Chisel/FIRRTL hardware generator ecosystem (Berkeley/SiFive)│
    │  Purpose: high-level typed RTL with width inference and bundles      │
    │                                                                      │
    │  FIRRTL types are richer than Verilog:                               │
    │    UInt<8>           unsigned 8-bit wire                             │
    │    SInt<32>          signed 32-bit wire                              │
    │    Bundle<a: UInt, b: SInt>   struct-like aggregated wire            │
    │    Vector<UInt<8>, 4>         array of 4 bytes                       │
    │    Clock, Reset, AsyncReset   special clock/reset types              │
    │                                                                      │
    │  FIRRTL operations:                                                  │
    │    firrtl.module @Adder(in %a: !firrtl.uint<8>, ...)                 │
    │    firrtl.add %a, %b : (!firrtl.uint<8>, ...) -> !firrtl.uint<9>     │
    │    firrtl.mux %sel, %a, %b   : conditional select                    │
    │    firrtl.reg %clk           : register with clock                   │
    │    firrtl.regReset %clk, %rst, %val  : register with reset           │
    └──────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────────┐
    │  HW dialect  (Hardware Structural IR)                               │
    │  Purpose: module hierarchy, port declarations, instances            │
    │  Level: RTL structural — modules and their connections              │
    │                                                                     │
    │  hw.module @Adder(%a: i8, %b: i8) -> (sum: i9) {                    │
    │    %sum = comb.add %a, %b : i8          ; NOT in HW — comb handles  │
    │    hw.output %sum : i9                                              │
    │  }                                                                  │
    │                                                                     │
    │  hw.instance "add0" @Adder(a: %x, b: %y) -> (sum: i9)               │
    │    ; instantiates module @Adder, connects ports                     │
    └─────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────────┐
    │  Comb dialect  (Combinational Logic)                                │
    │  Purpose: purely combinational (stateless) Boolean logic            │
    │  Key: NO side effects, NO state — purely functional gates           │
    │                                                                     │
    │  comb.add  %a, %b       : i8  → add (can produce carry = i9)        │
    │  comb.sub  %a, %b       : i8  → subtract                            │
    │  comb.mul  %a, %b       : i8  → multiply                            │
    │  comb.and  %a, %b       : i8  → bitwise AND                         │
    │  comb.or   %a, %b       : i8  → bitwise OR                          │
    │  comb.xor  %a, %b       : i8  → bitwise XOR                         │
    │  comb.shl  %a, %amt     : i8  → shift left                          │
    │  comb.shru %a, %amt     : i8  → shift right (unsigned)              │
    │  comb.mux  %sel, %a, %b : i8  → 2:1 multiplexer                     │
    │  comb.concat %hi, %lo   : bit concatenation                         │
    │  comb.extract %val from %lo to %hi  : bit slice                     │
    │  comb.icmp eq %a, %b    : i1  → equality comparator                 │
    └─────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────────┐
    │  Seq dialect  (Sequential Logic)                                    │
    │  Purpose: state elements — registers, memories, clocked logic       │
    │                                                                     │
    │  seq.firreg %next clock %clk : i8                                   │
    │    ; D flip-flop: captures %next on rising edge of %clk             │
    │  seq.firreg %next clock %clk reset sync %rst, %init : i8            │
    │    ; register with synchronous reset to %init                       │
    │  seq.firmem  : memory array (read/write ports, SRAM semantics)      │
    │  seq.hlmem   : high-level memory (before port lowering)             │
    └─────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────────┐
    │  SV dialect  (SystemVerilog)                                        │
    │  Purpose: structural SystemVerilog; direct EDA tool input           │
    │  Level: lowest — maps 1:1 to synthesisable SV constructs            │
    │                                                                     │
    │  sv.always posedge %clk {       ; always @(posedge clk)             │
    │    sv.if %rst {                 ; if (rst)                          │
    │      sv.passign %reg, %zero     ; reg <= 0;                         │
    │    } else {                                                         │
    │      sv.passign %reg, %next     ; reg <= next;                      │
    │    }                                                                │
    │  }                                                                  │
    │  sv.assign %out = %in           ; continuous assign (comb)          │
    │  sv.ifdef "SYNTHESIS" { ... }   ; synthesis guards                  │
    └─────────────────────────────────────────────────────────────────────┘

    Additional important dialects:
        Moore dialect:  SystemVerilog parsing/import (CIRCT's SV frontend)
        Arc dialect:    simulation-optimised arc IR (for fast RTL sim)
        Calyx dialect:  high-level hardware scheduling (from Cornell)
        MSFT dialect:   Microsoft FPGA extensions (place-and-route hints)
        OM dialect:     Object Model — design metadata and parameterisation
        LTL dialect:    Linear Temporal Logic properties (formal verification)
        Verif dialect:  verification constructs (assert, assume, cover)

### The FIRRTL Compiler — CIRCT's Flagship Pipeline

    The FIRRTL compiler is CIRCT's most mature pipeline.
    It compiles Chisel (Scala hardware DSL) → FIRRTL → Verilog.

    Chisel is how Berkeley/SiFive/most RISC-V chips are designed:
        class Adder(n: Int) extends Module {
            val io = IO(new Bundle {
                val a   = Input(UInt(n.W))
                val b   = Input(UInt(n.W))
                val sum = Output(UInt((n+1).W))
            })
            io.sum := io.a + io.b
        }

    Chisel emits FIRRTL. CIRCT compiles FIRRTL → Verilog:

        Step 1: FIRRTL parsing   → FIRRTL dialect in MLIR
        Step 2: Width inference  → resolve all implicit bit widths
        Step 3: Type lowering    → Bundle/Vector → flat wires
        Step 4: Grand Central    → extract interface specifications
        Step 5: Lower to HW/Comb/Seq dialects
        Step 6: SV emission      → exportVerilog → .sv files
        Step 7: EDA tools        → Synopsys / Yosys synthesis


##### PART 3 — PROGRESSIVE LOWERING IN HARDWARE: THE CIRCT STRATEGY

### Hardware Lowering Chain

    Just as MLIR lowers ML ops step by step, CIRCT lowers hardware
    descriptions step by step — each level preserves exactly the
    information needed for the optimisations at that level.

    Level 5: Chisel (Scala DSL)
        class MAC extends Module { ... io.out := io.a * io.b + io.c ... }

    Level 4: FIRRTL (typed RTL, width inference, bundle types)
        firrtl.circuit @MAC {
          firrtl.module @MAC(in %a: !firrtl.uint<8>, ...) {
            %mul = firrtl.mul %a, %b    ; width = 8+8 = 16
            %acc = firrtl.add %mul, %c  ; width = 16+16 = 17
            firrtl.connect %out, %acc
          }
        }
        Passes at this level:
          - Width inference (propagate bit widths from known to unknown)
          - Bundle flattening (Bundle<a,b> → two flat wires a, b)
          - Memory model lowering (firrtl.mem → read/write port modules)
          - Annotation processing (metadata for Grand Central taps)

    Level 3: HW + Comb + Seq dialects (register-transfer level)
        hw.module @MAC(%a: i8, %b: i8, %c: i17, %clk: i1) -> (out: i17) {
          %mul = comb.mul %a, %b   : i16   ; pure combinational
          %acc = comb.add %mul, %c : i17   ; pure combinational
          %reg = seq.firreg %acc clock %clk : i17   ; register
          hw.output %reg : i17
        }
        Passes at this level:
          - CSE (common sub-expression elimination on comb ops)
          - Canonicalisation (comb.add %x, 0 → %x)
          - Register retiming (move regs across comb logic)
          - Const propagation

    Level 2: SV dialect (structural SystemVerilog)
        sv.always posedge %clk {
          sv.passign %reg_q, %acc  ; reg_q <= acc;
        }
        sv.assign %out = %reg_q
        Passes at this level:
          - Clock domain crossing checks
          - SV-specific legalisations (no multi-driven nets)
          - Assertion insertion from LTL properties

    Level 1: Verilog/SystemVerilog text (EDA tool input)
        module MAC(input [7:0] a, b, input [16:0] c, input clk,
                   output reg [16:0] out);
          wire [15:0] mul;
          wire [16:0] acc;
          assign mul = a * b;
          assign acc = mul + c;
          always @(posedge clk) out <= acc;
        endmodule

    Level 0: Gate-level netlist (after EDA synthesis)
        AND2_X1 U1(.A(a[0]), .B(b[0]), .ZN(n1));
        FA_X1 U2(.A(n1), .B(n2), .CI(n3), .S(sum[0]), .CO(n4));
        ... (thousands of cells for even a simple multiply)

### Why Multiple Levels Matter for Hardware

    Level 4 (FIRRTL): Width inference — can only be done before lowering.
        Once Bundle<x: UInt, y: UInt> is flattened to two wires,
        the relationship between x and y is lost.
        Width inference can propagate constraints across the Bundle.

    Level 3 (HW/Comb/Seq): Combinational CSE and canonicalisation.
        The compiler knows comb.add has no side effects (pure).
        Dead comb ops can be eliminated without worrying about clocking.
        Separation of comb and seq dialects makes this SAFE and FORMAL.

    Level 2 (SV): Clock domain analysis.
        The SV dialect knows which signals are clocked on which clock.
        Crossing clock domains without synchronisers → timing violation.
        This check is only meaningful at the SV level.

    Level 1 (Verilog text): EDA tool compatibility.
        Synopsys DC expects synthesisable Verilog.
        exportVerilog respects exactly what each EDA tool accepts.
        It avoids constructs that are legal SV but unsynthesisable.


##### PART 4 — THE CALYX DIALECT: HW SCHEDULING FOR AI ACCELERATORS

### What Calyx Is

    Calyx is a hardware design language developed at Cornell that sits
    ABOVE RTL — it describes hardware in terms of cells (functional
    units) and a control schedule (when does each unit activate?).

    Calyx fills a critical gap for AI accelerators:
        ML frameworks produce dataflow graphs (what to compute).
        RTL describes logic gates and registers (how the silicon works).
        Calyx describes WHAT + WHEN: cells doing work on a schedule.

    This is exactly what an AI accelerator requires:
        A systolic array:  cells = MAC units, schedule = tiled wave-front
        A vector engine:   cells = ALU lanes, schedule = vectorised loop
        A DMA engine:      cells = memory controllers, schedule = prefetch

### Calyx Structure

    A Calyx program has three sections:

    1. CELLS — declare the functional units:
        cells {
            add0 = std_add(32);       ; 32-bit adder
            mul0 = std_mult_pipe(32); ; pipelined 32-bit multiplier
            mem  = std_mem_d1(32, 1024, 10);  ; 1024-element i32 memory
            acc  = std_reg(32);       ; 32-bit register
        }

    2. WIRES — combinational connections between cells:
        wires {
            group do_add {
                add0.left  = acc.out;
                add0.right = mem.read_data;
                acc.in     = add0.out;
                acc.write_en = 1'd1;
                do_add[done] = acc.done;
            }
        }

    3. CONTROL — schedule: when does each group execute?
        control {
            seq {                      ; sequential: load then compute
                invoke mem_load;       ; step 1: read memory
                repeat 8 {            ; step 2: accumulate 8 times
                    do_add;
                }
                invoke output;         ; step 3: write result
            }
        }

### Calyx → CIRCT RTL → Verilog Pipeline

    Calyx compiles via CIRCT:
        Calyx IR → Calyx dialect (CIRCT) → HW/Comb/Seq → SV → Verilog

    This means Calyx hardware designs are MLIR programs.
    They can mix Calyx dialect operations with FIRRTL dialect operations.
    The same MLIR pass manager handles both.

    Relevance for AI/ML:
        XLA can emit Calyx IR for custom accelerator targets.
        TVM has experimental Calyx backend for FPGA deployment.
        Calyx's control model maps directly to ML op scheduling.


##### PART 5 — FORMAL VERIFICATION: LTL AND VERIF DIALECTS

### Why Formal Verification Matters for AI Hardware

    AI accelerators run at the frontier of performance:
        - Clock frequencies of 1–2 GHz with thousands of MAC units
        - Designs are too complex for exhaustive simulation testing
        - A bug in a tapeout costs millions of dollars and 6+ months

    Formal verification uses mathematical proofs to verify hardware:
        "Does this design ALWAYS output the correct sum for any inputs?"
        (not just for the test vectors you happened to try)

### LTL Dialect (Linear Temporal Logic)

    LTL expresses TEMPORAL properties — things that must hold over time:

        ltl.always %property    ; □ P  — true at every clock cycle
        ltl.eventually %prop    ; ◇ P  — true at some future cycle
        ltl.until %cond, %prop  ; P U Q — P holds until Q becomes true
        ltl.next %prop          ; ○ P  — property holds next cycle

    Hardware LTL example:
        ; "Whenever request is asserted, grant follows within 4 cycles"
        %req_implies_grant = ltl.until %req, %grant
        verif.assert %req_implies_grant : !ltl.property

    This is compiled to SVA (SystemVerilog Assertions):
        assert property (@(posedge clk) req |-> ##[1:4] grant);

### Verif Dialect

    The Verif dialect adds verification constructs to the HW/Comb/Seq IR:
        verif.assert %cond : !i1     ; property must be true
        verif.assume %cond : !i1     ; assume this for the solver
        verif.cover  %cond : !i1     ; record when this is reached

    These feed into:
        - Bounded Model Checking (BMC) via CIRCT's bmc tool
        - Formal equivalence checking (compare RTL vs spec)
        - Integration with Yices2, Z3, Bitwuzla SMT solvers

    For AI accelerators, verif properties can express:
        "The output of the integer MAC unit equals the floating-point
         reference model to within 1 ULP for all 32-bit inputs."


##### PART 6 — THE ARC DIALECT: SIMULATION-OPTIMISED IR

### What Arc Is

    RTL simulation (running the hardware design in software to test it)
    is a massive bottleneck in hardware development:
        A complex SoC may have 100M+ gates.
        Simulating one second of clock time at 1 GHz takes days on a workstation.
        ML models running on custom hardware require long simulation runs.

    Verilator (the fastest open-source RTL simulator) compiles Verilog to C++.
    CIRCT's Arc dialect is a simulation-optimised IR that goes further.

### Arc's Model: Functional State Machines

    Arc models hardware as pure functions over state:
        Each clock cycle = one call to a pure function.
        State (registers, memories) is explicit input AND output.
        Combinational logic = pure function composition.

    arc.define @MAC_step(%a: i8, %b: i8, %state: i17) -> (i17) {
        %mul = comb.mul %a, %b   : i16
        %acc = comb.add %mul, %state : i17
        arc.output %acc : i17       ; new state
    }

    arc.state @MAC_step(%a, %b, %reg) clock %clk

    Why this model is fast for simulation:
        1. Pure functions are trivially parallelisable (no data races)
        2. Aggressive CSE across clock cycles (memoisation)
        3. Dead state elimination (unused registers removed)
        4. Passes can vectorise simulation over many input vectors simultaneously

### Arc for ML Training Data Generation

    AI hardware often requires large simulation datasets for:
        - Training ML models to predict timing/power (ML-for-EDA)
        - Generating test vectors for coverage-driven verification
        - Pre-silicon performance modelling of accelerator designs

    Arc's simulation throughput (much faster than Verilator for parallel runs)
    enables generating millions of simulation traces efficiently.


##### PART 7 — CIRCT IN THE AI/ML COMPILER STACK

### How CIRCT Connects to MLIR (Previous Module)

    CIRCT IS MLIR. It is not a separate project — it is a collection of
    MLIR dialects hosted in a separate repository (llvm/circt).

    CIRCT reuses EVERYTHING from MLIR:
        - Same Context, Module, Operation classes
        - Same pass manager (hardware passes are just MLIR passes)
        - Same pattern rewriting engine
        - Same type system infrastructure (HW adds i1–i65536 bitvectors)
        - Same textual IR format and binary serialisation
        - Same testing infrastructure (FileCheck-based lit tests)

    MLIR → CIRCT connection:
        func.func dialect:  used for Calyx's control functions
        arith dialect:      used inside Comb for arithmetic semantics
        LLVM dialect:       CIRCT can lower to LLVM for software co-simulation
        affine dialect:     used in some high-level synthesis paths

### How CIRCT Connects to XLA (Next Module)

    XLA compiles ML models to accelerator code.
    For TPU-like custom ASICs, XLA needs a hardware compiler backend.

    XLA → CIRCT path (experimental/research):
        XLA HLO graph → StableHLO → Calyx dialect (via academic tools)
            → CIRCT lowering pipeline → Verilog → FPGA/ASIC synthesis

    The OpenXLA project (2023) is working on:
        - A direct StableHLO → Calyx lowering (for systolic array targets)
        - Using CIRCT's Seq/Comb dialects for tensor core RTL generation
        - CIRCT-based verification of generated accelerator hardware

### How CIRCT Connects to TVM (Final Module)

    TVM targets FPGAs and custom accelerators via its VTA (Versatile
    Tensor Accelerator) hardware template.

    VTA → CIRCT path:
        TVM VTA schedule → HLS C code → Vivado HLS (old path)
        TVM VTA schedule → Calyx dialect (new research path)
            → CIRCT pipeline → Verilog → Xilinx/Intel FPGA synthesis

    The advantage of the CIRCT path:
        - Formal verification of generated hardware (vs. none in HLS)
        - Open-source toolchain (vs. proprietary Vivado HLS)
        - Direct MLIR interoperability (share passes with XLA/torch-mlir)

### The Full AI Accelerator Compilation Stack

    ┌─────────────────────────────────────────────────────────────────────┐
    │  ML Framework (PyTorch / JAX / TF)                                  │
    ├─────────────────────────────────────────────────────────────────────┤
    │  MLIR (module 2) — StableHLO / Linalg / Affine                      │
    │  [maps WHAT to compute: matmuls, convs, attention]                  │
    ├─────────────────────────────────────────────────────────────────────┤
    │  XLA / TVM (modules 4–5) — hardware-aware code generation           │
    │  [decides HOW to compute: kernel fusion, tile sizes, layouts]       │
    ├─────────────────────────────────────────────────────────────────────┤
    │  CIRCT (this module) — hardware description                         │
    │  Calyx dialect:  WHAT hardware to build + WHEN each unit fires      │
    │  HW/Comb/Seq:    register-transfer logic (the actual circuit)       │
    │  SV dialect:     synthesisable SystemVerilog for EDA tools          │
    ├─────────────────────────────────────────────────────────────────────┤
    │  EDA Tools (Yosys / Synopsys / Vivado)                              │
    │  Logic synthesis → Place & route → Timing closure                   │
    ├─────────────────────────────────────────────────────────────────────┤
    │  Silicon (ASIC) or FPGA bitstream                                   │
    │  [custom AI accelerator running the original ML model]              │
    └─────────────────────────────────────────────────────────────────────┘

### Key Differences: CIRCT vs Traditional EDA Tools

    ┌───────────────────────┬─────────────────────┬───────────────────────┐
    │  Property             │  CIRCT              │  Traditional EDA      │
    ├───────────────────────┼─────────────────────┼───────────────────────┤
    │  Infrastructure       │  Open / MLIR-based  │  Proprietary          │
    │  IR extensibility     │  Add a dialect      │  Not possible         │
    │  Pass reuse           │  Cross-dialect      │  Tool-specific        │
    │  Python integration   │  First-class        │  Scripting only       │
    │  Formal verification  │  Built-in (LTL)     │  Separate, expensive  │
    │  Simulation speed     │  Arc (parallel)     │  Verilator / VCS      │
    │  ML framework link    │  MLIR-native        │  None                 │
    │  Cost                 │  Free (Apache 2.0)  │  $100k–$1M+/year      │
    └───────────────────────┴────────────────────┴────────────────────────┘

"""


# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {

    # ── 1 ─────────────────────────────────────────────────────────────────────
    "1 · CIRCT Dialects & IR — Hardware Structure and Combinational Logic": {
        "description": (
            "Hands-on CIRCT IR. Read and understand HW, Comb, and Seq dialect syntax. "
            "Trace a multiply-accumulate unit from FIRRTL → HW/Comb/Seq → SV. "
            "Show the hardware type system: bitvectors, clocks, bundles. "
            "Demonstrate the separation between combinational (comb) and sequential (seq) logic. "
            "Connect to MLIR: show how CIRCT reuses func, arith, and affine dialects."
        ),
        "language": "python",
        "code": '''
import numpy as np

print("=" * 65)
print("  CIRCT DIALECTS & IR — HW, COMB, SEQ EXPLAINED")
print("=" * 65)
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: CIRCT IR textual format guide
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — CIRCT IR format: types and operation syntax")
print("━" * 65)
print()

CIRCT_FORMAT = """
  CIRCT IR SYNTAX GUIDE
  ════════════════════════════════════════════════════════════════

  HARDWARE TYPES (CIRCT-specific, not in base MLIR):
    i1, i8, i16, i32, i64    bitvectors (same notation as MLIR integers)
                              BUT semantics differ: i8 is a bundle of 8 wires,
                              not an arithmetic integer with overflow behaviour
    i0                        zero-width wire (used in parameterised designs)
    !hw.inout<i8>             inout wire (bidirectional, e.g. tri-state bus)
    !seq.clock                clock signal type (distinct from data i1)
    !firrtl.uint<8>           FIRRTL unsigned 8-bit; width can be implicit (<>)
    !firrtl.sint<16>          FIRRTL signed 16-bit
    !firrtl.bundle<a: uint<8>, b: sint<16>>   FIRRTL struct (aggregate)
    !firrtl.vector<uint<8>, 4>                FIRRTL array of 4 bytes

  MODULE PORTS:
    HW modules declare ports explicitly with direction:
      hw.module @Name(%in_port: i8, %clk: !seq.clock) -> (out_port: i16)
      ;                ^input ports                       ^output ports
    FIRRTL modules use flip for direction:
      firrtl.module @Name(in %a: !firrtl.uint<8>,
                          out %z: !firrtl.uint<9>)

  DIALECTS IN PLAY (CIRCT adds on top of MLIR):
    hw.*       module hierarchy, instances, aggregates
    comb.*     combinational logic (no state, no clock)
    seq.*      sequential logic (registers, memories, clocks)
    firrtl.*   FIRRTL typed RTL with width inference
    sv.*       structural SystemVerilog constructs
    calyx.*    Calyx hardware scheduling language
    arc.*      simulation-optimised arc IR
    ltl.*      Linear Temporal Logic properties
    verif.*    assertion / assumption / coverage

  SHARED WITH MLIR (reused without modification):
    func.func  function definitions (used in Calyx)
    arith.*    arithmetic semantics inside comb ops
    affine.*   loop schedules in some HLS paths
    llvm.*     software co-simulation lowering
"""
print(CIRCT_FORMAT)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: MAC unit — full lowering FIRRTL → HW/Comb/Seq → SV
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — MAC unit: FIRRTL → HW/Comb/Seq → SV lowering")
print("━" * 65)
print()

MAC_LOWERING = """
  Multiply-Accumulate: out = reg + (a * b)
  This is the fundamental compute cell in every AI accelerator.

  ── LEVEL 4: FIRRTL dialect (typed RTL, width inference) ─────────────

  firrtl.circuit @MAC {
    firrtl.module @MAC(
        in  %a   : !firrtl.uint<8>,
        in  %b   : !firrtl.uint<8>,
        in  %c   : !firrtl.uint<16>,        ; existing accumulator
        in  %clk : !firrtl.clock,
        out %out : !firrtl.uint<17>) {      ; width = 17 (infered: 16+16+1)

      %mul = firrtl.mul %a, %b
             : (!firrtl.uint<8>, !firrtl.uint<8>) -> !firrtl.uint<16>
             ; FIRRTL mul: output width = width(a) + width(b) = 16

      %add = firrtl.add %mul, %c
             : (!firrtl.uint<16>, !firrtl.uint<16>) -> !firrtl.uint<17>
             ; FIRRTL add: output width = max(w_a, w_b) + 1 = 17

      %reg = firrtl.reg %clk
             : !firrtl.clock, !firrtl.uint<17>
             ; register: captures %add on rising edge of %clk

      firrtl.connect %reg, %add           ; D input of register
      firrtl.connect %out, %reg           ; output is register Q
    }
  }

  FIRRTL passes applied here:
    1. Width inference   → fills in implicit <> widths from context
    2. Expand whens      → expands conditional connects to muxes
    3. Lower types       → Bundle/Vector → flat i8/i16 wires
    4. Lower intrinsics  → firrtl.mul → hw.module @MulUnit instance

  ── LEVEL 3: HW + Comb + Seq dialects (after FIRRTL lowering) ─────────

  hw.module @MAC(%a: i8, %b: i8, %c: i16,
                 %clk: !seq.clock) -> (out: i17) {

    // Combinational multiply: purely functional, no side effects
    %mul = comb.mul %a, %b : i8
    ; Note: comb.mul %a : i8, %b : i8 → result is i8 (same width)
    ; To get full width: use comb.concat with sign extension first

    // Zero-extend a and b to 16 bits before multiply
    %zero8 = hw.constant 0 : i8
    %a_ext = comb.concat %zero8, %a : (i8, i8) -> i16
    %b_ext = comb.concat %zero8, %b : (i8, i8) -> i16
    %mul16 = comb.mul %a_ext, %b_ext : i16

    // Combinational add: zero-extend to 17 bits
    %zero16 = hw.constant 0 : i16
    %zero1  = hw.constant 0 : i1
    %mul17  = comb.concat %zero1,  %mul16 : (i1,  i16) -> i17
    %c17    = comb.concat %zero1,  %c     : (i1,  i16) -> i17
    %sum    = comb.add %mul17, %c17 : i17

    // Sequential register: captures sum on rising clock edge
    %reg = seq.firreg %sum clock %clk : i17
    ; seq.firreg = D flip-flop: Q captures D on posedge clk

    hw.output %reg : i17
  }

  Key observations at HW/Comb/Seq level:
    - comb.* ops are PROVEN PURE: CSE and constant folding are safe.
    - seq.firreg is the ONLY state element: explicit in the IR.
    - hw.module declares the interface; no implicit global state.
    - Width handling is EXPLICIT: no implicit sign extension.

  ── LEVEL 2: SV dialect (after HW/Comb/Seq → SV lowering) ────────────

  sv.always_ff posedge %clk {
    ; if reset:   sv.passign %reg_q, 17'd0;
    ; else:       sv.passign %reg_q, %sum;
    sv.passign %reg_q, %sum   : i17
    ; non-blocking assignment: reg_q <= sum (at clock edge)
  }
  sv.assign %out = %reg_q     : i17
  ; continuous assignment: out = reg_q (combinational)

  ── LEVEL 1: Exported Verilog (after exportVerilog) ───────────────────

  module MAC(
    input  [7:0]  a,
    input  [7:0]  b,
    input  [15:0] c,
    input         clk,
    output [16:0] out
  );
    wire [15:0] mul16;
    wire [16:0] mul17, c17, sum;
    reg  [16:0] reg_q;

    assign mul16 = {8\\'d0, a} * {8\\'d0, b};
    assign mul17 = {1\\'d0, mul16};
    assign c17   = {1\\'d0, c};
    assign sum   = mul17 + c17;
    assign out   = reg_q;

    always @(posedge clk)
      reg_q <= sum;
  endmodule
"""
print(MAC_LOWERING)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Python simulation of the MAC unit behaviour
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — MAC unit behavioural simulation (Python)")
print("━" * 65)
print()

MAC_SIM = """
  Simulating the generated Verilog MAC unit's cycle-accurate behaviour.
  This is what an Arc-based CIRCT simulation would compute at high speed.
  We implement it here in Python to verify correctness.
"""
print(MAC_SIM)

class MACUnit:
    """Cycle-accurate model of the CIRCT-generated MAC register."""
    def __init__(self, width_a=8, width_b=8, width_c=16):
        self.width_a = width_a
        self.width_b = width_b
        self.width_out = width_a + width_b + 1   # full precision: 17 bits
        self.mask = (1 << self.width_out) - 1
        self._reg_q = 0      # sequential register (D flip-flop state)

    def comb_mac(self, a, b, c):
        """Purely combinational path: sum = (a * b) + c  (no register)."""
        a = a & ((1 << self.width_a) - 1)   # mask to width
        b = b & ((1 << self.width_b) - 1)
        c = c & ((1 << (self.width_out - 1)) - 1)
        return (a * b + c) & self.mask

    def posedge_clk(self, a, b, c):
        """Rising clock edge: register captures the combinational result."""
        d = self.comb_mac(a, b, c)
        self._reg_q = d          # D flip-flop: Q <= D at posedge clk
        return self._reg_q

    @property
    def out(self):
        return self._reg_q

print("  Instantiating MAC(a:8-bit, b:8-bit, c:16-bit) → out:17-bit")
print()

mac = MACUnit()

# ── clock cycle simulation ──────────────────────────────────────────────
print("  Clock cycle simulation:")
print(f"  {'Cycle':>5}  {'a':>5}  {'b':>5}  {'c (in)':>8}  {'a*b':>6}  {'out (reg)':>10}  {'expected':>10}")
print("  " + "-" * 60)

test_vectors = [
    (  3,   5,    0),    # cycle 0:  3*5=15,  acc=15
    (  4,   4,   15),    # cycle 1:  4*4=16,  acc=31
    ( 10,  10,   31),    # cycle 2: 10*10=100,acc=131
    (255, 255,  131),    # cycle 3: max*max=65025, acc=65156 (tests width)
    (  0,   0, 65156),   # cycle 4:  0*0=0,   acc=65156
    (  1,   1, 65156),   # cycle 5:  1*1=1,   acc=65157
]

for i, (a, b, c) in enumerate(test_vectors):
    expected = (a * b + c)
    result   = mac.posedge_clk(a, b, c)
    ok = "✅" if result == expected else "❌"
    print(f"  {i:>5}  {a:>5}  {b:>5}  {c:>8}  {a*b:>6}  {result:>10}  {expected:>10}  {ok}")

print()
print(f"  Maximum 17-bit value: {(1<<17)-1} = 131071")
print(f"  Max possible output:  255*255 + 65025 = {255*255 + 65025}")
print(f"  Fits in 17 bits?      {255*255 + 65025 <= (1<<17)-1}")
print()

# ── batch simulation: systolic array row ───────────────────────────────
print("  Systolic array row simulation (8 MACs in a pipeline):")
print()

class SystolicRow:
    """
    8 MAC units chained: output of MAC[i] feeds c-input of MAC[i+1].
    This is the fundamental structure of a matrix-multiply systolic array.
    In Google's TPU, thousands of these rows form the matrix unit.
    """
    def __init__(self, n=8):
        self.macs = [MACUnit() for _ in range(n)]

    def clock(self, a_vec, b_vec):
        """One clock cycle: feed one input vector, accumulate."""
        acc = 0
        for i, (mac, a, b) in enumerate(zip(self.macs, a_vec, b_vec)):
            acc = mac.posedge_clk(a, b, acc)   # chain: c = previous out
        return acc   # final accumulation = dot product

row = SystolicRow(n=8)

a = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int32)
b = np.array([8, 7, 6, 5, 4, 3, 2, 1], dtype=np.int32)
expected_dot = int(np.dot(a, b))

result = row.clock(a.tolist(), b.tolist())
print(f"  a = {a.tolist()}")
print(f"  b = {b.tolist()}")
print(f"  dot(a, b) = {expected_dot}")
print(f"  Systolic row result: {result}")
print(f"  Correct? {result == expected_dot} {'✅' if result == expected_dot else '❌'}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 4: CIRCT Python bindings
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 4 — CIRCT Python bindings")
print("━" * 65)
print()

try:
    import circt
    from circt.dialects import hw, comb, seq
    from mlir.ir import Context, Module, InsertionPoint, Location, IntegerType

    print(f"  CIRCT Python bindings available ✅")
    print()

    with Context() as ctx, Location.unknown():
        circt.register_dialects(ctx)
        module = Module.create()
        i8 = IntegerType.get_signless(8)
        i17 = IntegerType.get_signless(17)

        with InsertionPoint(module.body):
            ports = [
                hw.PortInfo(hw.PortDirection.INPUT,  "a",   i8),
                hw.PortInfo(hw.PortDirection.INPUT,  "b",   i8),
                hw.PortInfo(hw.PortDirection.OUTPUT, "out", i17),
            ]
            m = hw.HWModuleOp(name="MAC", ports=ports)
            with InsertionPoint(m.body):
                a, b = m.entry_block.arguments
                zero = hw.ConstantOp(i8, 0)
                a_ext = comb.ConcatOp(i17, [zero, a])
                b_ext = comb.ConcatOp(i17, [zero, b])
                mul   = comb.MulOp(i17, [a_ext, b_ext])
                hw.OutputOp([mul])

        print("  Generated CIRCT HW module:")
        for line in str(module).split("\\n"):
            print(f"    {line}")

except ImportError:
    print("  CIRCT Python bindings not installed.")
    print("  Install: pip install circt-core")
    print()

    CIRCT_PYTHON_EXAMPLE = """
  CIRCT Python bindings example:

  import circt
  from circt.dialects import hw, comb, seq
  from mlir.ir import Context, Module, InsertionPoint, Location, IntegerType

  with Context() as ctx, Location.unknown():
      circt.register_dialects(ctx)
      module = Module.create()
      i8  = IntegerType.get_signless(8)
      i16 = IntegerType.get_signless(16)

      with InsertionPoint(module.body):
          ports = [
              hw.PortInfo(hw.PortDirection.INPUT,  "a",   i8),
              hw.PortInfo(hw.PortDirection.INPUT,  "b",   i8),
              hw.PortInfo(hw.PortDirection.OUTPUT, "product", i16),
          ]
          m = hw.HWModuleOp(name="Mul8x8", ports=ports)
          with InsertionPoint(m.body):
              a, b = m.entry_block.arguments
              # Zero-extend to 16 bits
              z8 = hw.ConstantOp(i8, 0)
              a_ext = comb.ConcatOp(i16, [z8, a])
              b_ext = comb.ConcatOp(i16, [z8, b])
              prod  = comb.MulOp(i16, [a_ext, b_ext])
              hw.OutputOp([prod])

      print(module)
      # Output:
      # hw.module @Mul8x8(%a: i8, %b: i8) -> (product: i16) {
      #   %c0_i8  = hw.constant 0 : i8
      #   %0 = comb.concat %c0_i8, %a : i8, i8
      #   %1 = comb.concat %c0_i8, %b : i8, i8
      #   %2 = comb.mul %0, %1 : i16
      #   hw.output %2 : i16
      # }
  """
    print(CIRCT_PYTHON_EXAMPLE)
''',
    },

    # ── 2 ─────────────────────────────────────────────────────────────────────
    "2 · FIRRTL Compilation — Width Inference and Bundle Lowering": {
        "description": (
            "Deep dive into the FIRRTL compiler pipeline inside CIRCT. "
            "Show width inference: how implicit bit widths propagate through arithmetic. "
            "Demonstrate bundle flattening: firrtl.bundle → flat i8/i16 wires. "
            "Trace a RISC-V ALU module from Chisel-emitted FIRRTL → clean Verilog. "
            "Explain the annotation system (Grand Central taps, DUT boundaries). "
            "Connect to AI/ML: how Chisel-generated systolic arrays are compiled."
        ),
        "language": "python",
        "code": '''
import re

print("=" * 65)
print("  FIRRTL COMPILATION — WIDTH INFERENCE AND BUNDLE LOWERING")
print("=" * 65)
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: FIRRTL type system and width inference
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — FIRRTL type system and width inference rules")
print("━" * 65)
print()

FIRRTL_TYPES = """
  FIRRTL TYPE SYSTEM
  ════════════════════════════════════════════════════════════════

  Ground types (leaf wires):
    uint<N>        unsigned N-bit integer   (N explicit or inferred)
    uint<>         unsigned, width unknown  (FIRRTL infers it)
    sint<N>        signed N-bit (two\'s complement)
    sint<>         signed, width unknown
    clock          clock signal (i1, but type-checked separately)
    reset          synchronous reset (can be uint<1> or asyncreset)
    asyncreset     asynchronous reset signal

  Aggregate types (structural — lowered before RTL):
    bundle<field: type, ...>   struct-like
      Example: bundle<valid: uint<1>, data: uint<32>, id: uint<4>>
      After lowering → three flat wires: valid_i1, data_i32, id_i4

    vector<type, N>            array of N elements of same type
      Example: vector<uint<8>, 4>   ; 4-byte register file row
      After lowering → four flat wires: elem_0_i8, ..., elem_3_i8

  FIRRTL WIDTH INFERENCE RULES:
  ════════════════════════════════════════════════════════════════

  The key innovation: Chisel lets you write width-agnostic hardware.
  FIRRTL infers the minimum safe bit width for every wire.

  Arithmetic rules (a : uint<N>, b : uint<M>):
    add(a, b)  → uint<max(N,M) + 1>    ; need 1 extra bit for carry
    sub(a, b)  → uint<max(N,M) + 1>    ; subtract needs borrow bit
    mul(a, b)  → uint<N + M>           ; product needs N+M bits (exact)
    div(a, b)  → uint<N>               ; quotient fits in N bits
    rem(a, b)  → uint<min(N,M)>        ; remainder fits in smaller

  Shift rules:
    shl(a, n)  → uint<N + n>           ; left shift by n: N+n bits
    shr(a, n)  → uint<N - n> (min 1)   ; right shift: N-n bits

  Comparison (always returns uint<1>):
    eq, neq, lt, leq, gt, geq → uint<1>

  Mux (sel: uint<1>, a: T, b: T) → T  ; same type as inputs

  Example width propagation:
    Given: a = uint<8>, b = uint<8>
      mul = firrtl.mul a, b  → uint<16>  (8+8)
      add = firrtl.add mul, b → uint<17> (max(16,8)+1)
      cmp = firrtl.lt add, c → uint<1>
    Inferred without any explicit widths in the source!
"""
print(FIRRTL_TYPES)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Width inference engine simulation
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Width inference engine (Python simulation)")
print("━" * 65)
print()

class FIRRTLWidth:
    """
    Simulates FIRRTL\'s width inference engine.
    Each operation returns the inferred output width given input widths.
    This is what circt-opt --firrtl-infer-widths does.
    """
    @staticmethod
    def add(n, m):
        return max(n, m) + 1

    @staticmethod
    def sub(n, m):
        return max(n, m) + 1

    @staticmethod
    def mul(n, m):
        return n + m

    @staticmethod
    def div(n, m):
        return n

    @staticmethod
    def rem(n, m):
        return min(n, m)

    @staticmethod
    def shl(n, amount):
        return n + amount

    @staticmethod
    def shr(n, amount):
        return max(n - amount, 1)

    @staticmethod
    def mux(n, m):
        assert n == m, f"mux arms must have same width: {n} vs {m}"
        return n

    @staticmethod
    def concat(n, m):
        return n + m

    @staticmethod
    def eq(n, m):
        return 1

    @staticmethod
    def lt(n, m):
        return 1

w = FIRRTLWidth()

print("  Width inference trace for: MAC = reg + (a * b)")
print()

wa, wb = 8, 8
print(f"  a : uint<{wa}>")
print(f"  b : uint<{wb}>")

w_mul = w.mul(wa, wb)
print(f"  mul = firrtl.mul a, b  → uint<{w_mul}>   ({wa}+{wb})")

w_clk_not_counted = 16   # c input is same width as mul output
w_add = w.add(w_mul, w_clk_not_counted)
print(f"  c   : uint<{w_clk_not_counted}>  (existing accumulator)")
print(f"  acc = firrtl.add mul, c → uint<{w_add}>   (max({w_mul},{w_clk_not_counted})+1)")
print()

# trace a dot product accumulator
print("  Width trace for: dot8 = sum of 8 uint<8>*uint<8> products")
acc_w = 0   # initial accumulator width (zero value)
for i in range(8):
    prod_w = w.mul(8, 8)   # = 16
    if acc_w == 0:
        acc_w = prod_w
    else:
        acc_w = w.add(acc_w, prod_w)
    print(f"    after product {i+1}: acc width = {acc_w} bits")

print(f"  → accumulator needs {acc_w} bits to hold exact dot product")
print(f"  → maximum value: 8 × (255×255) = {8 * 255 * 255:,} ≤ 2^{acc_w}-1 = {(1<<acc_w)-1:,}")
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Bundle lowering — FIRRTL aggregate → flat wires
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Bundle lowering: aggregate types → flat Verilog wires")
print("━" * 65)
print()

BUNDLE_LOWERING = """
  FIRRTL bundles are a MAJOR convenience for hardware designers.
  They allow struct-like grouping of wires with named fields.
  Every bundle must be LOWERED to flat wires before Verilog emission.

  ── BEFORE LOWERING (FIRRTL bundle) ───────────────────────────────────

  ; AXI4-Lite Write Address Channel as a bundle
  firrtl.module @AXI4Lite_WAdrCh(
      out %aw : !firrtl.bundle<
                    valid  : uint<1>,
                    ready  : flip<uint<1>>,   ; flip = input direction
                    addr   : uint<32>,
                    prot   : uint<3>
                >) {

      ; Connect all fields as one operation:
      %ch = firrtl.bundlecreate %valid, %ready, %addr, %prot
              : (!firrtl.uint<1>, !firrtl.uint<1>,
                 !firrtl.uint<32>, !firrtl.uint<3>)
                → !firrtl.bundle<valid:uint<1>, ready:flip<uint<1>>,
                                  addr:uint<32>, prot:uint<3>>
      firrtl.connect %aw, %ch
  }

  ── AFTER BUNDLE FLATTENING (--firrtl-lower-types) ────────────────────

  ; All bundle fields become separate port wires
  firrtl.module @AXI4Lite_WAdrCh(
      out %aw_valid : !firrtl.uint<1>,
      in  %aw_ready : !firrtl.uint<1>,    ; flip → direction inversion
      out %aw_addr  : !firrtl.uint<32>,
      out %aw_prot  : !firrtl.uint<3>) {

      firrtl.connect %aw_valid, %_valid
      firrtl.connect %aw_addr,  %_addr
      firrtl.connect %aw_prot,  %_prot
  }

  ── AFTER FIRRTL → HW/COMB/SEQ LOWERING ──────────────────────────────

  hw.module @AXI4Lite_WAdrCh(
      %aw_ready : i1)                         ; only input
      -> (aw_valid: i1, aw_addr: i32, aw_prot: i3) {
      hw.output %valid_sig, %addr_sig, %prot_sig : i1, i32, i3
  }

  ── EMITTED VERILOG (after exportVerilog) ─────────────────────────────

  module AXI4Lite_WAdrCh (
    input        aw_ready,
    output       aw_valid,
    output [31:0] aw_addr,
    output [2:0]  aw_prot
  );
    // ... logic here ...
  endmodule

  KEY INSIGHT:
    The bundle exists ONLY in FIRRTL/Chisel for programmer convenience.
    It is COMPLETELY eliminated before Verilog. The EDA tool never sees it.
    CIRCT\'s progressive lowering handles this transparently.
"""
print(BUNDLE_LOWERING)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 4: AI accelerator context — Chisel systolic array
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 4 — AI accelerator context: Chisel systolic array")
print("━" * 65)
print()

SYSTOLIC_CHISEL = """
  Google\'s TPU uses a systolic array as its matrix unit.
  The systolic array is typically described in Chisel:

  ┌───────────────────────────────────────────────────────────────┐
  │  Chisel source → firtool (CIRCT) → Verilog → ASIC synthesis   │
  └───────────────────────────────────────────────────────────────┘

  Simplified Chisel description of a 4×4 systolic array:

    class PECell extends Module {                // Processing Element
      val io = IO(new Bundle {
        val a_in   = Input(UInt(8.W))            // from left
        val b_in   = Input(UInt(8.W))            // from top
        val c_in   = Input(UInt(32.W))           // partial sum in
        val a_out  = Output(UInt(8.W))           // pass right
        val b_out  = Output(UInt(8.W))           // pass down
        val c_out  = Output(UInt(32.W))          // partial sum out
      })
      io.a_out := RegNext(io.a_in)               // register + pass a right
      io.b_out := RegNext(io.b_in)               // register + pass b down
      io.c_out := RegNext(io.c_in + io.a_in * io.b_in) // MAC + register
    }

    class SystolicArray(n: Int = 4) extends Module {
      // ... n×n grid of PECell, wired as wavefront array
      val cells = Array.tabulate(n, n)((_, _) => Module(new PECell))
      for (i <- 0 until n; j <- 0 until n) {
        if (j > 0) cells(i)(j).io.a_in := cells(i)(j-1).io.a_out
        if (i > 0) cells(i)(j).io.b_in := cells(i-1)(j).io.b_out
        if (i > 0) cells(i)(j).io.c_in := cells(i-1)(j).io.c_out
      }
    }

  Compilation with firtool (CIRCT\'s FIRRTL compiler):

    $ chisel3 emit SystolicArray > SystolicArray.fir
    $ firtool SystolicArray.fir --format=fir  \\\\
              --split-verilog -o output_dir/
    →  output_dir/SystolicArray.sv   (top module)
    →  output_dir/PECell.sv          (cell module, one file per module)

  CIRCT transforms applied automatically:
    1. firrtl-infer-widths      → fill in all implicit bit widths
    2. firrtl-expand-whens      → when/otherwise → mux chains
    3. firrtl-lower-types       → bundle/vector → flat wires
    4. firrtl-imconstprop       → propagate constants
    5. firrtl-remove-unused     → dead wire elimination
    6. lower-firrtl-to-hw       → firrtl.* → hw/comb/seq dialects
    7. hw-legalize-modules      → remove unsynthesisable constructs
    8. exportVerilog            → emit .sv files

  Output quality vs hand-written Verilog:
    - Width annotations exact (no inadvertent truncations)
    - Formal correctness: bundle semantics preserved through lowering
    - firtool is the compiler used for EVERY SiFive/Western Digital RISC-V core
    - Google\'s TPU RTL flow uses a Chisel/FIRRTL/CIRCT-based pipeline
"""
print(SYSTOLIC_CHISEL)
''',
    },

    # ── 3 ─────────────────────────────────────────────────────────────────────
    "3 · Calyx Dialect — Hardware Scheduling for AI Accelerators": {
        "description": (
            "Deep dive into the Calyx hardware scheduling language inside CIRCT. "
            "Build a dot-product accelerator: cells (MAC units), wires, control schedule. "
            "Show the Calyx → HW/Comb/Seq → Verilog pipeline. "
            "Demonstrate the control model: seq, par, while, if, invoke. "
            "Connect to AI/ML: how TVM and XLA can target Calyx for FPGA acceleration."
        ),
        "language": "python",
        "code": '''
import numpy as np

print("=" * 65)
print("  CALYX DIALECT — HARDWARE SCHEDULING FOR AI ACCELERATORS")
print("=" * 65)
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Calyx model — cells, wires, control
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Calyx model: cells, wires, control")
print("━" * 65)
print()

CALYX_OVERVIEW = """
  CALYX: A HARDWARE INTERMEDIATE LANGUAGE FOR ACCELERATOR DESIGN
  ════════════════════════════════════════════════════════════════

  Calyx models hardware as three orthogonal concerns:

  1. CELLS — what functional units exist?
     Standard library cells (parameterised):
       std_add(N)         N-bit combinational adder
       std_mult_pipe(N)   N-bit pipelined multiplier (multi-cycle)
       std_reg(N)         N-bit D register
       std_mem_d1(W,D,A)  1D memory: W-bit words, D entries, A-bit addr
       std_mem_d2(W,R,C,RA,CA)  2D memory (for matrix storage)
       std_mux(N)         N-bit 2-to-1 multiplexer
       std_lt(N)          N-bit less-than comparator (→ 1-bit)
       std_ge(N)          N-bit greater-than-or-equal

  2. WIRES — how are cells connected when active?
     Groups (named activation contexts):
       group load_a {                   // group activates these wires
           mem_a.addr0  = i_reg.out;   // address = loop counter
           mem_a.read_en = 1\'d1;        // enable read
           rA.in        = mem_a.read_data;  // capture to register
           rA.write_en  = 1\'d1;
           load_a[done] = rA.done;     // group completes when reg done
       }
     Continuous assignments (always active, outside groups):
       wires {
           comb group reset_check {
               cond.left  = i_reg.out;
               cond.right = n_const.out;  // i < N ?
           }
       }

  3. CONTROL — when does each group execute?
     Control operators:
       seq { A; B; C }           execute A, then B, then C
       par { A; B }              execute A and B simultaneously
       while %cond { body }      repeat body while cond is true
       if %cond { T } else { F } conditional
       invoke cell(...)          call a sub-component
       enable group_name         execute one group for one cycle

  THE KEY INSIGHT FOR AI ACCELERATORS:
    Calyx separates WHAT (wires) from WHEN (control).
    This mirrors how ML frameworks separate the compute graph (what)
    from the schedule/tiling strategy (when).
    A matmul in Calyx = data path cells + tiled loop nest control.
"""
print(CALYX_OVERVIEW)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: Dot product accelerator in Calyx IR
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — Dot product accelerator: Calyx IR")
print("━" * 65)
print()

DOT_PRODUCT_CALYX = """
  DOT PRODUCT ACCELERATOR: sum = sum(a[i] * b[i]) for i in 0..N-1
  This is the inner computation of every GEMM / attention layer.

  ── CALYX IR ──────────────────────────────────────────────────────────

  component dot_product<"static"=false>(
      @clk clk: 1,
      @reset reset: 1,
      @go go: 1,
      n: 32                          ; number of elements
  ) -> (
      result: 32,
      @done done: 1
  ) {
    cells {
      // Memory for input vectors a and b
      mem_a  = std_mem_d1(8,  256, 8);  // 256 × 8-bit words, 8-bit addr
      mem_b  = std_mem_d1(8,  256, 8);

      // Pipelined multiplier: 8×8 → 16-bit, 3-cycle latency
      mul    = std_mult_pipe(8);

      // Accumulator adder: 32-bit (headroom for 256 products)
      adder  = std_add(32);

      // Registers
      rA     = std_reg(8);           // holds mem_a[i]
      rB     = std_reg(8);           // holds mem_b[i]
      rMul   = std_reg(16);          // holds mul output
      acc    = std_reg(32);          // accumulator

      // Loop counter
      i_reg  = std_reg(8);
      i_add  = std_add(8);           // i increment
      i_lt   = std_lt(8);            // i < n comparator
      n_reg  = std_reg(8);           // hold n

      // Constants
      one    = std_const(8, 1);
      zero32 = std_const(32, 0);
      zero8  = std_const(8, 0);
    }

    wires {
      // ── Continuous (always-on) ──
      comb group cond_check {
          i_lt.left  = i_reg.out;
          i_lt.right = n_reg.out;   // condition: i < n
      }

      // ── Step 0: initialise loop ───────────────────────────────────
      group init {
          acc.in       = zero32.out;
          acc.write_en = 1\'d1;
          i_reg.in     = zero8.out;
          i_reg.write_en = 1\'d1;
          n_reg.in     = n;           // capture input port n
          n_reg.write_en = 1\'d1;
          init[done]   = acc.done & i_reg.done & n_reg.done;
      }

      // ── Step 1: load a[i] from memory ────────────────────────────
      group load_a {
          mem_a.addr0    = i_reg.out;
          mem_a.read_en  = 1\'d1;
          rA.in          = mem_a.read_data;
          rA.write_en    = mem_a.done;
          load_a[done]   = rA.done;
      }

      // ── Step 2: load b[i] from memory ────────────────────────────
      group load_b {
          mem_b.addr0    = i_reg.out;
          mem_b.read_en  = 1\'d1;
          rB.in          = mem_b.read_data;
          rB.write_en    = mem_b.done;
          load_b[done]   = rB.done;
      }

      // ── Step 3: multiply (pipelined, 3 cycles) ───────────────────
      group do_mul {
          mul.left      = rA.out;
          mul.right     = rB.out;
          mul.go        = 1\'d1;
          rMul.in       = mul.out;
          rMul.write_en = mul.done;
          do_mul[done]  = rMul.done;
      }

      // ── Step 4: accumulate ───────────────────────────────────────
      group do_add {
          adder.left    = acc.out;
          adder.right   = {16\'d0, rMul.out};  // zero-extend 16→32 bit
          acc.in        = adder.out;
          acc.write_en  = 1\'d1;
          do_add[done]  = acc.done;
      }

      // ── Step 5: increment loop counter ───────────────────────────
      group incr_i {
          i_add.left    = i_reg.out;
          i_add.right   = one.out;
          i_reg.in      = i_add.out;
          i_reg.write_en = 1\'d1;
          incr_i[done]  = i_reg.done;
      }
    }

    control {
      seq {
        init;                          // initialise acc=0, i=0
        while i_lt.out with cond_check {   // while i < n
          seq {
            par { load_a; load_b; }    // PARALLEL: load a[i] and b[i] simultaneously
            do_mul;                    // multiply (3 cycles, pipelined)
            do_add;                    // acc += product
            incr_i;                    // i++
          }
        }
      }
    }
  }

  WHAT CIRCT DOES WITH THIS:
    1. calyx-remove-groups → inline group logic into always blocks
    2. calyx-go-insertion   → insert go/done handshaking signals
    3. calyx-to-hw          → lower to hw/comb/seq/sv dialects
    4. exportVerilog         → emit synthesisable SystemVerilog
"""
print(DOT_PRODUCT_CALYX)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Calyx scheduling simulation — cycle counts
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Calyx scheduling simulation: cycle counting")
print("━" * 65)
print()

class CalyxScheduler:
    """
    Simulates Calyx control schedule execution.
    Counts clock cycles for the dot product accelerator above.
    """
    def __init__(self, mem_latency=1, mul_latency=3, add_latency=1):
        self.mem_lat = mem_latency
        self.mul_lat = mul_latency
        self.add_lat = add_latency

    def dot_product_cycles(self, n):
        """Total clock cycles for dot product of length n."""
        init_cycles = 1   # acc=0, i=0 in parallel

        # Per-iteration:
        load_cycles = self.mem_lat   # par{load_a, load_b} → max latency = mem_lat
        mul_cycles  = self.mul_lat
        add_cycles  = self.add_lat
        incr_cycles = 1

        iter_cycles = load_cycles + mul_cycles + add_cycles + incr_cycles
        total = init_cycles + n * iter_cycles
        return total, iter_cycles

    def matmul_cycles(self, M, N, K):
        """Cycles for M×K @ K×N matmul as nested Calyx loops."""
        _, inner_dot = self.dot_product_cycles(K)
        # outer loop overhead: init + N iterations per row
        per_row = N * (inner_dot * K + 2)   # simplified
        return M * per_row

sched = CalyxScheduler()

print("  Dot product cycle counts for various lengths:")
print(f"  {'N':>6}  {'total cycles':>14}  {'per iteration':>14}")
print("  " + "-" * 40)
for n in [4, 8, 16, 32, 64, 128, 256]:
    total, per_iter = sched.dot_product_cycles(n)
    print(f"  {n:>6}  {total:>14}  {per_iter:>14}")

print()
print("  Comparison: sequential (no parallelism) vs Calyx par{load_a, load_b}:")
sched_seq = CalyxScheduler(mem_latency=1, mul_latency=3, add_latency=1)
sched_par = CalyxScheduler(mem_latency=1, mul_latency=3, add_latency=1)

n = 64
total_seq, _   = sched_seq.dot_product_cycles(n)
total_par_same, _ = sched_par.dot_product_cycles(n)
# with par, load_a and load_b overlap — already counted in our model
print(f"  N={n}: sequential load = {total_seq} cycles")
print(f"  N={n}: par{{load_a, load_b}} = {total_par_same} cycles (overlapped)")
print(f"  (par saves {n * sched.mem_lat} cycles = one mem_latency per iteration)")

# ─────────────────────────────────────────────────────────────────────────
# SECTION 4: TVM and XLA → Calyx connection
# ─────────────────────────────────────────────────────────────────────────
print()
print("━" * 65)
print("  SECTION 4 — TVM / XLA → Calyx → FPGA pipeline")
print("━" * 65)
print()

TVM_XLA_CALYX = """
  HOW ML FRAMEWORKS TARGET CALYX FOR FPGA DEPLOYMENT
  ════════════════════════════════════════════════════

  Current state (2024):
    TVM has an experimental Calyx backend:
      TVM → Relay → TIR (tiled loops) → Calyx IR → firtool → Verilog → FPGA

    XLA research path (CIRCT paper, 2022):
      XLA HLO → StableHLO → Calyx (via academic lowering) → firtool → FPGA

  Why this matters:

    Traditional FPGA ML deployment:
      TVM/XLA → HLS C++ → Vivado HLS (Xilinx) / Vitis AI
      PROBLEM: HLS C++ has unpredictable timing; tool generates slow circuits;
               no formal guarantees; proprietary black box; $200k+ licence.

    Calyx/CIRCT FPGA deployment:
      TVM/XLA → Calyx dialect → CIRCT → Verilog → open-source synthesis
      BENEFIT: Formal semantics; predictable cycles; open toolchain;
               directly composable with MLIR passes in the same pipeline.

  Calyx matmul from TVM perspective:
    TVM schedule:   tile(i, j, k, tile_size=8)
    → Calyx control: while i < M { while j < N { while k < K step 8 { ... }}}
    → Calyx cells:  8-wide MAC array (par 8 multiplies per cycle)
    → Verilog:      8 std_mult_pipe instances, shared mem read ports

  The unifying abstraction:
    ┌──────────────────────────────────────────────────────────────┐
    │  ML compute graph → scheduling IR → hardware data path       │
    │                                                              │
    │  MLIR Linalg  = compute graph (what operations, what shapes) │
    │  Calyx control= schedule (when each unit fires, loop bounds) │
    │  HW/Comb/Seq  = data path (actual gates and registers)       │
    │                                                              │
    │  MLIR handles levels 1–2. CIRCT handles levels 2–3.         │
    │  The boundary is the Calyx dialect — shared between them.    │
    └──────────────────────────────────────────────────────────────┘
"""
print(TVM_XLA_CALYX)
''',
    },

    # ── 4 ─────────────────────────────────────────────────────────────────────
    "4 · Formal Verification — LTL Properties and Hardware Correctness": {
        "description": (
            "Deep dive into CIRCT's formal verification infrastructure. "
            "Show LTL dialect: always, eventually, until, next. "
            "Write hardware properties for an AI accelerator: overflow guards, "
            "handshake protocol correctness, pipelining invariants. "
            "Demonstrate verif.assert/assume/cover on a MAC unit. "
            "Connect to BMC (bounded model checking) via circt-bmc."
        ),
        "language": "python",
        "code": '''
print("=" * 65)
print("  FORMAL VERIFICATION — LTL PROPERTIES FOR HARDWARE CORRECTNESS")
print("=" * 65)
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Why formal verification matters for AI hardware
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — Why formal verification for AI accelerators?")
print("━" * 65)
print()

FORMAL_MOTIVATION = """
  WHY SIMULATION IS NOT ENOUGH FOR AI HARDWARE
  ════════════════════════════════════════════════════════════════

  A Google TPU matrix unit has ~65,536 MAC cells.
  A simulation test for correctness might cover:
    - 10,000 random input vectors per test
    - 50 test configurations (batch sizes, matrix shapes)
    - Total: 500,000 simulation runs

  But the input space is:
    - Each cell: 8-bit × 8-bit × 32-bit accumulator
    - Total inputs: 2^8 × 2^8 × 2^32 per cell = 2^48 combinations
    - For 65,536 cells: astronomically large

  Simulation covers a TINY fraction of the state space.
  Formal verification covers ALL of it mathematically.

  What formal verification proves about AI accelerators:

  1. OVERFLOW FREEDOM:
     "For all 8-bit inputs a, b and 32-bit accumulator c,
      the output acc = a*b + c never overflows 33 bits."
     → Proved by bit-width analysis (CIRCT does this automatically).

  2. PIPELINE CORRECTNESS:
     "For a 3-stage pipelined multiplier, the output at cycle T+3
      always equals a_in[T] × b_in[T] for any inputs."
     → Proved by LTL inductive invariant.

  3. HANDSHAKE PROTOCOL:
     "The valid/ready AXI handshake: whenever valid is asserted,
      data is held stable until ready is asserted."
     → Proved by LTL property: always(valid → (data == stable until ready))

  4. MEMORY HAZARD FREEDOM:
     "A write to address A followed by a read from address A
      within 2 cycles always returns the written value."
     → Proved by bounded model checking (BMC) with depth 4.
"""
print(FORMAL_MOTIVATION)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: LTL dialect syntax and semantics
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — LTL dialect: operators and temporal semantics")
print("━" * 65)
print()

LTL_SYNTAX = """
  LTL (LINEAR TEMPORAL LOGIC) DIALECT IN CIRCT
  ════════════════════════════════════════════════════════════════

  LTL extends Boolean logic with TEMPORAL operators:
  All operators are evaluated relative to the current clock cycle.

  Types:
    !ltl.sequence    a sequence of Boolean conditions over time
    !ltl.property    a temporal property (to be verified)

  Operators:

  ltl.delay %seq, %delay, %length
    ; %seq holds starting %delay cycles from now,
    ; for %length consecutive cycles.
    ; ##2 in SVA = ltl.delay %s, 2, 1

  ltl.concat %seq_a, %seq_b
    ; %seq_a holds, then immediately %seq_b holds.
    ; Sequencing without gap.

  ltl.repeat %seq, %min, %max
    ; %seq repeats between %min and %max times.

  ltl.eventually %seq
    ; ◇ seq: at some future cycle, %seq becomes true.
    ; Used for liveness: "the computation will eventually complete."

  ltl.always %prop
    ; □ prop: true at EVERY clock cycle from now onwards.
    ; Used for safety: "the overflow flag is never set."

  ltl.until %cond, %prop
    ; %prop holds at every cycle UNTIL %cond becomes true.
    ; Used for stability: "data is stable until ready is asserted."

  ltl.implication %seq, %prop
    ; if %seq matches now, then %prop must hold.
    ; Maps to SVA: seq |-> prop
    ; This is the most common hardware verification pattern.

  ltl.not %prop
    ; logical negation of a property (for assume/assert duality)

  Connecting to verif dialect:
    verif.assert %prop : !ltl.property     ; property must hold
    verif.assume %prop : !ltl.property     ; constrain the solver
    verif.cover  %prop : !ltl.property     ; measure reachability

  Compilation targets:
    LTL + verif → SVA assertions in the SV dialect
    LTL + verif → circt-bmc tool (bounded model checking)
    LTL + verif → Yices2/Z3 SMT solver (via circt-lec)
"""
print(LTL_SYNTAX)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: Writing hardware properties for a MAC unit
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — Hardware properties for the MAC accelerator")
print("━" * 65)
print()

MAC_PROPERTIES = """
  FORMAL PROPERTIES FOR THE MAC UNIT
  ════════════════════════════════════════════════════════════════

  Recall the MAC: output = register(a * b + c)
  where a:i8, b:i8, c:i16, output:i17, clocked.

  ── PROPERTY 1: No overflow (safety) ────────────────────────────────

  ; The 17-bit register can hold any a*b+c for 8-bit a,b and 16-bit c.
  ; Maximum: 255*255 + 65535 = 65025 + 65535 = 130560 < 2^17 = 131072 ✅

  ; In LTL+verif MLIR IR:
  %max_val  = hw.constant 130560 : i17
  %out_ok   = comb.icmp ule %mac_out, %max_val : i17  ; out ≤ 130560?
  %always_ok = ltl.always %out_ok : (i1) → !ltl.property
  verif.assert %always_ok : !ltl.property
    { message = "MAC output never overflows 17 bits" }

  ; Emitted SVA (in SV dialect → Verilog):
  assert property (@(posedge clk) (out <= 17\'d130560));

  ── PROPERTY 2: Pipeline correctness for 3-stage multiplier ─────────

  ; A pipelined multiplier: result at T+3 = input_a[T] * input_b[T]
  ; We need to remember input values for 3 cycles.

  ; Use shift registers to capture a and b at cycle T:
  %a_T   = seq.firreg %a_in clock %clk        ; a[T]
  %a_T1  = seq.firreg %a_T  clock %clk        ; a[T-1]
  %a_T2  = seq.firreg %a_T1 clock %clk        ; a[T-2] = a at 3 cycles ago
  ; same for b_T2...

  ; expected output = a_T2 * b_T2 (what was input 3 cycles ago)
  %expected = comb.mul %a_T2, %b_T2 : i16

  ; actual output = registered pipeline result
  %correct  = comb.icmp eq %mul_out, %expected : i16
  %pipeline_ok = ltl.always %correct : (i1) → !ltl.property
  verif.assert %pipeline_ok : !ltl.property
    { message = "Pipelined mult: output[T+3] == a[T] * b[T]" }

  ── PROPERTY 3: AXI-valid handshake stability (protocol) ────────────

  ; AXI rule: once valid is asserted, data must not change
  ; until ready is also asserted (handshake complete).
  ;
  ; LTL encoding:
  ;   always: valid → (data stable until (valid & ready))

  %handshake  = comb.and %valid, %ready : i1
  %data_next  = seq.firreg %data clock %clk : i32   ; next-cycle data
  %data_stable = comb.icmp eq %data, %data_next : i32  ; data unchanged?

  %hold_until_ready = ltl.until %handshake, %data_stable
                      : (i1, i1) → !ltl.sequence
  %valid_implies = ltl.implication %valid_seq, %hold_until_ready
                   : (!ltl.sequence, !ltl.sequence) → !ltl.property
  %axi_prop = ltl.always %valid_implies : (!ltl.property) → !ltl.property
  verif.assert %axi_prop : !ltl.property
    { message = "AXI valid/ready: data stable until handshake" }

  ; Emitted SVA:
  assert property (@(posedge clk)
      valid |-> (data == $past(data) throughout !(!valid && !ready)[*]));
"""
print(MAC_PROPERTIES)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 4: Bounded model checking — Python simulation
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 4 — Bounded Model Checking: property exhaustion")
print("━" * 65)
print()

BMC_OVERVIEW = """
  BOUNDED MODEL CHECKING (BMC) — circt-bmc
  ════════════════════════════════════════════════════════════════

  BMC unrolls the hardware circuit for K clock cycles and checks:
  "Does any input sequence of length ≤ K violate the property?"

  For K = 10 and our MAC unit:
    - Unroll 10 cycles: 10 copies of MAC combinational logic
    - Connect outputs of cycle T to inputs of cycle T+1
    - Encode as SAT/SMT formula
    - Ask: "Is there an assignment to {a[0..9], b[0..9], c[0..9]}
            that makes the overflow property false?"
    - If UNSAT: property holds for all inputs up to depth 10
    - If SAT: counterexample found → inputs that violate property

  CIRCT BMC tool: circt-bmc input.mlir -b 20
    --unroll=20   : check up to 20 clock cycles
    Internally uses: Yices2 / Bitwuzla SMT solver
"""
print(BMC_OVERVIEW)

# Python simulation of BMC logic (simplified exhaustive check for small widths)
print("  BMC simulation (exhaustive for small bit widths):")
print()

def bmc_overflow_check(width_a=4, width_b=4, width_c=8, depth=10):
    """
    Simplified BMC: check overflow property exhaustively for reduced widths.
    Real BMC uses SAT solvers; this demonstrates the concept.
    """
    max_a    = (1 << width_a) - 1
    max_b    = (1 << width_b) - 1
    max_c    = (1 << width_c) - 1
    expected_max = max_a * max_b + max_c
    width_out = (max_a * max_b + max_c).bit_length()

    violations = []
    total_checked = 0

    # Check: for all a,b,c, does a*b+c fit in width_out bits?
    for a in range(max_a + 1):
        for b in range(max_b + 1):
            for c in range(max_c + 1):
                result = a * b + c
                total_checked += 1
                if result > (1 << width_out) - 1:
                    violations.append((a, b, c, result))

    return violations, total_checked, expected_max, width_out

print(f"  Checking MAC overflow property (reduced: a:4-bit, b:4-bit, c:8-bit)...")
violations, total, max_val, out_w = bmc_overflow_check(4, 4, 8)

print(f"  Total input combinations checked: {total:,}")
print(f"  Maximum possible output: {max_val} = requires {out_w} bits")
print(f"  Overflow violations found: {len(violations)}")
if not violations:
    print(f"  ✅ PROPERTY HOLDS: No overflow for any input combination.")
else:
    print(f"  ❌ PROPERTY VIOLATED: {violations[:3]}...")

print()

# Compare: what if we used a too-narrow accumulator?
print(f"  What if we used a 7-bit accumulator? (common mistake)")
violations_7, total_7, _, _ = bmc_overflow_check(4, 4, 7)  # c only 7 bits
# recheck: can output overflow 7 bits?
count_overflow = 0
for a in range(16):
    for b in range(16):
        for c in range(128):
            if a * b + c > 127:
                count_overflow += 1

print(f"  Overflow count with 7-bit output (should hold 15*15+127=352): {count_overflow:,} violations")
print(f"  → BMC would find this immediately and report the counterexample.")
print()
print(f"  This is what happens with hand-coded Verilog and no formal checks.")
print(f"  CIRCT\'s width inference prevents this class of bugs automatically.")
''',
    },

    # ── 5 ─────────────────────────────────────────────────────────────────────
    "5 · CIRCT in the Connected Stack — From ML Graphs to Silicon": {
        "description": (
            "End-to-end view: trace a matrix multiply from PyTorch → MLIR → Calyx → CIRCT → Verilog. "
            "Show how CIRCT connects to LLVM (shared infrastructure), MLIR (dialect reuse), "
            "XLA (TPU compilation path), and TVM (FPGA acceleration). "
            "Demonstrate firtool: the FIRRTL compiler command-line tool. "
            "Summarise the full AI accelerator toolchain and CIRCT's place in it."
        ),
        "language": "python",
        "code": '''
print("=" * 65)
print("  CIRCT IN THE CONNECTED STACK — ML GRAPHS TO SILICON")
print("=" * 65)
print()

# ─────────────────────────────────────────────────────────────────────────
# SECTION 1: Full end-to-end compilation path
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 1 — End-to-end: PyTorch → CIRCT → Verilog")
print("━" * 65)
print()

FULL_STACK = """
  FROM PYTORCH MODEL TO SILICON: THE COMPLETE PATH
  ════════════════════════════════════════════════════════════════

  ┌─────────────────────────────────────────────────────────────────────┐
  │  STEP 1: PyTorch model (Python)                                     │
  │                                                                     │
  │  class TinyLinear(nn.Module):                                       │
  │      def forward(self, x):   # x: [batch, 64]                       │
  │          return self.fc(x)   # weight: [32, 64]                     │
  │                                                                     │
  │  Operation: y[i,j] = sum_k(x[i,k] * W[j,k])  — a matmul             │
  └──────────────────┬──────────────────────────────────────────────────┘
                     │  torch.export / torch-mlir
                     ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │  STEP 2: MLIR (Compiler Infrastructure module)                      │
  │                                                                     │
  │  func.func @matmul(%x: tensor<Bx64xf32>, %W: tensor<32x64xf32>)     │
  │      → tensor<Bx32xf32> {                                           │
  │    %result = linalg.matmul ins(%x, %W) outs(%init)                  │
  │    return %result                                                   │
  │  }                                                                  │
  │                                                                     │
  │  XLA path (GPU/TPU): stablehlo.dot_general → XLA HLO → PTX/XLA      │
  │  FPGA path: linalg.matmul → Calyx dialect (step 3 below)            │
  └──────────────────┬──────────────────────────────────────────────────┘
                     │  linalg-to-calyx lowering pass (research/TVM)
                     ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │  STEP 3: Calyx dialect (CIRCT)                                      │
  │                                                                     │
  │  component matmul(M: 8, N: 8, K: 8) -> (done: 1) {                  │
  │    cells {                                                          │
  │      mac = dot_product;      // from previous operation             │
  │      out_mem = std_mem_d2(32, 32, 64, 5, 6);  // output matrix      │
  │      i_reg, j_reg = std_reg(8);                                     │
  │    }                                                                │
  │    control {                                                        │
  │      while i < M { while j < N {                                    │
  │          invoke mac(row_a=A[i], col_b=B[j]);                        │
  │          store out[i][j] = mac.result;                              │
  │      }}                                                             │
  │    }                                                                │
  │  }                                                                  │
  └──────────────────┬──────────────────────────────────────────────────┘
                     │  calyx-to-hw + comb/seq lowering
                     ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │  STEP 4: HW + Comb + Seq + SV dialects (CIRCT core)                 │
  │                                                                     │
  │  hw.module @matmul(%clk: !seq.clock, %go: i1, ...) -> (...) {       │
  │    %mac_out = seq.firreg %comb_sum clock %clk : i17                 │
  │    %done    = seq.firreg %loop_done clock %clk : i1                 │
  │    sv.always_ff posedge %clk { sv.passign %reg, %next; }            │
  │  }                                                                  │
  └──────────────────┬──────────────────────────────────────────────────┘
                     │  exportVerilog (circt-opt / firtool)
                     ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │  STEP 5: SystemVerilog (.sv files)                                  │
  │                                                                     │
  │  module matmul(                                                     │
  │    input         clk, go,                                           │
  │    input  [7:0]  M, N, K,                                           │
  │    output        done                                               │
  │  );                                                                 │
  │    // ... hundreds of lines of synthesisable RTL ...                │
  │  endmodule                                                          │
  └──────────────────┬──────────────────────────────────────────────────┘
                     │  EDA synthesis (Yosys / Synopsys DC / Vivado)
                     ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │  STEP 6: Gate-level netlist + FPGA bitstream / ASIC GDSII           │
  │  FPGA: Xilinx Alveo / Intel Stratix running the matmul accelerator  │
  │  ASIC: custom AI chip tape-out (e.g. TPU-like design)               │
  └─────────────────────────────────────────────────────────────────────┘
"""
print(FULL_STACK)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 2: firtool — the CIRCT FIRRTL compiler CLI
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 2 — firtool: the CIRCT FIRRTL compiler")
print("━" * 65)
print()

FIRTOOL_GUIDE = """
  firtool IS THE CIRCT COMMAND-LINE TOOL FOR FIRRTL COMPILATION
  ════════════════════════════════════════════════════════════════

  Installation:
    # Via conda-forge (recommended):
    conda install -c conda-forge circt

    # Or build from source:
    git clone https://github.com/llvm/circt && cd circt
    cmake -B build -DCMAKE_BUILD_TYPE=Release -DLLVM_DIR=...
    cmake --build build --target firtool

  Basic usage:
    # Compile FIRRTL → Verilog (single file):
    firtool design.fir --format=fir -o design.sv

    # Compile FIRRTL → split Verilog (one file per module):
    firtool design.fir --format=fir --split-verilog -o output/

    # Compile Chisel-emitted FIRRTL with annotations:
    firtool design.fir --format=fir \\
            --mlir-print-debuginfo \\
            --annotation-file design.anno.json \\
            -o output.sv

    # Emit MLIR for inspection at each stage:
    firtool design.fir --format=fir --emit-mlir -o -
    # → shows the MLIR after each lowering pass

    # Run specific passes (circt-opt):
    circt-opt --firrtl-infer-widths \\
              --firrtl-expand-whens \\
              --firrtl-lower-types \\
              --convert-firrtl-to-hw \\
              --export-verilog \\
              design.mlir -o design.sv

  Key firtool flags:
    --format=fir      input is FIRRTL text (.fir)
    --format=mlir     input is MLIR with FIRRTL dialect
    --split-verilog   emit one .sv file per module (best for EDA)
    --preserve-values=named    keep named wires in output Verilog
    --add-mux-pragmas          add synthesis pragmas for mux selection
    --disable-all-randomization  no $random (for deterministic sim)
    --lower-memories=true      lower mem to register arrays (small mems)
    --repl-seq-mem             replace seq mem with SRAM macro hints

  Chisel → firtool → Verilog (the standard RISC-V core flow):
    sbt "runMain chisel3.stage.ChiselStage --target-dir output"
    firtool output/TopModule.fir \\
            --format=fir \\
            --split-verilog \\
            -o rtl/
    → rtl/TopModule.sv, rtl/Core.sv, rtl/FPU.sv, ...
    → feed to Yosys / Synopsys Design Compiler / Cadence Genus

  Real-world users of firtool:
    SiFive:           all their RISC-V IP (U74, U84, P650) via Chisel/firtool
    Western Digital:  SweRV RISC-V cores
    Google:           TPU v4 and onwards use Chisel/CIRCT components
    RISC-V community: rocket-chip, boom out-of-order CPU
    Esperanto:        ET-SoC-1 (1092 RISC-V cores) via Chisel
"""
print(FIRTOOL_GUIDE)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 3: CIRCT connection to each stack module
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 3 — CIRCT connections to the full compiler stack")
print("━" * 65)
print()

CONNECTIONS = """
  HOW CIRCT CONNECTS TO EACH MODULE IN THE STACK
  ════════════════════════════════════════════════════════════════

  ── MODULE 1: LLVM ────────────────────────────────────────────────────
    Shared infrastructure:
      CIRCT uses LLVM's build system (CMake), testing (FileCheck/lit),
      JIT (ORC JIT for software co-simulation), and data structures.

    Software co-simulation path:
      CIRCT hardware model → Arc dialect → LLVM dialect → LLVM IR
        → llc / ORC JIT → native x86/ARM binary
      This generates a fast software simulator for the hardware design.

    The Arc dialect converts clocked RTL to pure functions:
      seq.firreg → explicit state passed as function arguments
      comb.*     → LLVM arithmetic ops (llvm.add, llvm.mul, etc.)
      hw.module  → llvm.func (one function per module, called per cycle)

  ── MODULE 2: MLIR ────────────────────────────────────────────────────
    CIRCT IS MLIR. Key reused infrastructure:
      MLIRContext, Operation, Block, Region — identical
      Pass manager: pm.addPass(createCIRCTPass()) — same API
      Pattern rewriting: hardware canonicalisation reuses RewriteDriver
      Type system: hw::IntegerType extends mlir::Type
      FileCheck testing: CIRCT tests use identical lit/FileCheck setup

    MLIR dialects used inside CIRCT programs:
      func.func:   Calyx component functions
      arith.*:     arithmetic semantics in comb ops
      cf.*:        control flow in Arc simulation functions
      affine.*:    loop bounds in high-level synthesis experiments
      memref.*:    memory modelling in co-simulation

  ── MODULE 4: XLA ─────────────────────────────────────────────────────
    XLA → CIRCT path:
      Google uses Chisel to describe TPU hardware.
      Chisel generates FIRRTL, compiled by firtool (CIRCT).
      TPU matrix units, scalar cores, DMA engines: all via CIRCT.

    Experimental research path (OpenXLA project):
      XLA HLO → StableHLO → Calyx → firtool → FPGA Verilog
      Status: proof-of-concept, not production, as of 2024.

    Shared IR:
      StableHLO is an MLIR dialect; Calyx is an MLIR dialect.
      Both live in the same MLIR Context. One pass can lower
      StableHLO directly to Calyx without leaving MLIR.

  ── MODULE 5: TVM ─────────────────────────────────────────────────────
    TVM → CIRCT path (experimental Calyx backend):
      TVM VTA schedule → TIR → Calyx IR (via calyx-py frontend)
        → calyx-opt → hw/comb/seq → exportVerilog → .sv
        → Yosys synthesis → Xilinx FPGA bitstream

    Why use CIRCT over VTA's HLS path:
      CIRCT: formal semantics, open toolchain, MLIR-composable
      HLS (Vivado HLS): proprietary, unpredictable timing, no formal checks

  ── SUMMARY TABLE ─────────────────────────────────────────────────────

  ┌──────────────┬───────────────────────────────────────────────────────┐
  │  Module      │  Relationship to CIRCT                                │
  ├──────────────┼───────────────────────────────────────────────────────┤
  │  LLVM        │  Shared build/test infra; co-sim lowers to LLVM IR    │
  │  MLIR        │  CIRCT IS MLIR; reuses all infrastructure + dialects  │
  │  CIRCT       │  Hardware compilation: FIRRTL/Calyx → Verilog         │
  │  XLA         │  TPU hardware described in Chisel, compiled by CIRCT  │
  │  TVM         │  Experimental Calyx backend for FPGA acceleration     │
  └──────────────┴───────────────────────────────────────────────────────┘
"""
print(CONNECTIONS)

# ─────────────────────────────────────────────────────────────────────────
# SECTION 4: Resources
# ─────────────────────────────────────────────────────────────────────────
print("━" * 65)
print("  SECTION 4 — CIRCT resources")
print("━" * 65)
print()

print("  CIRCT PROJECT RESOURCES:")
print("  circt.llvm.org                    — official documentation")
print("  github.com/llvm/circt             — source code")
print("  github.com/llvm/circt/releases    — pre-built firtool binaries")
print("  calyxir.org                       — Calyx language reference")
print("  github.com/calyxir/calyx          — Calyx frontend + tools")
print("  github.com/chipsalliance/firrtl-spec  — FIRRTL specification")
print("  github.com/chipsalliance/chisel   — Chisel HDL (Scala/Chisel3)")
print("  github.com/ucb-bar/rocket-chip    — RISC-V via Chisel/CIRCT")
print("  github.com/google/xls             — Google's HLS (uses CIRCT ideas)")
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
        "display_name":  DISPLAY_NAME,
        "icon":          ICON,
        "subtitle":      SUBTITLE,
        "theory":        THEORY,
        "visual_html":   "",
        "visual_height": 400,
        "complexity":    None,
        "operations":    OPERATIONS,
    }