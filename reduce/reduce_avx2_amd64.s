//go:build amd64

#include "textflag.h"
#include "../simd/asm_macros.h"

TEXT ·SpeedOfLightTest(SB), NOSPLIT, $0
    VZEROUPPER
    RET

TEXT ·SpeedOfLightTest_Throughput(SB), NOSPLIT, $0-8
    MOVQ frame+0(FP), DI
    MOVQ FRAME_BUF0_PTR(DI), AX 
    MOVQ FRAME_DIM0(DI), BX    

unrolled_loop:
    CMPQ BX, $32
    JL tail

    VMOVAPD 0(AX), Y0
    VMOVAPD 32(AX), Y1
    VMOVAPD 64(AX), Y2
    VMOVAPD 96(AX), Y3
    VMOVAPD 128(AX), Y4
    VMOVAPD 160(AX), Y5
    VMOVAPD 192(AX), Y6
    VMOVAPD 224(AX), Y7

    ADDQ $256, AX
    SUBQ $32, BX
    JMP unrolled_loop

tail:
    // We ignore the tail for SoL calibration as we only 
    // care about the peak performance in the unrolled section.
    VZEROUPPER
    RET

TEXT ·VectorSumF64iF64o__AVX2(SB), NOSPLIT, $0-8
    // Load Frame Pointer from Go stack
    MOVQ frame+0(FP), DI

    // 1. Load Execution Bounds & Pointers from Frame
    MOVQ FRAME_BUF0_PTR(DI), AX  // AX = Buffers[0].Ptr (Input Data)
    MOVQ FRAME_DIM0(DI), BX      // BX = Dim[0] (Count N)
    MOVQ FRAME_RET0(DI), DX      // DX = Returns[0] (Output Scalar Pointer)

    // 2. Initialize 4 Accumulators (Y0-Y3) to 0.0
    VPXOR Y0, Y0, Y0
    VPXOR Y1, Y1, Y1
    VPXOR Y2, Y2, Y2
    VPXOR Y3, Y3, Y3

unrolled_loop:
    // Check if we have at least 16 elements left
    CMPQ BX, $16
    JL middle_loop

    // Prefetch next cache line (heuristic: 256 bytes ahead)
    PREFETCHT0 256(AX)

    // Load 16x float64s (4 Vectors)
    // NOTE: This assumes Unit Stride (contiguous memory)
    VMOVAPD 0(AX), Y4
    VMOVAPD 32(AX), Y5
    VMOVAPD 64(AX), Y6
    VMOVAPD 96(AX), Y7

    // Parallel Fused Accumulation
    VADDPD Y4, Y0, Y0
    VADDPD Y5, Y1, Y1
    VADDPD Y6, Y2, Y2
    VADDPD Y7, Y3, Y3

    // Advance Pointers
    ADDQ $128, AX  // 16 elements * 8 bytes
    SUBQ $16, BX
    JMP unrolled_loop

middle_loop:
    // Check if we have chunks of 4 left
    CMPQ BX, $4
    JL merge_accumulators

    VMOVAPD 0(AX), Y4
    VADDPD Y4, Y0, Y0

    ADDQ $32, AX
    SUBQ $4, BX
    JMP middle_loop

merge_accumulators:
    // Fold 4 accumulators into 1 (Y0)
    VADDPD Y1, Y0, Y0
    VADDPD Y3, Y2, Y2
    VADDPD Y2, Y0, Y0

tail:
    // Handle remaining 0-3 elements scalar-wise
    CMPQ BX, $0
    JE reduce

    MOVSD 0(AX), X4
    ADDSD X4, X0

    ADDQ $8, AX
    DECQ BX
    JMP tail

reduce:
    // Horizontal Reduction: Fold YMM (256-bit) -> XMM (64-bit scalar)
    // Y0 = [D, C, B, A]
    VEXTRACTF128 $1, Y0, X1 // X1 = [D, C]
    VADDPD X1, X0, X0       // X0 = [D+B, C+A]

    MOVHLPS X0, X1          // X1 = [?, D+B] (Move High to Low)
    ADDSD X1, X0            // X0 = (D+B) + (C+A)

    // Store Final Result to Frame.Returns[0]
    MOVSD X0, (DX)

    // Clean up AVX state
    VZEROUPPER
    RET

TEXT ·VectorSumF32iF64o__AVX2(SB), NOSPLIT, $0-8
    MOVQ frame+0(FP), DI

    // 1. Load Execution Bounds & Pointers
    MOVQ FRAME_BUF0_PTR(DI), AX  // AX = Data
    MOVQ FRAME_DIM0(DI), BX      // BX = N
    MOVQ FRAME_RET0(DI), DX      // DX = Return Ptr

    // 2. Initialize F64 Accumulators
    VPXOR Y0, Y0, Y0
    VPXOR Y1, Y1, Y1
    VPXOR Y2, Y2, Y2
    VPXOR Y3, Y3, Y3

unrolled_loop:
    CMPQ BX, $16
    JL middle_loop

    PREFETCHT0 128(AX) // Prefetch 128 bytes ahead

    // Convert 4x F32 (16 bytes) -> 4x F64 (32 bytes) per register
    VCVTPS2PD 0(AX), Y4
    VCVTPS2PD 16(AX), Y5
    VCVTPS2PD 32(AX), Y6
    VCVTPS2PD 48(AX), Y7

    VADDPD Y4, Y0, Y0
    VADDPD Y5, Y1, Y1
    VADDPD Y6, Y2, Y2
    VADDPD Y7, Y3, Y3

    ADDQ $64, AX   // 16 elements * 4 bytes = 64
    SUBQ $16, BX
    JMP unrolled_loop

middle_loop:
    CMPQ BX, $4
    JL merge_accumulators 

    VCVTPS2PD 0(AX), Y4
    VADDPD Y4, Y0, Y0

    ADDQ $16, AX  
    SUBQ $4, BX
    JMP middle_loop

merge_accumulators:
    VADDPD Y1, Y0, Y0
    VADDPD Y3, Y2, Y2
    VADDPD Y2, Y0, Y0

tail:
    CMPQ BX, $0
    JE reduce

    MOVSS 0(AX), X4
    VCVTSS2SD X4, X4, X4 
    ADDSD X4, X0 

    ADDQ $4, AX 
    DECQ BX
    JMP tail

reduce:
    VEXTRACTF128 $1, Y0, X1
    VADDPD X1, X0, X0
    MOVHLPS X0, X1
    ADDSD X1, X0

    MOVSD X0, (DX)
    VZEROUPPER
    RET

TEXT ·VectorSumF32iF32o__AVX2(SB), NOSPLIT, $0-8
    MOVQ frame+0(FP), DI

    // 1. Load Execution Bounds & Pointers
    MOVQ FRAME_BUF0_PTR(DI), AX  // AX = Data
    MOVQ FRAME_DIM0(DI), BX      // BX = N
    MOVQ FRAME_RET0(DI), DX      // DX = Return Ptr

    // 2. Initialize Accumulators
    VPXOR Y0, Y0, Y0
    VPXOR Y1, Y1, Y1
    VPXOR Y2, Y2, Y2
    VPXOR Y3, Y3, Y3

unrolled_loop:
    // Check if we have at least 32 elements (4 vectors of 8)
    CMPQ BX, $32
    JL middle_loop

    PREFETCHT0 256(AX)

    // Load 32x float32s (4 Vectors)
    VMOVAPS 0(AX), Y4
    VMOVAPS 32(AX), Y5
    VMOVAPS 64(AX), Y6
    VMOVAPS 96(AX), Y7

    // Parallel Fused Accumulation
    VADDPS Y4, Y0, Y0
    VADDPS Y5, Y1, Y1
    VADDPS Y6, Y2, Y2
    VADDPS Y7, Y3, Y3

    // Advance Pointers: 32 elements * 4 bytes = 128
    ADDQ $128, AX
    SUBQ $32, BX
    JMP unrolled_loop

middle_loop:
    CMPQ BX, $8
    JL merge_accumulators

    VMOVAPS 0(AX), Y4
    VADDPS Y4, Y0, Y0

    // Advance Pointers: 8 elements * 4 bytes = 32
    ADDQ $32, AX
    SUBQ $8, BX
    JMP middle_loop

merge_accumulators:
    // Fold 4 accumulators into 1 (Y0)
    VADDPS Y1, Y0, Y0
    VADDPS Y3, Y2, Y2
    VADDPS Y2, Y0, Y0

tail:
    CMPQ BX, $0
    JE reduce

    MOVSS 0(AX), X4
    ADDSS X4, X0

    ADDQ $4, AX
    DECQ BX
    JMP tail

reduce:    
    // 1. Fold YMM (256-bit) -> XMM (128-bit)
    // Y0 = [H, G, F, E, D, C, B, A]
    VEXTRACTF128 $1, Y0, X1  // X1 = [H, G, F, E]
    VADDPS X1, X0, X0        // X0 = [H+D, G+C, F+B, E+A]

    // 2. Fold High 64-bits -> Low 64-bits
    MOVHLPS X0, X1           // X1 = [?, ?, H+D, G+C] (Upper 2 floats move to lower 2)
    VADDPS X1, X0, X0        // X0 = [?, ?, (H+D)+(F+B), (G+C)+(E+A)]
    
    // 3. Fold Final 2 Floats
    // X0 low is now two partial sums. We need to add them.
    // Move 2nd float to 1st position (Shift Right 4 bytes)
    MOVSHDUP X0, X1          // X1[0] = X0[1] ((G+C)+(E+A))
    ADDSS X1, X0             // Scalar Add

    MOVSS X0, (DX)
    VZEROUPPER
    RET

TEXT ·DotProductF64F64o_AVX2(SB), NOSPLIT, $0-8
    MOVQ frame+0(FP), DI

    // 1. Load Context
    MOVQ FRAME_DIM0(DI), BX      // BX = N (Count)
    MOVQ FRAME_BUF0_PTR(DI), SI  // SI = Ptr A
    MOVQ FRAME_BUF1_PTR(DI), DX  // DX = Ptr B
    MOVQ FRAME_RET0(DI), CX      // CX = Return Ptr

    // 2. Initialize 8 Accumulators (Y0 - Y7)
    VPXOR Y0, Y0, Y0
    VPXOR Y1, Y1, Y1
    VPXOR Y2, Y2, Y2
    VPXOR Y3, Y3, Y3
    VPXOR Y4, Y4, Y4
    VPXOR Y5, Y5, Y5
    VPXOR Y6, Y6, Y6
    VPXOR Y7, Y7, Y7

unrolled_loop:
    CMPQ BX, $32
    JL middle_loop

    PREFETCHT0 512(SI) 
    PREFETCHT0 512(DX)

    // Block 0-3
    VMOVUPD 0(SI), Y8
    VFMADD231PD 0(DX), Y8, Y0  

    // Block 4-7
    VMOVUPD 32(SI), Y9
    VFMADD231PD 32(DX), Y9, Y1

    // Block 8-11
    VMOVUPD 64(SI), Y10
    VFMADD231PD 64(DX), Y10, Y2

    // Block 12-15
    VMOVUPD 96(SI), Y11
    VFMADD231PD 96(DX), Y11, Y3

    // Block 16-19
    VMOVUPD 128(SI), Y12
    VFMADD231PD 128(DX), Y12, Y4

    // Block 20-23
    VMOVUPD 160(SI), Y13
    VFMADD231PD 160(DX), Y13, Y5

    // Block 24-27
    VMOVUPD 192(SI), Y14
    VFMADD231PD 192(DX), Y14, Y6

    // Block 28-31
    VMOVUPD 224(SI), Y15
    VFMADD231PD 224(DX), Y15, Y7

    ADDQ $256, SI 
    ADDQ $256, DX
    SUBQ $32, BX
    JMP unrolled_loop

middle_loop:
    CMPQ BX, $4
    JL merge_accumulators

    VMOVUPD 0(SI), Y8
    VFMADD231PD 0(DX), Y8, Y0

    ADDQ $32, SI
    ADDQ $32, DX
    SUBQ $4, BX
    JMP middle_loop

merge_accumulators:
    // Fold partial sums Y1..Y7 into Y0
    VADDPD Y1, Y0, Y0
    VADDPD Y3, Y2, Y2
    VADDPD Y5, Y4, Y4
    VADDPD Y7, Y6, Y6
    
    VADDPD Y2, Y0, Y0
    VADDPD Y6, Y4, Y4
    
    VADDPD Y4, Y0, Y0 // Y0 now holds [SumA, SumB, SumC, SumD]

    VEXTRACTF128 $1, Y0, X1  // X1 = [D, C]
    VADDPD X1, X0, X0        // X0 = [D+B, C+A] (Result in lower 128 bits)
    
    MOVHLPS X0, X1           // X1 = [?, D+B]
    ADDSD X1, X0             // X0 = (D+B) + (C+A) (Final Scalar in X0)

tail:
    CMPQ BX, $0
    JE done

    MOVSD 0(SI), X8   // Load Scalar A
    MOVSD 0(DX), X9   // Load Scalar B
    
    VFMADD231SD X9, X8, X0 

    ADDQ $8, SI
    ADDQ $8, DX
    DECQ BX
    JMP tail

done:
    MOVSD X0, (CX)
    VZEROUPPER
    RET
