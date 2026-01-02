//go:build amd64

#include "textflag.h"
#include "../simd/asm_macros.h"

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
