//go:build amd64

#include "textflag.h"
#include "../simd/asm_macros.h"

TEXT ·ScalarVectorDivideF64iF64o_AVX2(SB), NOSPLIT, $0-8
    MOVQ frame+0(FP), DI

    // 1. Load Pointers & Bounds
    MOVQ FRAME_BUF0_PTR(DI), SI   // SI = Source Ptr (F64)
    MOVQ FRAME_BUF1_PTR(DI), DX   // DX = Dest Ptr (F64)
    MOVQ FRAME_DIM0(DI), BX       // BX = N (Count)

    // 2. Load and Broadcast Scalar (Divisor)
    // Params[0] contains the float64 scalar bits
    MOVQ FRAME_PARAM0(DI), X0     // Move scalar bits to X0 (low qword)
    VBROADCASTSD X0, Y0           // Broadcast scalar to all 4 lanes of Y0

unrolled_loop:
    // -------------------------------------------------------------------------
    // PRIMARY LOOP: 16 Elements (4x YMM) per Iteration
    // -------------------------------------------------------------------------
    CMPQ BX, $16
    JL middle_loop

    PREFETCHT0 384(SI)  // Prefetch Source (Heuristic: 3 cache lines ahead)
    PREFETCHT0 384(DX)  // Prefetch Dest

    // Load 16x float64s
    VMOVUPD 0(SI), Y1
    VMOVUPD 32(SI), Y2
    VMOVUPD 64(SI), Y3
    VMOVUPD 96(SI), Y4

    // Perform Division: Dst = Src / Scalar
    // Y0 holds the broadcasted scalar
    VDIVPD Y0, Y1, Y1
    VDIVPD Y0, Y2, Y2
    VDIVPD Y0, Y3, Y3
    VDIVPD Y0, Y4, Y4

    // Store Results
    VMOVUPD Y1, 0(DX)
    VMOVUPD Y2, 32(DX)
    VMOVUPD Y3, 64(DX)
    VMOVUPD Y4, 96(DX)

    ADDQ $128, SI       // 16 * 8 bytes
    ADDQ $128, DX
    SUBQ $16, BX
    JMP unrolled_loop

middle_loop:
    // -------------------------------------------------------------------------
    // SECONDARY LOOP: 4 Elements (1x YMM) per Iteration
    // -------------------------------------------------------------------------
    CMPQ BX, $4
    JL tail

    VMOVUPD 0(SI), Y1
    VDIVPD Y0, Y1, Y1
    VMOVUPD Y1, 0(DX)

    ADDQ $32, SI
    ADDQ $32, DX
    SUBQ $4, BX
    JMP middle_loop

tail:
    // -------------------------------------------------------------------------
    // TAIL LOOP: Scalar Fallback
    // -------------------------------------------------------------------------
    CMPQ BX, $0
    JE done

    MOVSD 0(SI), X1
    VDIVSD X0, X1, X1   // Scalar Divide: X1[0] = X1[0] / X0[0]
    MOVSD X1, 0(DX)

    ADDQ $8, SI
    ADDQ $8, DX
    DECQ BX
    JMP tail

done:
    VZEROUPPER
    RET


TEXT ·ScalarVectorDivideF32iF64o_AVX2(SB), NOSPLIT, $0-8
    MOVQ frame+0(FP), DI

    // 1. Load Pointers & Bounds
    MOVQ FRAME_BUF0_PTR(DI), SI   // SI = Source Ptr (F32)
    MOVQ FRAME_BUF1_PTR(DI), DX   // DX = Dest Ptr (F64)
    MOVQ FRAME_DIM0(DI), BX       // BX = N (Count)

    // 2. Load and Broadcast Scalar
    MOVQ FRAME_PARAM0(DI), X0     
    VBROADCASTSD X0, Y0           // Y0 = [S, S, S, S] (float64)

unrolled_loop:
    // -------------------------------------------------------------------------
    // PRIMARY LOOP: 16 Elements per Iteration
    // Note: Input stride is 4 bytes (F32), Output is 8 bytes (F64)
    // -------------------------------------------------------------------------
    CMPQ BX, $16
    JL middle_loop

    PREFETCHT0 192(SI)  // Prefetch Source (16 * 4 = 64 bytes -> 3 lines ahead ~192)
    PREFETCHT0 384(DX)  // Prefetch Dest

    // Block 1: Elements 0-3
    VCVTPS2PD 0(SI), Y1 // Load 4x F32, Expand to 4x F64
    VDIVPD Y0, Y1, Y1   // Divide
    VMOVUPD Y1, 0(DX)   // Store 4x F64

    // Block 2: Elements 4-7
    VCVTPS2PD 16(SI), Y2
    VDIVPD Y0, Y2, Y2
    VMOVUPD Y2, 32(DX)

    // Block 3: Elements 8-11
    VCVTPS2PD 32(SI), Y3
    VDIVPD Y0, Y3, Y3
    VMOVUPD Y3, 64(DX)

    // Block 4: Elements 12-15
    VCVTPS2PD 48(SI), Y4
    VDIVPD Y0, Y4, Y4
    VMOVUPD Y4, 96(DX)

    ADDQ $64, SI        // 16 * 4 bytes
    ADDQ $128, DX       // 16 * 8 bytes
    SUBQ $16, BX
    JMP unrolled_loop

middle_loop:
    // -------------------------------------------------------------------------
    // SECONDARY LOOP: 4 Elements per Iteration
    // -------------------------------------------------------------------------
    CMPQ BX, $4
    JL tail

    VCVTPS2PD 0(SI), Y1
    VDIVPD Y0, Y1, Y1
    VMOVUPD Y1, 0(DX)

    ADDQ $16, SI        // 4 * 4
    ADDQ $32, DX        // 4 * 8
    SUBQ $4, BX
    JMP middle_loop

tail:
    // -------------------------------------------------------------------------
    // TAIL LOOP: Scalar Fallback
    // -------------------------------------------------------------------------
    CMPQ BX, $0
    JE done

    MOVSS 0(SI), X1     // Load float32
    VCVTSS2SD X1, X1, X1 // Convert to float64
    VDIVSD X0, X1, X1   // Divide
    MOVSD X1, 0(DX)     // Store float64

    ADDQ $4, SI
    ADDQ $8, DX
    DECQ BX
    JMP tail

done:
    VZEROUPPER
    RET

TEXT ·ScalarVectorMultiplyF64iF64o_AVX2(SB), NOSPLIT, $0-8
    MOVQ frame+0(FP), DI

    // 1. Load Pointers & Bounds
    MOVQ FRAME_BUF0_PTR(DI), SI   // SI = Source Ptr (F64)
    MOVQ FRAME_BUF1_PTR(DI), DX   // DX = Dest Ptr (F64)
    MOVQ FRAME_DIM0(DI), BX       // BX = N (Count)

    // 2. Load and Broadcast Scalar
    MOVQ FRAME_PARAM0(DI), X0     // Move scalar bits to X0 (low qword)
    VBROADCASTSD X0, Y0           // Broadcast scalar to all 4 lanes of Y0

unrolled_loop:
    // -------------------------------------------------------------------------
    // PRIMARY LOOP: 16 Elements (4x YMM) per Iteration
    // -------------------------------------------------------------------------
    CMPQ BX, $16
    JL middle_loop

    PREFETCHT0 384(SI)  // Prefetch Source
    PREFETCHT0 384(DX)  // Prefetch Dest

    // Load 16x float64s
    VMOVUPD 0(SI), Y1
    VMOVUPD 32(SI), Y2
    VMOVUPD 64(SI), Y3
    VMOVUPD 96(SI), Y4

    // Perform Multiplication: Dst = Src * Scalar
    VMULPD Y0, Y1, Y1
    VMULPD Y0, Y2, Y2
    VMULPD Y0, Y3, Y3
    VMULPD Y0, Y4, Y4

    // Store Results
    VMOVUPD Y1, 0(DX)
    VMOVUPD Y2, 32(DX)
    VMOVUPD Y3, 64(DX)
    VMOVUPD Y4, 96(DX)

    ADDQ $128, SI       // 16 * 8 bytes
    ADDQ $128, DX
    SUBQ $16, BX
    JMP unrolled_loop

middle_loop:
    // -------------------------------------------------------------------------
    // SECONDARY LOOP: 4 Elements (1x YMM) per Iteration
    // -------------------------------------------------------------------------
    CMPQ BX, $4
    JL tail

    VMOVUPD 0(SI), Y1
    VMULPD Y0, Y1, Y1
    VMOVUPD Y1, 0(DX)

    ADDQ $32, SI
    ADDQ $32, DX
    SUBQ $4, BX
    JMP middle_loop

tail:
    // -------------------------------------------------------------------------
    // TAIL LOOP: Scalar Fallback
    // -------------------------------------------------------------------------
    CMPQ BX, $0
    JE done

    MOVSD 0(SI), X1
    VMULSD X0, X1, X1   // Scalar Multiply
    MOVSD X1, 0(DX)

    ADDQ $8, SI
    ADDQ $8, DX
    DECQ BX
    JMP tail

done:
    VZEROUPPER
    RET

TEXT ·ScalarVectorMultiplyF32iF64o_AVX2(SB), NOSPLIT, $0-8
    MOVQ frame+0(FP), DI

    // 1. Load Pointers & Bounds
    MOVQ FRAME_BUF0_PTR(DI), SI   // SI = Source Ptr (F32)
    MOVQ FRAME_BUF1_PTR(DI), DX   // DX = Dest Ptr (F64)
    MOVQ FRAME_DIM0(DI), BX       // BX = N (Count)

    // 2. Load and Broadcast Scalar
    MOVQ FRAME_PARAM0(DI), X0     
    VBROADCASTSD X0, Y0           // Y0 = [S, S, S, S] (float64)

unrolled_loop:
    // -------------------------------------------------------------------------
    // PRIMARY LOOP: 16 Elements per Iteration
    // Note: Input stride is 4 bytes (F32), Output is 8 bytes (F64)
    // -------------------------------------------------------------------------
    CMPQ BX, $16
    JL middle_loop

    PREFETCHT0 192(SI)  // Prefetch Source (16 * 4 = 64 bytes)
    PREFETCHT0 384(DX)  // Prefetch Dest

    // Block 1: Elements 0-3
    VCVTPS2PD 0(SI), Y1 // Load 4x F32, Expand to 4x F64
    VMULPD Y0, Y1, Y1   // Multiply
    VMOVUPD Y1, 0(DX)   // Store 4x F64

    // Block 2: Elements 4-7
    VCVTPS2PD 16(SI), Y2
    VMULPD Y0, Y2, Y2
    VMOVUPD Y2, 32(DX)

    // Block 3: Elements 8-11
    VCVTPS2PD 32(SI), Y3
    VMULPD Y0, Y3, Y3
    VMOVUPD Y3, 64(DX)

    // Block 4: Elements 12-15
    VCVTPS2PD 48(SI), Y4
    VMULPD Y0, Y4, Y4
    VMOVUPD Y4, 96(DX)

    ADDQ $64, SI        // 16 * 4 bytes
    ADDQ $128, DX       // 16 * 8 bytes
    SUBQ $16, BX
    JMP unrolled_loop

middle_loop:
    // -------------------------------------------------------------------------
    // SECONDARY LOOP: 4 Elements per Iteration
    // -------------------------------------------------------------------------
    CMPQ BX, $4
    JL tail

    VCVTPS2PD 0(SI), Y1
    VMULPD Y0, Y1, Y1
    VMOVUPD Y1, 0(DX)

    ADDQ $16, SI        // 4 * 4
    ADDQ $32, DX        // 4 * 8
    SUBQ $4, BX
    JMP middle_loop

tail:
    // -------------------------------------------------------------------------
    // TAIL LOOP: Scalar Fallback
    // -------------------------------------------------------------------------
    CMPQ BX, $0
    JE done

    MOVSS 0(SI), X1     // Load float32
    VCVTSS2SD X1, X1, X1 // Convert to float64
    VMULSD X0, X1, X1   // Multiply
    MOVSD X1, 0(DX)     // Store float64

    ADDQ $4, SI
    ADDQ $8, DX
    DECQ BX
    JMP tail

done:
    VZEROUPPER
    RET
