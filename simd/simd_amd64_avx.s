//go:build amd64

#include "textflag.h"

// AVX implementations (256-bit, but fewer instruction variants than AVX2)
// These are fallbacks when AVX2 is not available but AVX is

// Most AVX implementations are similar to AVX2, but we use AVX-only instructions
// For simplicity, we can reuse AVX2 code since AVX2 is a superset of AVX
// However, we'll create separate implementations to be explicit about instruction sets

// Note: AVX implementations use the same instruction set as AVX2 for these operations
// The main difference is that AVX2 adds more instruction variants, but for basic
// arithmetic operations (ADD, MUL, etc.), AVX is sufficient.
// We'll create stub implementations that call the same logic but are separate symbols.

// For now, we'll implement simplified versions that work with AVX
// In practice, these could be identical to AVX2 for these operations

// blazeReduceVectorSumSquaredF64AVX - same as AVX2 for this operation
TEXT ·blazeReduceVectorSumSquaredF64AVX(SB), NOSPLIT, $0-32
    MOVQ data+0(FP), SI
    MOVQ size+8(FP), DX
    MOVQ capacity+16(FP), CX

    VXORPS Y0, Y0, Y0
    MOVQ CX, AX
    SHRQ $2, AX
    JZ   remainder_loop_avx64

simd_loop_avx64:
    VMOVUPD (SI), Y1
    VMULPD Y1, Y1, Y1
    VADDPD Y1, Y0, Y0

    MOVQ DX, DI
    SHLQ $2, DI
    ADDQ DI, SI
    DECQ AX
    JNZ  simd_loop_avx64

    VEXTRACTF128 $1, Y0, X1
    VADDPD X0, X1, X0
    VHADDPD X0, X0, X0
    VMOVSD X0, ret+24(FP)

remainder_loop_avx64:
    MOVQ CX, AX
    ANDQ $3, AX
    JZ   done_avx64
    VMOVSD ret+24(FP), X0

remainder_iter_avx64:
    VMOVSD (SI), X1
    VMULSD X1, X1, X1
    VADDSD X1, X0, X0
    ADDQ DX, SI
    DECQ AX
    JNZ  remainder_iter_avx64
    VMOVSD X0, ret+24(FP)

done_avx64:
    VZEROUPPER
    RET

// blazeReduceVectorSumSquaredF32AVX
TEXT ·blazeReduceVectorSumSquaredF32AVX(SB), NOSPLIT, $0-24
    MOVQ data+0(FP), SI
    MOVQ size+8(FP), DX
    MOVQ capacity+16(FP), CX

    VXORPS Y0, Y0, Y0
    MOVQ CX, AX
    SHRQ $3, AX
    JZ   remainder_loop_avx32

simd_loop_avx32:
    VMOVUPS (SI), Y1
    VMULPS Y1, Y1, Y1
    VADDPS Y1, Y0, Y0

    MOVQ DX, DI
    SHLQ $3, DI
    ADDQ DI, SI
    DECQ AX
    JNZ  simd_loop_avx32

    VEXTRACTF128 $1, Y0, X1
    VADDPS X0, X1, X0
    VHADDPS X0, X0, X0
    VHADDPS X0, X0, X0
    VMOVSS X0, ret+24(FP)

remainder_loop_avx32:
    MOVQ CX, AX
    ANDQ $7, AX
    JZ   done_avx32
    VMOVSS ret+24(FP), X0

remainder_iter_avx32:
    VMOVSS (SI), X1
    VMULSS X1, X1, X1
    VADDSS X1, X0, X0
    ADDQ DX, SI
    DECQ AX
    JNZ  remainder_iter_avx32
    VMOVSS X0, ret+24(FP)

done_avx32:
    VZEROUPPER
    RET

// blazeReduceDotProductF64AVX
TEXT ·blazeReduceDotProductF64AVX(SB), NOSPLIT, $0-48
    MOVQ aData+0(FP), SI
    MOVQ bData+8(FP), DI
    MOVQ aSize+16(FP), DX
    MOVQ bSize+24(FP), R8
    MOVQ capacity+32(FP), CX

    VXORPS Y0, Y0, Y0
    MOVQ CX, AX
    SHRQ $2, AX
    JZ   remainder_loop_dot_avx64

simd_loop_dot_avx64:
    VMOVUPD (SI), Y1
    VMOVUPD (DI), Y2
    VMULPD Y2, Y1, Y1
    VADDPD Y1, Y0, Y0

    MOVQ DX, R9
    SHLQ $2, R9
    ADDQ R9, SI
    MOVQ R8, R9
    SHLQ $2, R9
    ADDQ R9, DI
    DECQ AX
    JNZ  simd_loop_dot_avx64

    VEXTRACTF128 $1, Y0, X1
    VADDPD X0, X1, X0
    VHADDPD X0, X0, X0
    VMOVSD X0, ret+40(FP)

remainder_loop_dot_avx64:
    MOVQ CX, AX
    ANDQ $3, AX
    JZ   done_dot_avx64
    VMOVSD ret+40(FP), X0

remainder_iter_dot_avx64:
    VMOVSD (SI), X1
    VMOVSD (DI), X2
    VMULSD X2, X1, X1
    VADDSD X1, X0, X0
    ADDQ DX, SI
    ADDQ R8, DI
    DECQ AX
    JNZ  remainder_iter_dot_avx64
    VMOVSD X0, ret+40(FP)

done_dot_avx64:
    VZEROUPPER
    RET

// blazeReduceDotProductF32AVX
TEXT ·blazeReduceDotProductF32AVX(SB), NOSPLIT, $0-40
    MOVQ aData+0(FP), SI
    MOVQ bData+8(FP), DI
    MOVQ aSize+16(FP), DX
    MOVQ bSize+24(FP), R8
    MOVQ capacity+32(FP), CX

    VXORPS Y0, Y0, Y0
    MOVQ CX, AX
    SHRQ $3, AX
    JZ   remainder_loop_dot_avx32

simd_loop_dot_avx32:
    VMOVUPS (SI), Y1
    VMOVUPS (DI), Y2
    VMULPS Y2, Y1, Y1
    VADDPS Y1, Y0, Y0

    MOVQ DX, R9
    SHLQ $3, R9
    ADDQ R9, SI
    MOVQ R8, R9
    SHLQ $3, R9
    ADDQ R9, DI
    DECQ AX
    JNZ  simd_loop_dot_avx32

    VEXTRACTF128 $1, Y0, X1
    VADDPS X0, X1, X0
    VHADDPS X0, X0, X0
    VHADDPS X0, X0, X0
    VMOVSS X0, ret+40(FP)

remainder_loop_dot_avx32:
    MOVQ CX, AX
    ANDQ $7, AX
    JZ   done_dot_avx32
    VMOVSS ret+40(FP), X0

remainder_iter_dot_avx32:
    VMOVSS (SI), X1
    VMOVSS (DI), X2
    VMULSS X2, X1, X1
    VADDSS X1, X0, X0
    ADDQ DX, SI
    ADDQ R8, DI
    DECQ AX
    JNZ  remainder_iter_dot_avx32
    VMOVSS X0, ret+40(FP)

done_dot_avx32:
    VZEROUPPER
    RET

// Elementwise and scalar operations - similar pattern, simplified for AVX
// blazeElementWiseVectorAddF32AVX
TEXT ·blazeElementWiseVectorAddF32AVX(SB), NOSPLIT, $0-56
    MOVQ aData+0(FP), SI
    MOVQ bData+8(FP), DI
    MOVQ dstData+16(FP), R9
    MOVQ aSize+24(FP), DX
    MOVQ bSize+32(FP), R8
    MOVQ dstSize+40(FP), R10
    MOVQ capacity+48(FP), CX

    MOVQ CX, AX
    SHRQ $3, AX
    JZ   remainder_loop_add_avx32

simd_loop_add_avx32:
    VMOVUPS (SI), Y0
    VMOVUPS (DI), Y1
    VADDPS Y1, Y0, Y0
    VMOVUPS Y0, (R9)

    MOVQ DX, R11
    SHLQ $3, R11
    ADDQ R11, SI
    MOVQ R8, R11
    SHLQ $3, R11
    ADDQ R11, DI
    MOVQ R10, R11
    SHLQ $3, R11
    ADDQ R11, R9
    DECQ AX
    JNZ  simd_loop_add_avx32

remainder_loop_add_avx32:
    MOVQ CX, AX
    ANDQ $7, AX
    JZ   done_add_avx32

remainder_iter_add_avx32:
    VMOVSS (SI), X0
    VMOVSS (DI), X1
    VADDSS X1, X0, X0
    VMOVSS X0, (R9)
    ADDQ DX, SI
    ADDQ R8, DI
    ADDQ R10, R9
    DECQ AX
    JNZ  remainder_iter_add_avx32

done_add_avx32:
    VZEROUPPER
    RET

// blazeElementWiseVectorAddF64AVX
TEXT ·blazeElementWiseVectorAddF64AVX(SB), NOSPLIT, $0-56
    MOVQ aData+0(FP), SI
    MOVQ bData+8(FP), DI
    MOVQ dstData+16(FP), R9
    MOVQ aSize+24(FP), DX
    MOVQ bSize+32(FP), R8
    MOVQ dstSize+40(FP), R10
    MOVQ capacity+48(FP), CX

    MOVQ CX, AX
    SHRQ $2, AX
    JZ   remainder_loop_add_avx64

simd_loop_add_avx64:
    VMOVUPD (SI), Y0
    VMOVUPD (DI), Y1
    VADDPD Y1, Y0, Y0
    VMOVUPD Y0, (R9)

    MOVQ DX, R11
    SHLQ $2, R11
    ADDQ R11, SI
    MOVQ R8, R11
    SHLQ $2, R11
    ADDQ R11, DI
    MOVQ R10, R11
    SHLQ $2, R11
    ADDQ R11, R9
    DECQ AX
    JNZ  simd_loop_add_avx64

remainder_loop_add_avx64:
    MOVQ CX, AX
    ANDQ $3, AX
    JZ   done_add_avx64

remainder_iter_add_avx64:
    VMOVSD (SI), X0
    VMOVSD (DI), X1
    VADDSD X1, X0, X0
    VMOVSD X0, (R9)
    ADDQ DX, SI
    ADDQ R8, DI
    ADDQ R10, R9
    DECQ AX
    JNZ  remainder_iter_add_avx64

done_add_avx64:
    VZEROUPPER
    RET

// blazeElementWiseVectorMultiplyF32AVX
TEXT ·blazeElementWiseVectorMultiplyF32AVX(SB), NOSPLIT, $0-56
    MOVQ aData+0(FP), SI
    MOVQ bData+8(FP), DI
    MOVQ dstData+16(FP), R9
    MOVQ aSize+24(FP), DX
    MOVQ bSize+32(FP), R8
    MOVQ dstSize+40(FP), R10
    MOVQ capacity+48(FP), CX

    MOVQ CX, AX
    SHRQ $3, AX
    JZ   remainder_loop_mul_avx32

simd_loop_mul_avx32:
    VMOVUPS (SI), Y0
    VMOVUPS (DI), Y1
    VMULPS Y1, Y0, Y0
    VMOVUPS Y0, (R9)

    MOVQ DX, R11
    SHLQ $3, R11
    ADDQ R11, SI
    MOVQ R8, R11
    SHLQ $3, R11
    ADDQ R11, DI
    MOVQ R10, R11
    SHLQ $3, R11
    ADDQ R11, R9
    DECQ AX
    JNZ  simd_loop_mul_avx32

remainder_loop_mul_avx32:
    MOVQ CX, AX
    ANDQ $7, AX
    JZ   done_mul_avx32

remainder_iter_mul_avx32:
    VMOVSS (SI), X0
    VMOVSS (DI), X1
    VMULSS X1, X0, X0
    VMOVSS X0, (R9)
    ADDQ DX, SI
    ADDQ R8, DI
    ADDQ R10, R9
    DECQ AX
    JNZ  remainder_iter_mul_avx32

done_mul_avx32:
    VZEROUPPER
    RET

// blazeElementWiseVectorMultiplyF64AVX
TEXT ·blazeElementWiseVectorMultiplyF64AVX(SB), NOSPLIT, $0-56
    MOVQ aData+0(FP), SI
    MOVQ bData+8(FP), DI
    MOVQ dstData+16(FP), R9
    MOVQ aSize+24(FP), DX
    MOVQ bSize+32(FP), R8
    MOVQ dstSize+40(FP), R10
    MOVQ capacity+48(FP), CX

    MOVQ CX, AX
    SHRQ $2, AX
    JZ   remainder_loop_mul_avx64

simd_loop_mul_avx64:
    VMOVUPD (SI), Y0
    VMOVUPD (DI), Y1
    VMULPD Y1, Y0, Y0
    VMOVUPD Y0, (R9)

    MOVQ DX, R11
    SHLQ $2, R11
    ADDQ R11, SI
    MOVQ R8, R11
    SHLQ $2, R11
    ADDQ R11, DI
    MOVQ R10, R11
    SHLQ $2, R11
    ADDQ R11, R9
    DECQ AX
    JNZ  simd_loop_mul_avx64

remainder_loop_mul_avx64:
    MOVQ CX, AX
    ANDQ $3, AX
    JZ   done_mul_avx64

remainder_iter_mul_avx64:
    VMOVSD (SI), X0
    VMOVSD (DI), X1
    VMULSD X1, X0, X0
    VMOVSD X0, (R9)
    ADDQ DX, SI
    ADDQ R8, DI
    ADDQ R10, R9
    DECQ AX
    JNZ  remainder_iter_mul_avx64

done_mul_avx64:
    VZEROUPPER
    RET

// blazeScalarVectorMultiplyF32AVX
TEXT ·blazeScalarVectorMultiplyF32AVX(SB), NOSPLIT, $0-48
    MOVQ srcData+0(FP), SI
    MOVQ dstData+8(FP), DI
    MOVQ srcSize+16(FP), DX
    MOVQ dstSize+24(FP), R8
    MOVSS scalar+32(FP), X15
    MOVQ capacity+40(FP), CX

    VBROADCASTSS X15, Y15
    MOVQ CX, AX
    SHRQ $3, AX
    JZ   remainder_loop_scalar_avx32

simd_loop_scalar_avx32:
    VMOVUPS (SI), Y0
    VMULPS Y15, Y0, Y0
    VMOVUPS Y0, (DI)

    MOVQ DX, R9
    SHLQ $3, R9
    ADDQ R9, SI
    MOVQ R8, R9
    SHLQ $3, R9
    ADDQ R9, DI
    DECQ AX
    JNZ  simd_loop_scalar_avx32

remainder_loop_scalar_avx32:
    MOVQ CX, AX
    ANDQ $7, AX
    JZ   done_scalar_avx32

remainder_iter_scalar_avx32:
    VMOVSS (SI), X0
    VMULSS X15, X0, X0
    VMOVSS X0, (DI)
    ADDQ DX, SI
    ADDQ R8, DI
    DECQ AX
    JNZ  remainder_iter_scalar_avx32

done_scalar_avx32:
    VZEROUPPER
    RET

// blazeScalarVectorMultiplyF64AVX
TEXT ·blazeScalarVectorMultiplyF64AVX(SB), NOSPLIT, $0-48
    MOVQ srcData+0(FP), SI
    MOVQ dstData+8(FP), DI
    MOVQ srcSize+16(FP), DX
    MOVQ dstSize+24(FP), R8
    MOVSD scalar+32(FP), X15
    MOVQ capacity+40(FP), CX

    VBROADCASTSD X15, Y15
    MOVQ CX, AX
    SHRQ $2, AX
    JZ   remainder_loop_scalar_avx64

simd_loop_scalar_avx64:
    VMOVUPD (SI), Y0
    VMULPD Y15, Y0, Y0
    VMOVUPD Y0, (DI)

    MOVQ DX, R9
    SHLQ $2, R9
    ADDQ R9, SI
    MOVQ R8, R9
    SHLQ $2, R9
    ADDQ R9, DI
    DECQ AX
    JNZ  simd_loop_scalar_avx64

remainder_loop_scalar_avx64:
    MOVQ CX, AX
    ANDQ $3, AX
    JZ   done_scalar_avx64

remainder_iter_scalar_avx64:
    VMOVSD (SI), X0
    VMULSD X15, X0, X0
    VMOVSD X0, (DI)
    ADDQ DX, SI
    ADDQ R8, DI
    DECQ AX
    JNZ  remainder_iter_scalar_avx64

done_scalar_avx64:
    VZEROUPPER
    RET

// blazeReduceVectorSumF64AVX
TEXT ·blazeReduceVectorSumF64AVX(SB), NOSPLIT, $0-32
    MOVQ data+0(FP), SI
    MOVQ size+8(FP), DX
    MOVQ capacity+16(FP), CX

    VXORPS Y0, Y0, Y0
    MOVQ CX, AX
    SHRQ $2, AX
    JZ   remainder_loop_sum_avx64

simd_loop_sum_avx64:
    VMOVUPD (SI), Y1
    VADDPD Y1, Y0, Y0

    MOVQ DX, DI
    SHLQ $2, DI
    ADDQ DI, SI
    DECQ AX
    JNZ  simd_loop_sum_avx64

    VEXTRACTF128 $1, Y0, X1
    VADDPD X0, X1, X0
    VHADDPD X0, X0, X0
    VMOVSD X0, ret+24(FP)

remainder_loop_sum_avx64:
    MOVQ CX, AX
    ANDQ $3, AX
    JZ   done_sum_avx64
    VMOVSD ret+24(FP), X0

remainder_iter_sum_avx64:
    VMOVSD (SI), X1
    VADDSD X1, X0, X0
    ADDQ DX, SI
    DECQ AX
    JNZ  remainder_iter_sum_avx64
    VMOVSD X0, ret+24(FP)

done_sum_avx64:
    VZEROUPPER
    RET

// blazeReduceVectorSumF32AVX
TEXT ·blazeReduceVectorSumF32AVX(SB), NOSPLIT, $0-24
    MOVQ data+0(FP), SI
    MOVQ size+8(FP), DX
    MOVQ capacity+16(FP), CX

    VXORPS Y0, Y0, Y0
    MOVQ CX, AX
    SHRQ $3, AX
    JZ   remainder_loop_sum_avx32

simd_loop_sum_avx32:
    VMOVUPS (SI), Y1
    VADDPS Y1, Y0, Y0

    MOVQ DX, DI
    SHLQ $3, DI
    ADDQ DI, SI
    DECQ AX
    JNZ  simd_loop_sum_avx32

    VEXTRACTF128 $1, Y0, X1
    VADDPS X0, X1, X0
    VHADDPS X0, X0, X0
    VHADDPS X0, X0, X0
    VMOVSS X0, ret+24(FP)

remainder_loop_sum_avx32:
    MOVQ CX, AX
    ANDQ $7, AX
    JZ   done_sum_avx32
    VMOVSS ret+24(FP), X0

remainder_iter_sum_avx32:
    VMOVSS (SI), X1
    VADDSS X1, X0, X0
    ADDQ DX, SI
    DECQ AX
    JNZ  remainder_iter_sum_avx32
    VMOVSS X0, ret+24(FP)

done_sum_avx32:
    VZEROUPPER
    RET

