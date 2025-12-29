//go:build amd64

#include "textflag.h"

// SSE4.1 implementations (128-bit XMM registers)
// These are fallbacks when AVX is not available

// blazeReduceVectorSumSquaredF64SSE41 - 2 float64 per XMM register
TEXT ·blazeReduceVectorSumSquaredF64SSE41(SB), NOSPLIT, $0-32
    MOVQ data+0(FP), SI
    MOVQ size+8(FP), DX
    MOVQ capacity+16(FP), CX

    XORPS X0, X0             // X0 = [0, 0] (accumulator)
    MOVQ CX, AX
    SHRQ $1, AX              // AX = capacity / 2
    JZ   remainder_loop_sse64

simd_loop_sse64:
    MOVUPD (SI), X1          // X1 = [v0, v1]
    MULPD X1, X1             // X1 = [v0², v1²]
    ADDPD X1, X0              // X0 += X1

    MOVQ DX, DI
    SHLQ $1, DI               // DI = size * 2
    ADDQ DI, SI
    DECQ AX
    JNZ  simd_loop_sse64

    // Horizontal reduction: haddpd X0, X0 = [X0[0]+X0[1], X0[0]+X0[1]]
    HADDPD X0, X0             // X0 = [sum, sum]
    MOVSD X0, ret+24(FP)

remainder_loop_sse64:
    MOVQ CX, AX
    ANDQ $1, AX               // AX = capacity % 2
    JZ   done_sse64
    MOVSD ret+24(FP), X0

remainder_iter_sse64:
    MOVSD (SI), X1
    MULSD X1, X1
    ADDSD X1, X0
    ADDQ DX, SI
    DECQ AX
    JNZ  remainder_iter_sse64
    MOVSD X0, ret+24(FP)

done_sse64:
    RET

// blazeReduceVectorSumSquaredF32SSE41 - 4 float32 per XMM register
TEXT ·blazeReduceVectorSumSquaredF32SSE41(SB), NOSPLIT, $0-24
    MOVQ data+0(FP), SI
    MOVQ size+8(FP), DX
    MOVQ capacity+16(FP), CX

    XORPS X0, X0              // X0 = [0, 0, 0, 0]
    MOVQ CX, AX
    SHRQ $2, AX               // AX = capacity / 4
    JZ   remainder_loop_sse32

simd_loop_sse32:
    MOVUPS (SI), X1           // X1 = [v0, v1, v2, v3]
    MULPS X1, X1              // X1 = [v0², v1², v2², v3²]
    ADDPS X1, X0              // X0 += X1

    MOVQ DX, DI
    SHLQ $2, DI               // DI = size * 4
    ADDQ DI, SI
    DECQ AX
    JNZ  simd_loop_sse32

    // Horizontal reduction
    HADDPS X0, X0             // X0 = [sum0+sum1, sum2+sum3, ...]
    HADDPS X0, X0             // X0 = [sum, sum, sum, sum]
    MOVSS X0, ret+24(FP)

remainder_loop_sse32:
    MOVQ CX, AX
    ANDQ $3, AX               // AX = capacity % 4
    JZ   done_sse32
    MOVSS ret+24(FP), X0

remainder_iter_sse32:
    MOVSS (SI), X1
    MULSS X1, X1
    ADDSS X1, X0
    ADDQ DX, SI
    DECQ AX
    JNZ  remainder_iter_sse32
    MOVSS X0, ret+24(FP)

done_sse32:
    RET

// blazeReduceDotProductF64SSE41
TEXT ·blazeReduceDotProductF64SSE41(SB), NOSPLIT, $0-48
    MOVQ aData+0(FP), SI
    MOVQ bData+8(FP), DI
    MOVQ aSize+16(FP), DX
    MOVQ bSize+24(FP), R8
    MOVQ capacity+32(FP), CX

    XORPS X0, X0
    MOVQ CX, AX
    SHRQ $1, AX               // AX = capacity / 2
    JZ   remainder_loop_dot_sse64

simd_loop_dot_sse64:
    MOVUPD (SI), X1           // X1 = [a0, a1]
    MOVUPD (DI), X2           // X2 = [b0, b1]
    MULPD X2, X1              // X1 = [a0*b0, a1*b1]
    ADDPD X1, X0              // X0 += X1

    MOVQ DX, R9
    SHLQ $1, R9
    ADDQ R9, SI
    MOVQ R8, R9
    SHLQ $1, R9
    ADDQ R9, DI
    DECQ AX
    JNZ  simd_loop_dot_sse64

    HADDPD X0, X0
    MOVSD X0, ret+40(FP)

remainder_loop_dot_sse64:
    MOVQ CX, AX
    ANDQ $1, AX
    JZ   done_dot_sse64
    MOVSD ret+40(FP), X0

remainder_iter_dot_sse64:
    MOVSD (SI), X1
    MOVSD (DI), X2
    MULSD X2, X1
    ADDSD X1, X0
    ADDQ DX, SI
    ADDQ R8, DI
    DECQ AX
    JNZ  remainder_iter_dot_sse64
    MOVSD X0, ret+40(FP)

done_dot_sse64:
    RET

// blazeReduceDotProductF32SSE41
TEXT ·blazeReduceDotProductF32SSE41(SB), NOSPLIT, $0-40
    MOVQ aData+0(FP), SI
    MOVQ bData+8(FP), DI
    MOVQ aSize+16(FP), DX
    MOVQ bSize+24(FP), R8
    MOVQ capacity+32(FP), CX

    XORPS X0, X0
    MOVQ CX, AX
    SHRQ $2, AX               // AX = capacity / 4
    JZ   remainder_loop_dot_sse32

simd_loop_dot_sse32:
    MOVUPS (SI), X1           // X1 = [a0, a1, a2, a3]
    MOVUPS (DI), X2           // X2 = [b0, b1, b2, b3]
    MULPS X2, X1              // X1 = [a0*b0, a1*b1, a2*b2, a3*b3]
    ADDPS X1, X0              // X0 += X1

    MOVQ DX, R9
    SHLQ $2, R9
    ADDQ R9, SI
    MOVQ R8, R9
    SHLQ $2, R9
    ADDQ R9, DI
    DECQ AX
    JNZ  simd_loop_dot_sse32

    HADDPS X0, X0
    HADDPS X0, X0
    MOVSS X0, ret+40(FP)

remainder_loop_dot_sse32:
    MOVQ CX, AX
    ANDQ $3, AX
    JZ   done_dot_sse32
    MOVSS ret+40(FP), X0

remainder_iter_dot_sse32:
    MOVSS (SI), X1
    MOVSS (DI), X2
    MULSS X2, X1
    ADDSS X1, X0
    ADDQ DX, SI
    ADDQ R8, DI
    DECQ AX
    JNZ  remainder_iter_dot_sse32
    MOVSS X0, ret+40(FP)

done_dot_sse32:
    RET

// blazeElementWiseVectorAddF32SSE41
TEXT ·blazeElementWiseVectorAddF32SSE41(SB), NOSPLIT, $0-56
    MOVQ aData+0(FP), SI
    MOVQ bData+8(FP), DI
    MOVQ dstData+16(FP), R9
    MOVQ aSize+24(FP), DX
    MOVQ bSize+32(FP), R8
    MOVQ dstSize+40(FP), R10
    MOVQ capacity+48(FP), CX

    MOVQ CX, AX
    SHRQ $2, AX               // AX = capacity / 4
    JZ   remainder_loop_add_sse32

simd_loop_add_sse32:
    MOVUPS (SI), X0           // X0 = [a0, a1, a2, a3]
    MOVUPS (DI), X1           // X1 = [b0, b1, b2, b3]
    ADDPS X1, X0              // X0 = [a0+b0, a1+b1, a2+b2, a3+b3]
    MOVUPS X0, (R9)           // dstData = X0

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
    JNZ  simd_loop_add_sse32

remainder_loop_add_sse32:
    MOVQ CX, AX
    ANDQ $3, AX
    JZ   done_add_sse32

remainder_iter_add_sse32:
    MOVSS (SI), X0
    MOVSS (DI), X1
    ADDSS X1, X0
    MOVSS X0, (R9)
    ADDQ DX, SI
    ADDQ R8, DI
    ADDQ R10, R9
    DECQ AX
    JNZ  remainder_iter_add_sse32

done_add_sse32:
    RET

// blazeElementWiseVectorAddF64SSE41
TEXT ·blazeElementWiseVectorAddF64SSE41(SB), NOSPLIT, $0-56
    MOVQ aData+0(FP), SI
    MOVQ bData+8(FP), DI
    MOVQ dstData+16(FP), R9
    MOVQ aSize+24(FP), DX
    MOVQ bSize+32(FP), R8
    MOVQ dstSize+40(FP), R10
    MOVQ capacity+48(FP), CX

    MOVQ CX, AX
    SHRQ $1, AX               // AX = capacity / 2
    JZ   remainder_loop_add_sse64

simd_loop_add_sse64:
    MOVUPD (SI), X0           // X0 = [a0, a1]
    MOVUPD (DI), X1           // X1 = [b0, b1]
    ADDPD X1, X0              // X0 = [a0+b0, a1+b1]
    MOVUPD X0, (R9)           // dstData = X0

    MOVQ DX, R11
    SHLQ $1, R11
    ADDQ R11, SI
    MOVQ R8, R11
    SHLQ $1, R11
    ADDQ R11, DI
    MOVQ R10, R11
    SHLQ $1, R11
    ADDQ R11, R9
    DECQ AX
    JNZ  simd_loop_add_sse64

remainder_loop_add_sse64:
    MOVQ CX, AX
    ANDQ $1, AX
    JZ   done_add_sse64

remainder_iter_add_sse64:
    MOVSD (SI), X0
    MOVSD (DI), X1
    ADDSD X1, X0
    MOVSD X0, (R9)
    ADDQ DX, SI
    ADDQ R8, DI
    ADDQ R10, R9
    DECQ AX
    JNZ  remainder_iter_add_sse64

done_add_sse64:
    RET

// blazeElementWiseVectorMultiplyF32SSE41
TEXT ·blazeElementWiseVectorMultiplyF32SSE41(SB), NOSPLIT, $0-56
    MOVQ aData+0(FP), SI
    MOVQ bData+8(FP), DI
    MOVQ dstData+16(FP), R9
    MOVQ aSize+24(FP), DX
    MOVQ bSize+32(FP), R8
    MOVQ dstSize+40(FP), R10
    MOVQ capacity+48(FP), CX

    MOVQ CX, AX
    SHRQ $2, AX               // AX = capacity / 4
    JZ   remainder_loop_mul_sse32

simd_loop_mul_sse32:
    MOVUPS (SI), X0           // X0 = [a0, a1, a2, a3]
    MOVUPS (DI), X1           // X1 = [b0, b1, b2, b3]
    MULPS X1, X0              // X0 = [a0*b0, a1*b1, a2*b2, a3*b3]
    MOVUPS X0, (R9)           // dstData = X0

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
    JNZ  simd_loop_mul_sse32

remainder_loop_mul_sse32:
    MOVQ CX, AX
    ANDQ $3, AX
    JZ   done_mul_sse32

remainder_iter_mul_sse32:
    MOVSS (SI), X0
    MOVSS (DI), X1
    MULSS X1, X0
    MOVSS X0, (R9)
    ADDQ DX, SI
    ADDQ R8, DI
    ADDQ R10, R9
    DECQ AX
    JNZ  remainder_iter_mul_sse32

done_mul_sse32:
    RET

// blazeElementWiseVectorMultiplyF64SSE41
TEXT ·blazeElementWiseVectorMultiplyF64SSE41(SB), NOSPLIT, $0-56
    MOVQ aData+0(FP), SI
    MOVQ bData+8(FP), DI
    MOVQ dstData+16(FP), R9
    MOVQ aSize+24(FP), DX
    MOVQ bSize+32(FP), R8
    MOVQ dstSize+40(FP), R10
    MOVQ capacity+48(FP), CX

    MOVQ CX, AX
    SHRQ $1, AX               // AX = capacity / 2
    JZ   remainder_loop_mul_sse64

simd_loop_mul_sse64:
    MOVUPD (SI), X0           // X0 = [a0, a1]
    MOVUPD (DI), X1           // X1 = [b0, b1]
    MULPD X1, X0              // X0 = [a0*b0, a1*b1]
    MOVUPD X0, (R9)           // dstData = X0

    MOVQ DX, R11
    SHLQ $1, R11
    ADDQ R11, SI
    MOVQ R8, R11
    SHLQ $1, R11
    ADDQ R11, DI
    MOVQ R10, R11
    SHLQ $1, R11
    ADDQ R11, R9
    DECQ AX
    JNZ  simd_loop_mul_sse64

remainder_loop_mul_sse64:
    MOVQ CX, AX
    ANDQ $1, AX
    JZ   done_mul_sse64

remainder_iter_mul_sse64:
    MOVSD (SI), X0
    MOVSD (DI), X1
    MULSD X1, X0
    MOVSD X0, (R9)
    ADDQ DX, SI
    ADDQ R8, DI
    ADDQ R10, R9
    DECQ AX
    JNZ  remainder_iter_mul_sse64

done_mul_sse64:
    RET

// blazeScalarVectorMultiplyF32SSE41
TEXT ·blazeScalarVectorMultiplyF32SSE41(SB), NOSPLIT, $0-48
    MOVQ srcData+0(FP), SI
    MOVQ dstData+8(FP), DI
    MOVQ srcSize+16(FP), DX
    MOVQ dstSize+24(FP), R8
    MOVSS scalar+32(FP), X15  // X15 = [scalar, 0, 0, 0]
    MOVQ capacity+40(FP), CX

    // Broadcast scalar to all 4 positions
    SHUFPS $0, X15, X15       // X15 = [scalar, scalar, scalar, scalar]

    MOVQ CX, AX
    SHRQ $2, AX               // AX = capacity / 4
    JZ   remainder_loop_scalar_sse32

simd_loop_scalar_sse32:
    MOVUPS (SI), X0           // X0 = [v0, v1, v2, v3]
    MULPS X15, X0             // X0 = [v0*scalar, v1*scalar, v2*scalar, v3*scalar]
    MOVUPS X0, (DI)           // dstData = X0

    MOVQ DX, R9
    SHLQ $2, R9
    ADDQ R9, SI
    MOVQ R8, R9
    SHLQ $2, R9
    ADDQ R9, DI
    DECQ AX
    JNZ  simd_loop_scalar_sse32

remainder_loop_scalar_sse32:
    MOVQ CX, AX
    ANDQ $3, AX
    JZ   done_scalar_sse32

remainder_iter_scalar_sse32:
    MOVSS (SI), X0
    MULSS X15, X0
    MOVSS X0, (DI)
    ADDQ DX, SI
    ADDQ R8, DI
    DECQ AX
    JNZ  remainder_iter_scalar_sse32

done_scalar_sse32:
    RET

// blazeScalarVectorMultiplyF64SSE41
TEXT ·blazeScalarVectorMultiplyF64SSE41(SB), NOSPLIT, $0-48
    MOVQ srcData+0(FP), SI
    MOVQ dstData+8(FP), DI
    MOVQ srcSize+16(FP), DX
    MOVQ dstSize+24(FP), R8
    MOVSD scalar+32(FP), X15  // X15 = [scalar, 0]
    MOVQ capacity+40(FP), CX

    // Broadcast scalar to both positions
    MOVDDUP X15, X15          // X15 = [scalar, scalar]

    MOVQ CX, AX
    SHRQ $1, AX               // AX = capacity / 2
    JZ   remainder_loop_scalar_sse64

simd_loop_scalar_sse64:
    MOVUPD (SI), X0           // X0 = [v0, v1]
    MULPD X15, X0             // X0 = [v0*scalar, v1*scalar]
    MOVUPD X0, (DI)           // dstData = X0

    MOVQ DX, R9
    SHLQ $1, R9
    ADDQ R9, SI
    MOVQ R8, R9
    SHLQ $1, R9
    ADDQ R9, DI
    DECQ AX
    JNZ  simd_loop_scalar_sse64

remainder_loop_scalar_sse64:
    MOVQ CX, AX
    ANDQ $1, AX
    JZ   done_scalar_sse64

remainder_iter_scalar_sse64:
    MOVSD (SI), X0
    MULSD X15, X0
    MOVSD X0, (DI)
    ADDQ DX, SI
    ADDQ R8, DI
    DECQ AX
    JNZ  remainder_iter_scalar_sse64

done_scalar_sse64:
    RET

// blazeReduceVectorSumF64SSE41
TEXT ·blazeReduceVectorSumF64SSE41(SB), NOSPLIT, $0-32
    MOVQ data+0(FP), SI
    MOVQ size+8(FP), DX
    MOVQ capacity+16(FP), CX

    XORPS X0, X0
    MOVQ CX, AX
    SHRQ $1, AX               // AX = capacity / 2
    JZ   remainder_loop_sum_sse64

simd_loop_sum_sse64:
    MOVUPD (SI), X1           // X1 = [v0, v1]
    ADDPD X1, X0              // X0 += X1

    MOVQ DX, DI
    SHLQ $1, DI
    ADDQ DI, SI
    DECQ AX
    JNZ  simd_loop_sum_sse64

    HADDPD X0, X0
    MOVSD X0, ret+24(FP)

remainder_loop_sum_sse64:
    MOVQ CX, AX
    ANDQ $1, AX
    JZ   done_sum_sse64
    MOVSD ret+24(FP), X0

remainder_iter_sum_sse64:
    MOVSD (SI), X1
    ADDSD X1, X0
    ADDQ DX, SI
    DECQ AX
    JNZ  remainder_iter_sum_sse64
    MOVSD X0, ret+24(FP)

done_sum_sse64:
    RET

// blazeReduceVectorSumF32SSE41
TEXT ·blazeReduceVectorSumF32SSE41(SB), NOSPLIT, $0-24
    MOVQ data+0(FP), SI
    MOVQ size+8(FP), DX
    MOVQ capacity+16(FP), CX

    XORPS X0, X0
    MOVQ CX, AX
    SHRQ $2, AX               // AX = capacity / 4
    JZ   remainder_loop_sum_sse32

simd_loop_sum_sse32:
    MOVUPS (SI), X1           // X1 = [v0, v1, v2, v3]
    ADDPS X1, X0              // X0 += X1

    MOVQ DX, DI
    SHLQ $2, DI
    ADDQ DI, SI
    DECQ AX
    JNZ  simd_loop_sum_sse32

    HADDPS X0, X0
    HADDPS X0, X0
    MOVSS X0, ret+24(FP)

remainder_loop_sum_sse32:
    MOVQ CX, AX
    ANDQ $3, AX
    JZ   done_sum_sse32
    MOVSS ret+24(FP), X0

remainder_iter_sum_sse32:
    MOVSS (SI), X1
    ADDSS X1, X0
    ADDQ DX, SI
    DECQ AX
    JNZ  remainder_iter_sum_sse32
    MOVSS X0, ret+24(FP)

done_sum_sse32:
    RET

