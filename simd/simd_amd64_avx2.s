//go:build amd64

#include "textflag.h"

// blazeReduceVectorSumSquaredF64AVX2 computes sum of squares for float64 vector using AVX2
// func blazeReduceVectorSumSquaredF64AVX2(data unsafe.Pointer, size uintptr, capacity uint64) float64
TEXT ·blazeReduceVectorSumSquaredF64AVX2(SB), NOSPLIT, $0-32
    MOVQ data+0(FP), SI      // data pointer
    MOVQ size+8(FP), DX      // element size in bytes
    MOVQ capacity+16(FP), CX // capacity (number of elements)

    // Initialize accumulator: Y0 = [0, 0, 0, 0] (4x float64)
    VXORPS Y0, Y0, Y0        // Y0 = 0.0 (use VXORPS for zero, works for float64 too)

    // Calculate number of SIMD iterations (4 float64 per YMM register)
    MOVQ CX, AX
    SHRQ $2, AX              // AX = capacity / 4
    JZ   remainder_loop      // Skip if less than 4 elements

    // Main loop: process 4 elements at a time
simd_loop:
    // Load 4x float64 from data[offset]
    VMOVUPD (SI), Y1         // Y1 = [v0, v1, v2, v3] (unaligned load)

    // Square each element: Y1 = Y1 * Y1
    VMULPD Y1, Y1, Y1        // Y1 = [v0², v1², v2², v3²]

    // Accumulate: Y0 = Y0 + Y1
    VADDPD Y1, Y0, Y0        // Y0 += Y1

    // Advance pointer: SI += size * 4
    MOVQ DX, DI
    SHLQ $2, DI              // DI = size * 4
    ADDQ DI, SI              // SI += size * 4

    DECQ AX
    JNZ  simd_loop

    // Horizontal reduction: sum all 4 elements in Y0
    // Extract high 128 bits to XMM1
    VEXTRACTF128 $1, Y0, X1  // X1 = [Y0[2], Y0[3]]
    VADDPD X0, X1, X0        // X0 = [Y0[0]+Y0[2], Y0[1]+Y0[3]]
    VHADDPD X0, X0, X0       // X0 = [sum, sum] (horizontal add)
    VMOVSD X0, ret+24(FP)    // Return low 64 bits (the sum)

remainder_loop:
    // Handle remaining elements (0-3)
    MOVQ CX, AX
    ANDQ $3, AX              // AX = capacity % 4
    JZ   done

    // Load current sum into XMM0
    VMOVSD ret+24(FP), X0

remainder_iter:
    // Load single float64
    VMOVSD (SI), X1          // X1 = [v, 0]
    VMULSD X1, X1, X1        // X1 = [v², 0]
    VADDSD X1, X0, X0        // X0 += X1

    ADDQ DX, SI              // SI += size
    DECQ AX
    JNZ  remainder_iter

    VMOVSD X0, ret+24(FP)

done:
    VZEROUPPER               // Clear upper 128 bits of all YMM registers
    RET

// blazeReduceVectorSumSquaredF32AVX2 computes sum of squares for float32 vector using AVX2
// func blazeReduceVectorSumSquaredF32AVX2(data unsafe.Pointer, size uintptr, capacity uint64) float32
TEXT ·blazeReduceVectorSumSquaredF32AVX2(SB), NOSPLIT, $0-24
    MOVQ data+0(FP), SI      // data pointer
    MOVQ size+8(FP), DX      // element size in bytes
    MOVQ capacity+16(FP), CX // capacity

    // Initialize accumulator: Y0 = [0, 0, 0, 0, 0, 0, 0, 0] (8x float32)
    VXORPS Y0, Y0, Y0

    // Calculate number of SIMD iterations (8 float32 per YMM register)
    MOVQ CX, AX
    SHRQ $3, AX              // AX = capacity / 8
    JZ   remainder_loop_f32

    // Main loop: process 8 elements at a time
simd_loop_f32:
    // Load 8x float32 from data[offset]
    VMOVUPS (SI), Y1         // Y1 = [v0, v1, ..., v7]

    // Square each element
    VMULPS Y1, Y1, Y1        // Y1 = [v0², v1², ..., v7²]

    // Accumulate
    VADDPS Y1, Y0, Y0        // Y0 += Y1

    // Advance pointer: SI += size * 8
    MOVQ DX, DI
    SHLQ $3, DI              // DI = size * 8
    ADDQ DI, SI

    DECQ AX
    JNZ  simd_loop_f32

    // Horizontal reduction: sum all 8 elements
    // First, reduce 256-bit to 128-bit
    VEXTRACTF128 $1, Y0, X1  // X1 = [Y0[4], Y0[5], Y0[6], Y0[7]]
    VADDPS X0, X1, X0        // X0 = [sum0, sum1, sum2, sum3]
    // Then horizontal add within 128-bit
    VHADDPS X0, X0, X0       // X0 = [sum, sum, sum, sum]
    VHADDPS X0, X0, X0       // X0 = [final_sum, final_sum, final_sum, final_sum]
    VMOVSS X0, ret+24(FP)    // Return low 32 bits

remainder_loop_f32:
    // Handle remaining elements (0-7)
    MOVQ CX, AX
    ANDQ $7, AX              // AX = capacity % 8
    JZ   done_f32

    // Load current sum into XMM0
    VMOVSS ret+24(FP), X0

remainder_iter_f32:
    // Load single float32
    VMOVSS (SI), X1          // X1 = [v, 0, 0, 0]
    VMULSS X1, X1, X1        // X1 = [v², 0, 0, 0]
    VADDSS X1, X0, X0        // X0 += X1

    ADDQ DX, SI              // SI += size
    DECQ AX
    JNZ  remainder_iter_f32

    VMOVSS X0, ret+24(FP)

done_f32:
    VZEROUPPER
    RET

// blazeReduceDotProductF64AVX2 computes dot product for two float64 vectors using AVX2
// func blazeReduceDotProductF64AVX2(aData, bData unsafe.Pointer, aSize, bSize uintptr, capacity uint64) float64
TEXT ·blazeReduceDotProductF64AVX2(SB), NOSPLIT, $0-48
    MOVQ aData+0(FP), SI     // aData pointer
    MOVQ bData+8(FP), DI     // bData pointer
    MOVQ aSize+16(FP), DX    // a element size
    MOVQ bSize+24(FP), R8    // b element size
    MOVQ capacity+32(FP), CX // capacity

    // Initialize accumulator: Y0 = [0, 0, 0, 0]
    VXORPS Y0, Y0, Y0

    // Calculate number of SIMD iterations
    MOVQ CX, AX
    SHRQ $2, AX              // AX = capacity / 4
    JZ   remainder_loop_dot64

    // Main loop: process 4 elements at a time
simd_loop_dot64:
    // Load 4x float64 from aData
    VMOVUPD (SI), Y1         // Y1 = [a0, a1, a2, a3]

    // Load 4x float64 from bData
    VMOVUPD (DI), Y2         // Y2 = [b0, b1, b2, b3]

    // Multiply: Y1 = Y1 * Y2
    VMULPD Y2, Y1, Y1        // Y1 = [a0*b0, a1*b1, a2*b2, a3*b3]

    // Accumulate: Y0 = Y0 + Y1
    VADDPD Y1, Y0, Y0        // Y0 += Y1

    // Advance pointers
    MOVQ DX, R9
    SHLQ $2, R9              // R9 = aSize * 4
    ADDQ R9, SI              // SI += aSize * 4

    MOVQ R8, R9
    SHLQ $2, R9              // R9 = bSize * 4
    ADDQ R9, DI              // DI += bSize * 4

    DECQ AX
    JNZ  simd_loop_dot64

    // Horizontal reduction
    VEXTRACTF128 $1, Y0, X1  // X1 = [Y0[2], Y0[3]]
    VADDPD X0, X1, X0        // X0 = [Y0[0]+Y0[2], Y0[1]+Y0[3]]
    VHADDPD X0, X0, X0       // X0 = [sum, sum]
    VMOVSD X0, ret+40(FP)    // Return sum

remainder_loop_dot64:
    // Handle remaining elements (0-3)
    MOVQ CX, AX
    ANDQ $3, AX              // AX = capacity % 4
    JZ   done_dot64

    // Load current sum into XMM0
    VMOVSD ret+40(FP), X0

remainder_iter_dot64:
    // Load single float64 from aData
    VMOVSD (SI), X1          // X1 = [a, 0]
    // Load single float64 from bData
    VMOVSD (DI), X2          // X2 = [b, 0]
    // Multiply and accumulate
    VMULSD X2, X1, X1        // X1 = [a*b, 0]
    VADDSD X1, X0, X0        // X0 += X1

    ADDQ DX, SI              // SI += aSize
    ADDQ R8, DI              // DI += bSize
    DECQ AX
    JNZ  remainder_iter_dot64

    VMOVSD X0, ret+40(FP)

done_dot64:
    VZEROUPPER
    RET

// blazeReduceDotProductF32AVX2 computes dot product for two float32 vectors using AVX2
// func blazeReduceDotProductF32AVX2(aData, bData unsafe.Pointer, aSize, bSize uintptr, capacity uint64) float32
TEXT ·blazeReduceDotProductF32AVX2(SB), NOSPLIT, $0-40
    MOVQ aData+0(FP), SI     // aData pointer
    MOVQ bData+8(FP), DI     // bData pointer
    MOVQ aSize+16(FP), DX    // a element size
    MOVQ bSize+24(FP), R8    // b element size
    MOVQ capacity+32(FP), CX // capacity

    // Initialize accumulator: Y0 = [0, 0, 0, 0, 0, 0, 0, 0]
    VXORPS Y0, Y0, Y0

    // Calculate number of SIMD iterations
    MOVQ CX, AX
    SHRQ $3, AX              // AX = capacity / 8
    JZ   remainder_loop_dot32

    // Main loop: process 8 elements at a time
simd_loop_dot32:
    // Load 8x float32 from aData
    VMOVUPS (SI), Y1         // Y1 = [a0, a1, ..., a7]

    // Load 8x float32 from bData
    VMOVUPS (DI), Y2         // Y2 = [b0, b1, ..., b7]

    // Multiply: Y1 = Y1 * Y2
    VMULPS Y2, Y1, Y1        // Y1 = [a0*b0, a1*b1, ..., a7*b7]

    // Accumulate: Y0 = Y0 + Y1
    VADDPS Y1, Y0, Y0        // Y0 += Y1

    // Advance pointers
    MOVQ DX, R9
    SHLQ $3, R9              // R9 = aSize * 8
    ADDQ R9, SI              // SI += aSize * 8

    MOVQ R8, R9
    SHLQ $3, R9              // R9 = bSize * 8
    ADDQ R9, DI              // DI += bSize * 8

    DECQ AX
    JNZ  simd_loop_dot32

    // Horizontal reduction
    VEXTRACTF128 $1, Y0, X1  // X1 = [Y0[4], Y0[5], Y0[6], Y0[7]]
    VADDPS X0, X1, X0        // X0 = [sum0, sum1, sum2, sum3]
    VHADDPS X0, X0, X0       // X0 = [sum, sum, sum, sum]
    VHADDPS X0, X0, X0       // X0 = [final_sum, final_sum, final_sum, final_sum]
    VMOVSS X0, ret+40(FP)    // Return sum

remainder_loop_dot32:
    // Handle remaining elements (0-7)
    MOVQ CX, AX
    ANDQ $7, AX              // AX = capacity % 8
    JZ   done_dot32

    // Load current sum into XMM0
    VMOVSS ret+40(FP), X0

remainder_iter_dot32:
    // Load single float32 from aData
    VMOVSS (SI), X1          // X1 = [a, 0, 0, 0]
    // Load single float32 from bData
    VMOVSS (DI), X2          // X2 = [b, 0, 0, 0]
    // Multiply and accumulate
    VMULSS X2, X1, X1        // X1 = [a*b, 0, 0, 0]
    VADDSS X1, X0, X0        // X0 += X1

    ADDQ DX, SI              // SI += aSize
    ADDQ R8, DI              // DI += bSize
    DECQ AX
    JNZ  remainder_iter_dot32

    VMOVSS X0, ret+40(FP)

done_dot32:
    VZEROUPPER
    RET

// blazeElementWiseVectorAddF32AVX2 performs element-wise addition for float32 vectors using AVX2
// func blazeElementWiseVectorAddF32AVX2(aData, bData, dstData unsafe.Pointer, aSize, bSize, dstSize uintptr, capacity uint64)
TEXT ·blazeElementWiseVectorAddF32AVX2(SB), NOSPLIT, $0-56
    MOVQ aData+0(FP), SI     // aData pointer
    MOVQ bData+8(FP), DI     // bData pointer
    MOVQ dstData+16(FP), R9  // dstData pointer
    MOVQ aSize+24(FP), DX    // a element size
    MOVQ bSize+32(FP), R8    // b element size
    MOVQ dstSize+40(FP), R10 // dst element size
    MOVQ capacity+48(FP), CX // capacity

    // Calculate number of SIMD iterations (8 float32 per YMM register)
    MOVQ CX, AX
    SHRQ $3, AX              // AX = capacity / 8
    JZ   remainder_loop_add32

    // Main loop: process 8 elements at a time
simd_loop_add32:
    // Load 8x float32 from aData
    VMOVUPS (SI), Y0         // Y0 = [a0, a1, ..., a7]

    // Load 8x float32 from bData
    VMOVUPS (DI), Y1         // Y1 = [b0, b1, ..., b7]

    // Add: Y0 = Y0 + Y1
    VADDPS Y1, Y0, Y0        // Y0 = [a0+b0, a1+b1, ..., a7+b7]

    // Store result
    VMOVUPS Y0, (R9)         // dstData = Y0

    // Advance pointers
    MOVQ DX, R11
    SHLQ $3, R11             // R11 = aSize * 8
    ADDQ R11, SI             // SI += aSize * 8

    MOVQ R8, R11
    SHLQ $3, R11             // R11 = bSize * 8
    ADDQ R11, DI             // DI += bSize * 8

    MOVQ R10, R11
    SHLQ $3, R11             // R11 = dstSize * 8
    ADDQ R11, R9             // R9 += dstSize * 8

    DECQ AX
    JNZ  simd_loop_add32

remainder_loop_add32:
    // Handle remaining elements (0-7)
    MOVQ CX, AX
    ANDQ $7, AX              // AX = capacity % 8
    JZ   done_add32

remainder_iter_add32:
    // Load single float32 from aData
    VMOVSS (SI), X0          // X0 = [a, 0, 0, 0]
    // Load single float32 from bData
    VMOVSS (DI), X1          // X1 = [b, 0, 0, 0]
    // Add
    VADDSS X1, X0, X0        // X0 = [a+b, 0, 0, 0]
    // Store result
    VMOVSS X0, (R9)          // dstData = a+b

    ADDQ DX, SI              // SI += aSize
    ADDQ R8, DI              // DI += bSize
    ADDQ R10, R9             // R9 += dstSize
    DECQ AX
    JNZ  remainder_iter_add32

done_add32:
    VZEROUPPER
    RET

// blazeElementWiseVectorAddF64AVX2 performs element-wise addition for float64 vectors using AVX2
// func blazeElementWiseVectorAddF64AVX2(aData, bData, dstData unsafe.Pointer, aSize, bSize, dstSize uintptr, capacity uint64)
TEXT ·blazeElementWiseVectorAddF64AVX2(SB), NOSPLIT, $0-56
    MOVQ aData+0(FP), SI     // aData pointer
    MOVQ bData+8(FP), DI     // bData pointer
    MOVQ dstData+16(FP), R9  // dstData pointer
    MOVQ aSize+24(FP), DX    // a element size
    MOVQ bSize+32(FP), R8    // b element size
    MOVQ dstSize+40(FP), R10 // dst element size
    MOVQ capacity+48(FP), CX // capacity

    // Calculate number of SIMD iterations (4 float64 per YMM register)
    MOVQ CX, AX
    SHRQ $2, AX              // AX = capacity / 4
    JZ   remainder_loop_add64

    // Main loop: process 4 elements at a time
simd_loop_add64:
    // Load 4x float64 from aData
    VMOVUPD (SI), Y0         // Y0 = [a0, a1, a2, a3]

    // Load 4x float64 from bData
    VMOVUPD (DI), Y1         // Y1 = [b0, b1, b2, b3]

    // Add: Y0 = Y0 + Y1
    VADDPD Y1, Y0, Y0        // Y0 = [a0+b0, a1+b1, a2+b2, a3+b3]

    // Store result
    VMOVUPD Y0, (R9)         // dstData = Y0

    // Advance pointers
    MOVQ DX, R11
    SHLQ $2, R11             // R11 = aSize * 4
    ADDQ R11, SI             // SI += aSize * 4

    MOVQ R8, R11
    SHLQ $2, R11             // R11 = bSize * 4
    ADDQ R11, DI             // DI += bSize * 4

    MOVQ R10, R11
    SHLQ $2, R11             // R11 = dstSize * 4
    ADDQ R11, R9             // R9 += dstSize * 4

    DECQ AX
    JNZ  simd_loop_add64

remainder_loop_add64:
    // Handle remaining elements (0-3)
    MOVQ CX, AX
    ANDQ $3, AX              // AX = capacity % 4
    JZ   done_add64

remainder_iter_add64:
    // Load single float64 from aData
    VMOVSD (SI), X0          // X0 = [a, 0]
    // Load single float64 from bData
    VMOVSD (DI), X1          // X1 = [b, 0]
    // Add
    VADDSD X1, X0, X0        // X0 = [a+b, 0]
    // Store result
    VMOVSD X0, (R9)          // dstData = a+b

    ADDQ DX, SI              // SI += aSize
    ADDQ R8, DI              // DI += bSize
    ADDQ R10, R9             // R9 += dstSize
    DECQ AX
    JNZ  remainder_iter_add64

done_add64:
    VZEROUPPER
    RET

// blazeElementWiseVectorMultiplyF32AVX2 performs element-wise multiplication for float32 vectors using AVX2
// func blazeElementWiseVectorMultiplyF32AVX2(aData, bData, dstData unsafe.Pointer, aSize, bSize, dstSize uintptr, capacity uint64)
TEXT ·blazeElementWiseVectorMultiplyF32AVX2(SB), NOSPLIT, $0-56
    MOVQ aData+0(FP), SI     // aData pointer
    MOVQ bData+8(FP), DI     // bData pointer
    MOVQ dstData+16(FP), R9  // dstData pointer
    MOVQ aSize+24(FP), DX    // a element size
    MOVQ bSize+32(FP), R8    // b element size
    MOVQ dstSize+40(FP), R10 // dst element size
    MOVQ capacity+48(FP), CX // capacity

    // Calculate number of SIMD iterations (8 float32 per YMM register)
    MOVQ CX, AX
    SHRQ $3, AX              // AX = capacity / 8
    JZ   remainder_loop_mul32

    // Main loop: process 8 elements at a time
simd_loop_mul32:
    // Load 8x float32 from aData
    VMOVUPS (SI), Y0         // Y0 = [a0, a1, ..., a7]

    // Load 8x float32 from bData
    VMOVUPS (DI), Y1         // Y1 = [b0, b1, ..., b7]

    // Multiply: Y0 = Y0 * Y1
    VMULPS Y1, Y0, Y0        // Y0 = [a0*b0, a1*b1, ..., a7*b7]

    // Store result
    VMOVUPS Y0, (R9)         // dstData = Y0

    // Advance pointers
    MOVQ DX, R11
    SHLQ $3, R11             // R11 = aSize * 8
    ADDQ R11, SI             // SI += aSize * 8

    MOVQ R8, R11
    SHLQ $3, R11             // R11 = bSize * 8
    ADDQ R11, DI             // DI += bSize * 8

    MOVQ R10, R11
    SHLQ $3, R11             // R11 = dstSize * 8
    ADDQ R11, R9             // R9 += dstSize * 8

    DECQ AX
    JNZ  simd_loop_mul32

remainder_loop_mul32:
    // Handle remaining elements (0-7)
    MOVQ CX, AX
    ANDQ $7, AX              // AX = capacity % 8
    JZ   done_mul32

remainder_iter_mul32:
    // Load single float32 from aData
    VMOVSS (SI), X0          // X0 = [a, 0, 0, 0]
    // Load single float32 from bData
    VMOVSS (DI), X1          // X1 = [b, 0, 0, 0]
    // Multiply
    VMULSS X1, X0, X0        // X0 = [a*b, 0, 0, 0]
    // Store result
    VMOVSS X0, (R9)          // dstData = a*b

    ADDQ DX, SI              // SI += aSize
    ADDQ R8, DI              // DI += bSize
    ADDQ R10, R9             // R9 += dstSize
    DECQ AX
    JNZ  remainder_iter_mul32

done_mul32:
    VZEROUPPER
    RET

// blazeElementWiseVectorMultiplyF64AVX2 performs element-wise multiplication for float64 vectors using AVX2
// func blazeElementWiseVectorMultiplyF64AVX2(aData, bData, dstData unsafe.Pointer, aSize, bSize, dstSize uintptr, capacity uint64)
TEXT ·blazeElementWiseVectorMultiplyF64AVX2(SB), NOSPLIT, $0-56
    MOVQ aData+0(FP), SI     // aData pointer
    MOVQ bData+8(FP), DI     // bData pointer
    MOVQ dstData+16(FP), R9  // dstData pointer
    MOVQ aSize+24(FP), DX    // a element size
    MOVQ bSize+32(FP), R8    // b element size
    MOVQ dstSize+40(FP), R10 // dst element size
    MOVQ capacity+48(FP), CX // capacity

    // Calculate number of SIMD iterations (4 float64 per YMM register)
    MOVQ CX, AX
    SHRQ $2, AX              // AX = capacity / 4
    JZ   remainder_loop_mul64

    // Main loop: process 4 elements at a time
simd_loop_mul64:
    // Load 4x float64 from aData
    VMOVUPD (SI), Y0         // Y0 = [a0, a1, a2, a3]

    // Load 4x float64 from bData
    VMOVUPD (DI), Y1         // Y1 = [b0, b1, b2, b3]

    // Multiply: Y0 = Y0 * Y1
    VMULPD Y1, Y0, Y0        // Y0 = [a0*b0, a1*b1, a2*b2, a3*b3]

    // Store result
    VMOVUPD Y0, (R9)         // dstData = Y0

    // Advance pointers
    MOVQ DX, R11
    SHLQ $2, R11             // R11 = aSize * 4
    ADDQ R11, SI             // SI += aSize * 4

    MOVQ R8, R11
    SHLQ $2, R11             // R11 = bSize * 4
    ADDQ R11, DI             // DI += bSize * 4

    MOVQ R10, R11
    SHLQ $2, R11             // R11 = dstSize * 4
    ADDQ R11, R9             // R9 += dstSize * 4

    DECQ AX
    JNZ  simd_loop_mul64

remainder_loop_mul64:
    // Handle remaining elements (0-3)
    MOVQ CX, AX
    ANDQ $3, AX              // AX = capacity % 4
    JZ   done_mul64

remainder_iter_mul64:
    // Load single float64 from aData
    VMOVSD (SI), X0          // X0 = [a, 0]
    // Load single float64 from bData
    VMOVSD (DI), X1          // X1 = [b, 0]
    // Multiply
    VMULSD X1, X0, X0        // X0 = [a*b, 0]
    // Store result
    VMOVSD X0, (R9)          // dstData = a*b

    ADDQ DX, SI              // SI += aSize
    ADDQ R8, DI              // DI += bSize
    ADDQ R10, R9             // R9 += dstSize
    DECQ AX
    JNZ  remainder_iter_mul64

done_mul64:
    VZEROUPPER
    RET

// blazeScalarVectorMultiplyF32AVX2 multiplies each float32 vector element by a scalar using AVX2
// func blazeScalarVectorMultiplyF32AVX2(srcData, dstData unsafe.Pointer, srcSize, dstSize uintptr, scalar float32, capacity uint64)
TEXT ·blazeScalarVectorMultiplyF32AVX2(SB), NOSPLIT, $0-48
    MOVQ srcData+0(FP), SI   // srcData pointer
    MOVQ dstData+8(FP), DI   // dstData pointer
    MOVQ srcSize+16(FP), DX // src element size
    MOVQ dstSize+24(FP), R8  // dst element size
    MOVSS scalar+32(FP), X15 // X15 = [scalar, 0, 0, 0]
    MOVQ capacity+40(FP), CX // capacity

    // Broadcast scalar to all 8 positions in YMM register
    VBROADCASTSS X15, Y15    // Y15 = [scalar, scalar, ..., scalar] (8x)

    // Calculate number of SIMD iterations (8 float32 per YMM register)
    MOVQ CX, AX
    SHRQ $3, AX              // AX = capacity / 8
    JZ   remainder_loop_scalar_mul32

    // Main loop: process 8 elements at a time
simd_loop_scalar_mul32:
    // Load 8x float32 from srcData
    VMOVUPS (SI), Y0         // Y0 = [v0, v1, ..., v7]

    // Multiply by scalar: Y0 = Y0 * Y15
    VMULPS Y15, Y0, Y0       // Y0 = [v0*scalar, v1*scalar, ..., v7*scalar]

    // Store result
    VMOVUPS Y0, (DI)         // dstData = Y0

    // Advance pointers
    MOVQ DX, R9
    SHLQ $3, R9              // R9 = srcSize * 8
    ADDQ R9, SI              // SI += srcSize * 8

    MOVQ R8, R9
    SHLQ $3, R9              // R9 = dstSize * 8
    ADDQ R9, DI              // DI += dstSize * 8

    DECQ AX
    JNZ  simd_loop_scalar_mul32

remainder_loop_scalar_mul32:
    // Handle remaining elements (0-7)
    MOVQ CX, AX
    ANDQ $7, AX              // AX = capacity % 8
    JZ   done_scalar_mul32

remainder_iter_scalar_mul32:
    // Load single float32 from srcData
    VMOVSS (SI), X0          // X0 = [v, 0, 0, 0]
    // Multiply by scalar
    VMULSS X15, X0, X0       // X0 = [v*scalar, 0, 0, 0]
    // Store result
    VMOVSS X0, (DI)          // dstData = v*scalar

    ADDQ DX, SI              // SI += srcSize
    ADDQ R8, DI              // DI += dstSize
    DECQ AX
    JNZ  remainder_iter_scalar_mul32

done_scalar_mul32:
    VZEROUPPER
    RET

// blazeScalarVectorMultiplyF64AVX2 multiplies each float64 vector element by a scalar using AVX2
// func blazeScalarVectorMultiplyF64AVX2(srcData, dstData unsafe.Pointer, srcSize, dstSize uintptr, scalar float64, capacity uint64)
TEXT ·blazeScalarVectorMultiplyF64AVX2(SB), NOSPLIT, $0-48
    MOVQ srcData+0(FP), SI   // srcData pointer
    MOVQ dstData+8(FP), DI   // dstData pointer
    MOVQ srcSize+16(FP), DX  // src element size
    MOVQ dstSize+24(FP), R8  // dst element size
    MOVSD scalar+32(FP), X15 // X15 = [scalar, 0]
    MOVQ capacity+40(FP), CX // capacity

    // Broadcast scalar to all 4 positions in YMM register
    VBROADCASTSD X15, Y15    // Y15 = [scalar, scalar, scalar, scalar] (4x)

    // Calculate number of SIMD iterations (4 float64 per YMM register)
    MOVQ CX, AX
    SHRQ $2, AX              // AX = capacity / 4
    JZ   remainder_loop_scalar_mul64

    // Main loop: process 4 elements at a time
simd_loop_scalar_mul64:
    // Load 4x float64 from srcData
    VMOVUPD (SI), Y0         // Y0 = [v0, v1, v2, v3]

    // Multiply by scalar: Y0 = Y0 * Y15
    VMULPD Y15, Y0, Y0       // Y0 = [v0*scalar, v1*scalar, v2*scalar, v3*scalar]

    // Store result
    VMOVUPD Y0, (DI)         // dstData = Y0

    // Advance pointers
    MOVQ DX, R9
    SHLQ $2, R9              // R9 = srcSize * 4
    ADDQ R9, SI              // SI += srcSize * 4

    MOVQ R8, R9
    SHLQ $2, R9              // R9 = dstSize * 4
    ADDQ R9, DI              // DI += dstSize * 4

    DECQ AX
    JNZ  simd_loop_scalar_mul64

remainder_loop_scalar_mul64:
    // Handle remaining elements (0-3)
    MOVQ CX, AX
    ANDQ $3, AX              // AX = capacity % 4
    JZ   done_scalar_mul64

remainder_iter_scalar_mul64:
    // Load single float64 from srcData
    VMOVSD (SI), X0          // X0 = [v, 0]
    // Multiply by scalar
    VMULSD X15, X0, X0       // X0 = [v*scalar, 0]
    // Store result
    VMOVSD X0, (DI)          // dstData = v*scalar

    ADDQ DX, SI              // SI += srcSize
    ADDQ R8, DI              // DI += dstSize
    DECQ AX
    JNZ  remainder_iter_scalar_mul64

done_scalar_mul64:
    VZEROUPPER
    RET

// blazeReduceVectorSumF64AVX2 computes sum for float64 vector using AVX2
// func blazeReduceVectorSumF64AVX2(data unsafe.Pointer, size uintptr, capacity uint64) float64
TEXT ·blazeReduceVectorSumF64AVX2(SB), NOSPLIT, $0-32
    MOVQ data+0(FP), SI      // data pointer
    MOVQ size+8(FP), DX      // element size in bytes
    MOVQ capacity+16(FP), CX // capacity

    // Initialize accumulator: Y0 = [0, 0, 0, 0]
    VXORPS Y0, Y0, Y0

    // Calculate number of SIMD iterations (4 float64 per YMM register)
    MOVQ CX, AX
    SHRQ $2, AX              // AX = capacity / 4
    JZ   remainder_loop_sum64

    // Main loop: process 4 elements at a time
simd_loop_sum64:
    // Load 4x float64 from data[offset]
    VMOVUPD (SI), Y1         // Y1 = [v0, v1, v2, v3]

    // Accumulate: Y0 = Y0 + Y1
    VADDPD Y1, Y0, Y0        // Y0 += Y1

    // Advance pointer: SI += size * 4
    MOVQ DX, DI
    SHLQ $2, DI              // DI = size * 4
    ADDQ DI, SI              // SI += size * 4

    DECQ AX
    JNZ  simd_loop_sum64

    // Horizontal reduction: sum all 4 elements in Y0
    VEXTRACTF128 $1, Y0, X1  // X1 = [Y0[2], Y0[3]]
    VADDPD X0, X1, X0        // X0 = [Y0[0]+Y0[2], Y0[1]+Y0[3]]
    VHADDPD X0, X0, X0       // X0 = [sum, sum] (horizontal add)
    VMOVSD X0, ret+24(FP)    // Return low 64 bits (the sum)

remainder_loop_sum64:
    // Handle remaining elements (0-3)
    MOVQ CX, AX
    ANDQ $3, AX              // AX = capacity % 4
    JZ   done_sum64

    // Load current sum into XMM0
    VMOVSD ret+24(FP), X0

remainder_iter_sum64:
    // Load single float64
    VMOVSD (SI), X1          // X1 = [v, 0]
    VADDSD X1, X0, X0        // X0 += X1

    ADDQ DX, SI              // SI += size
    DECQ AX
    JNZ  remainder_iter_sum64

    VMOVSD X0, ret+24(FP)

done_sum64:
    VZEROUPPER
    RET

// blazeReduceVectorSumF32AVX2 computes sum for float32 vector using AVX2
// func blazeReduceVectorSumF32AVX2(data unsafe.Pointer, size uintptr, capacity uint64) float32
TEXT ·blazeReduceVectorSumF32AVX2(SB), NOSPLIT, $0-24
    MOVQ data+0(FP), SI      // data pointer
    MOVQ size+8(FP), DX      // element size in bytes
    MOVQ capacity+16(FP), CX // capacity

    // Initialize accumulator: Y0 = [0, 0, 0, 0, 0, 0, 0, 0]
    VXORPS Y0, Y0, Y0

    // Calculate number of SIMD iterations (8 float32 per YMM register)
    MOVQ CX, AX
    SHRQ $3, AX              // AX = capacity / 8
    JZ   remainder_loop_sum32

    // Main loop: process 8 elements at a time
simd_loop_sum32:
    // Load 8x float32 from data[offset]
    VMOVUPS (SI), Y1         // Y1 = [v0, v1, ..., v7]

    // Accumulate: Y0 = Y0 + Y1
    VADDPS Y1, Y0, Y0        // Y0 += Y1

    // Advance pointer: SI += size * 8
    MOVQ DX, DI
    SHLQ $3, DI              // DI = size * 8
    ADDQ DI, SI              // SI += size * 8

    DECQ AX
    JNZ  simd_loop_sum32

    // Horizontal reduction: sum all 8 elements
    VEXTRACTF128 $1, Y0, X1  // X1 = [Y0[4], Y0[5], Y0[6], Y0[7]]
    VADDPS X0, X1, X0        // X0 = [sum0, sum1, sum2, sum3]
    VHADDPS X0, X0, X0       // X0 = [sum, sum, sum, sum]
    VHADDPS X0, X0, X0       // X0 = [final_sum, final_sum, final_sum, final_sum]
    VMOVSS X0, ret+24(FP)    // Return low 32 bits

remainder_loop_sum32:
    // Handle remaining elements (0-7)
    MOVQ CX, AX
    ANDQ $7, AX              // AX = capacity % 8
    JZ   done_sum32

    // Load current sum into XMM0
    VMOVSS ret+24(FP), X0

remainder_iter_sum32:
    // Load single float32
    VMOVSS (SI), X1          // X1 = [v, 0, 0, 0]
    VADDSS X1, X0, X0        // X0 += X1

    ADDQ DX, SI              // SI += size
    DECQ AX
    JNZ  remainder_iter_sum32

    VMOVSS X0, ret+24(FP)

done_sum32:
    VZEROUPPER
    RET

