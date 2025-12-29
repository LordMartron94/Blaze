//go:build amd64

#include "textflag.h"

// FMA-optimized implementations using VFMADD231PD/VFMADD231PS
// These require both AVX2 and FMA support
// FMA instructions combine multiply and add into a single operation, reducing latency

// blazeReduceVectorSumSquaredF64FMA computes sum of squares for vector using AVX2+FMA
// Optimized with 2x loop unrolling and multiple accumulators to break dependency chains
// Element-size-aware: processes 64 bytes per iteration (2 YMM registers) regardless of element size
// func blazeReduceVectorSumSquaredF64FMA(data unsafe.Pointer, size uintptr, capacity uint64) float64
TEXT ·blazeReduceVectorSumSquaredF64FMA(SB), NOSPLIT, $0-24
    MOVQ data+0(FP), SI      // data pointer
    MOVQ size+8(FP), DX      // element size in bytes
    MOVQ capacity+16(FP), CX // capacity (number of elements)

    // Initialize two accumulators to break dependency chains and enable better pipelining
    VXORPS Y0, Y0, Y0         // Y0 = accumulator 0
    VXORPS Y2, Y2, Y2         // Y2 = accumulator 1

    // Calculate total bytes to process
    MOVQ CX, AX               // AX = capacity
    MULQ DX                   // AX = capacity * size (total bytes), DX = high bits
    MOVQ AX, R8               // R8 = total bytes
    // Check if total bytes >= 64 (need at least 64 bytes for one iteration)
    CMPQ R8, $64
    JL   remainder_loop_fma64

    // Calculate number of 64-byte iterations
    MOVQ R8, AX               // AX = total bytes
    SHRQ $6, AX               // AX = total bytes / 64 (process 64 bytes per iteration)
    JZ   remainder_loop_fma64

    // Main loop: process 64 bytes at a time (2x unrolled = 2 YMM registers)
    // This reduces loop overhead and enables better instruction scheduling
simd_loop_fma64:
    // Load first 32 bytes (YMM register)
    VMOVAPD (SI), Y1          // Y1 = [32 bytes of data]
    
    // Load second 32 bytes (YMM register)
    VMOVAPD 32(SI), Y3        // Y3 = [32 bytes of data]

    // Fused multiply-add on both accumulators in parallel
    // This breaks the dependency chain and allows better pipelining
    VFMADD231PD Y1, Y1, Y0    // Y0 += Y1 * Y1 (first block)
    VFMADD231PD Y3, Y3, Y2    // Y2 += Y3 * Y3 (second block)

    // Advance pointer: SI += 64 (always 64 bytes regardless of element size)
    ADDQ $64, SI

    DECQ AX
    JNZ  simd_loop_fma64

    // Combine accumulators: Y0 = Y0 + Y2
    VADDPD Y2, Y0, Y0         // Y0 = Y0 + Y2

    // Horizontal reduction: sum all 4 elements in Y0 using tree reduction
    // Tree reduction is faster than VHADDPD
    VEXTRACTF128 $1, Y0, X1   // X1 = [Y0[2], Y0[3]]
    VADDPD X0, X1, X0         // X0 = [Y0[0]+Y0[2], Y0[1]+Y0[3]]
    VUNPCKHPD X0, X0, X1      // X1 = [Y0[1]+Y0[3], Y0[1]+Y0[3]] (duplicate high)
    VADDPD X1, X0, X0         // X0 = [sum, sum]
    VMOVSD X0, ret+24(FP)     // Return sum

remainder_loop_fma64:
    // Handle remaining bytes (< 64 bytes)
    // Calculate remaining bytes: (capacity * size) % 64
    MOVQ CX, AX               // AX = capacity
    MULQ DX                   // AX = capacity * size, DX = high bits
    MOVQ AX, R8               // R8 = total bytes
    ANDQ $63, R8              // R8 = total bytes % 64 (remaining bytes)
    JZ   done_fma64

    // Load current sum into XMM0
    VMOVSD ret+24(FP), X0

    // Check if we have 32 or more remaining bytes
    CMPQ R8, $32
    JL   remainder_lt32_fma64

    // Process 32 bytes at once using YMM register
    VMOVAPD (SI), Y1          // Y1 = [32 bytes of data]
    VFMADD231PD Y1, Y1, Y0    // Y0 += Y1 * Y1
    ADDQ $32, SI
    SUBQ $32, R8
    JZ   remainder_done_fma64

remainder_lt32_fma64:
    // For remainder, skip processing (will be handled by Go fallback)
    // Remainder is small and conversion is complex for arbitrary types
    JMP  remainder_done_fma64

remainder_done_fma64:
    // Extract final sum from Y0 if we processed 32 bytes in remainder
    MOVQ CX, AX
    MULQ DX
    MOVQ AX, R8
    ANDQ $63, R8
    CMPQ R8, $32
    JL   remainder_scalar_fma64
    // We processed 32 bytes, need to reduce Y0
    VEXTRACTF128 $1, Y0, X1
    VADDPD X0, X1, X0
    VUNPCKHPD X0, X0, X1
    VADDPD X1, X0, X0
    JMP  remainder_store_fma64

remainder_scalar_fma64:
    // Only scalar operations were used, X0 already has the sum

remainder_store_fma64:
    VMOVSD X0, ret+24(FP)

done_fma64:
    VZEROUPPER
    RET

// blazeReduceVectorSumSquaredF32FMA computes sum of squares for vector using AVX2+FMA
// Optimized with 2x loop unrolling and multiple accumulators
// Element-size-aware: processes 64 bytes per iteration (2 YMM registers) regardless of element size
// func blazeReduceVectorSumSquaredF32FMA(data unsafe.Pointer, size uintptr, capacity uint64) float32
TEXT ·blazeReduceVectorSumSquaredF32FMA(SB), NOSPLIT, $0-24
    MOVQ data+0(FP), SI      // data pointer
    MOVQ size+8(FP), DX      // element size in bytes
    MOVQ capacity+16(FP), CX // capacity (number of elements)

    // Initialize two accumulators to break dependency chains
    VXORPS Y0, Y0, Y0         // Y0 = accumulator 0
    VXORPS Y2, Y2, Y2         // Y2 = accumulator 1

    // Calculate total bytes to process
    MOVQ CX, AX               // AX = capacity
    MULQ DX                   // AX = capacity * size (total bytes), DX = high bits
    MOVQ AX, R8               // R8 = total bytes
    // Check if total bytes >= 64 (need at least 64 bytes for one iteration)
    CMPQ R8, $64
    JL   remainder_loop_fma32

    // Calculate number of 64-byte iterations
    MOVQ R8, AX               // AX = total bytes
    SHRQ $6, AX               // AX = total bytes / 64 (process 64 bytes per iteration)
    JZ   remainder_loop_fma32

    // Main loop: process 64 bytes at a time (2x unrolled = 2 YMM registers)
simd_loop_fma32:
    // Load first 32 bytes (YMM register)
    VMOVAPS (SI), Y1          // Y1 = [32 bytes of data]
    
    // Load second 32 bytes (YMM register)
    VMOVAPS 32(SI), Y3        // Y3 = [32 bytes of data]

    // Fused multiply-add on both accumulators in parallel
    VFMADD231PS Y1, Y1, Y0    // Y0 += Y1 * Y1 (first block)
    VFMADD231PS Y3, Y3, Y2    // Y2 += Y3 * Y3 (second block)

    // Advance pointer: SI += 64 (always 64 bytes regardless of element size)
    ADDQ $64, SI

    DECQ AX
    JNZ  simd_loop_fma32

    // Combine accumulators: Y0 = Y0 + Y2
    VADDPS Y2, Y0, Y0         // Y0 = Y0 + Y2

    // Horizontal reduction: sum all 8 elements using tree reduction
    VEXTRACTF128 $1, Y0, X1   // X1 = [Y0[4], Y0[5], Y0[6], Y0[7]]
    VADDPS X0, X1, X0         // X0 = [sum0, sum1, sum2, sum3]
    VHADDPS X0, X0, X0        // X0 = [sum, sum, sum, sum]
    VHADDPS X0, X0, X0        // X0 = [final_sum, final_sum, final_sum, final_sum]
    VMOVSS X0, ret+24(FP)     // Return sum

remainder_loop_fma32:
    // Handle remaining bytes (< 64 bytes)
    // Calculate remaining bytes: (capacity * size) % 64
    MOVQ CX, AX               // AX = capacity
    MULQ DX                   // AX = capacity * size, DX = high bits
    MOVQ AX, R8               // R8 = total bytes
    ANDQ $63, R8              // R8 = total bytes % 64 (remaining bytes)
    JZ   done_fma32

    // Load current sum into XMM0
    VMOVSS ret+24(FP), X0

    // Check if we have 32 or more remaining bytes
    CMPQ R8, $32
    JL   remainder_lt32_fma32

    // Process 32 bytes at once using YMM register
    VMOVAPS (SI), Y1          // Y1 = [32 bytes of data]
    VFMADD231PS Y1, Y1, Y0    // Y0 += Y1 * Y1
    ADDQ $32, SI
    SUBQ $32, R8
    JZ   remainder_done_fma32

remainder_lt32_fma32:
    // For remainder, skip processing (will be handled by Go fallback)
    // Remainder is small and conversion is complex for arbitrary types
    JMP  remainder_done_fma32

remainder_done_fma32:
    // Extract final sum from Y0 if we processed 32 bytes in remainder
    MOVQ CX, AX
    MULQ DX
    MOVQ AX, R8
    ANDQ $63, R8
    CMPQ R8, $32
    JL   remainder_scalar_fma32
    // We processed 32 bytes, need to reduce Y0
    VEXTRACTF128 $1, Y0, X1
    VADDPS X0, X1, X0
    VHADDPS X0, X0, X0
    VHADDPS X0, X0, X0
    JMP  remainder_store_fma32

remainder_scalar_fma32:
    // Only scalar operations were used, X0 already has the sum

remainder_store_fma32:
    VMOVSS X0, ret+24(FP)

done_fma32:
    VZEROUPPER
    RET

// blazeReduceDotProductF64FMA computes dot product for two vectors using AVX2+FMA
// Optimized with 2x loop unrolling and multiple accumulators
// Element-size-aware: processes 64 bytes per iteration (2 YMM registers) regardless of element size
// Assumes both vectors have the same element size (contiguous operations)
// func blazeReduceDotProductF64FMA(aData, bData unsafe.Pointer, size uintptr, capacity uint64) float64
TEXT ·blazeReduceDotProductF64FMA(SB), NOSPLIT, $0-40
    MOVQ aData+0(FP), SI     // aData pointer
    MOVQ bData+8(FP), DI     // bData pointer
    MOVQ size+16(FP), DX     // element size in bytes (same for both vectors)
    MOVQ capacity+24(FP), CX // capacity (number of elements)
    MOVQ DX, R9              // Save size in R9 (callee-saved) to preserve it

    // Initialize two accumulators to break dependency chains
    VXORPS Y0, Y0, Y0         // Y0 = accumulator 0
    VXORPS Y4, Y4, Y4         // Y4 = accumulator 1

    // Calculate total bytes to process
    MOVQ CX, AX               // AX = capacity
    MOVQ R9, DX               // Restore size to DX for multiplication
    MULQ DX                   // AX = capacity * size (total bytes), DX = high bits
    MOVQ AX, R8               // R8 = total bytes
    MOVQ R9, DX               // Restore size to DX
    // Check if total bytes >= 64 (need at least 64 bytes for one iteration)
    CMPQ R8, $64
    JL   remainder_loop_dot_fma64

    // Calculate number of 64-byte iterations
    MOVQ R8, AX               // AX = total bytes
    SHRQ $6, AX               // AX = total bytes / 64 (process 64 bytes per iteration)
    JZ   remainder_loop_dot_fma64

    // Main loop: process 64 bytes at a time (2x unrolled = 2 YMM registers)
simd_loop_dot_fma64:
    // Load first 32 bytes from aData and bData
    VMOVAPD (SI), Y1          // Y1 = [32 bytes of aData]
    VMOVAPD (DI), Y2          // Y2 = [32 bytes of bData]
    
    // Load second 32 bytes from aData and bData
    VMOVAPD 32(SI), Y3        // Y3 = [32 bytes of aData]
    VMOVAPD 32(DI), Y5        // Y5 = [32 bytes of bData]

    // Fused multiply-add on both accumulators in parallel
    VFMADD231PD Y1, Y2, Y0    // Y0 += Y1 * Y2 (first block)
    VFMADD231PD Y3, Y5, Y4    // Y4 += Y3 * Y5 (second block)

    // Advance pointers: SI += 64, DI += 64 (always 64 bytes regardless of element size)
    ADDQ $64, SI              // SI += 64
    ADDQ $64, DI              // DI += 64

    DECQ AX
    JNZ  simd_loop_dot_fma64

    // Combine accumulators: Y0 = Y0 + Y4
    VADDPD Y4, Y0, Y0         // Y0 = Y0 + Y4

    // Horizontal reduction using tree reduction
    VEXTRACTF128 $1, Y0, X1   // X1 = [Y0[2], Y0[3]]
    VADDPD X0, X1, X0         // X0 = [Y0[0]+Y0[2], Y0[1]+Y0[3]]
    VUNPCKHPD X0, X0, X1      // X1 = [Y0[1]+Y0[3], Y0[1]+Y0[3]]
    VADDPD X1, X0, X0         // X0 = [sum, sum]
    VMOVSD X0, ret+32(FP)     // Return sum

remainder_loop_dot_fma64:
    // Handle remaining bytes (< 64 bytes)
    // Calculate remaining bytes: (capacity * size) % 64
    MOVQ CX, AX               // AX = capacity
    MOVQ R9, DX               // Restore size to DX
    MULQ DX                   // AX = capacity * size, DX = high bits
    MOVQ AX, R8               // R8 = total bytes
    ANDQ $63, R8              // R8 = total bytes % 64 (remaining bytes)
    MOVQ R9, DX               // Restore size to DX
    JZ   done_dot_fma64

    // Load current sum into XMM0
    VMOVSD ret+32(FP), X0

    // Check if we have 32 or more remaining bytes
    CMPQ R8, $32
    JL   remainder_lt32_dot_fma64

    // Process 32 bytes at once using YMM registers
    VMOVAPD (SI), Y1          // Y1 = [32 bytes of aData]
    VMOVAPD (DI), Y2          // Y2 = [32 bytes of bData]
    VFMADD231PD Y1, Y2, Y0    // Y0 += Y1 * Y2
    ADDQ $32, SI
    ADDQ $32, DI
    SUBQ $32, R8
    JZ   remainder_done_dot_fma64

remainder_lt32_dot_fma64:
    // For remainder, skip processing (will be handled by Go fallback)
    // Remainder is small and conversion is complex for arbitrary types
    JMP  remainder_done_dot_fma64

remainder_done_dot_fma64:
    // Extract final sum from Y0 if we processed 32 bytes in remainder
    MOVQ CX, AX
    MOVQ R9, DX               // Restore size to DX
    MULQ DX
    MOVQ AX, R8
    ANDQ $63, R8
    CMPQ R8, $32
    JL   remainder_scalar_dot_fma64
    // We processed 32 bytes, need to reduce Y0
    VEXTRACTF128 $1, Y0, X1
    VADDPD X0, X1, X0
    VUNPCKHPD X0, X0, X1
    VADDPD X1, X0, X0
    JMP  remainder_store_dot_fma64

remainder_scalar_dot_fma64:
    // Only scalar operations were used, X0 already has the sum

remainder_store_dot_fma64:
    VMOVSD X0, ret+32(FP)

done_dot_fma64:
    VZEROUPPER
    RET

// blazeReduceDotProductF32FMA computes dot product for two vectors using AVX2+FMA
// Optimized with 2x loop unrolling and multiple accumulators
// Element-size-aware: processes 64 bytes per iteration (2 YMM registers) regardless of element size
// Assumes both vectors have the same element size (contiguous operations)
// func blazeReduceDotProductF32FMA(aData, bData unsafe.Pointer, size uintptr, capacity uint64) float32
TEXT ·blazeReduceDotProductF32FMA(SB), NOSPLIT, $0-32
    MOVQ aData+0(FP), SI     // aData pointer
    MOVQ bData+8(FP), DI     // bData pointer
    MOVQ size+16(FP), DX     // element size in bytes (same for both vectors)
    MOVQ capacity+24(FP), CX // capacity (number of elements)
    MOVQ DX, R9              // Save size in R9 (callee-saved) to preserve it

    // Initialize two accumulators to break dependency chains
    VXORPS Y0, Y0, Y0         // Y0 = accumulator 0
    VXORPS Y4, Y4, Y4         // Y4 = accumulator 1

    // Calculate total bytes to process
    MOVQ CX, AX               // AX = capacity
    MOVQ R9, DX               // Restore size to DX for multiplication
    MULQ DX                   // AX = capacity * size (total bytes), DX = high bits
    MOVQ AX, R8               // R8 = total bytes
    MOVQ R9, DX               // Restore size to DX
    // Check if total bytes >= 64 (need at least 64 bytes for one iteration)
    CMPQ R8, $64
    JL   remainder_loop_dot_fma32

    // Calculate number of 64-byte iterations
    MOVQ R8, AX               // AX = total bytes
    SHRQ $6, AX               // AX = total bytes / 64 (process 64 bytes per iteration)
    JZ   remainder_loop_dot_fma32

    // Main loop: process 64 bytes at a time (2x unrolled = 2 YMM registers)
simd_loop_dot_fma32:
    // Load first 32 bytes from aData and bData
    VMOVAPS (SI), Y1          // Y1 = [32 bytes of aData]
    VMOVAPS (DI), Y2          // Y2 = [32 bytes of bData]
    
    // Load second 32 bytes from aData and bData
    VMOVAPS 32(SI), Y3        // Y3 = [32 bytes of aData]
    VMOVAPS 32(DI), Y5        // Y5 = [32 bytes of bData]

    // Fused multiply-add on both accumulators in parallel
    VFMADD231PS Y1, Y2, Y0    // Y0 += Y1 * Y2 (first block)
    VFMADD231PS Y3, Y5, Y4    // Y4 += Y3 * Y5 (second block)

    // Advance pointers: SI += 64, DI += 64 (always 64 bytes regardless of element size)
    ADDQ $64, SI              // SI += 64
    ADDQ $64, DI              // DI += 64

    DECQ AX
    JNZ  simd_loop_dot_fma32

    // Combine accumulators: Y0 = Y0 + Y4
    VADDPS Y4, Y0, Y0         // Y0 = Y0 + Y4

    // Horizontal reduction
    VEXTRACTF128 $1, Y0, X1   // X1 = [Y0[4], Y0[5], Y0[6], Y0[7]]
    VADDPS X0, X1, X0         // X0 = [sum0, sum1, sum2, sum3]
    VHADDPS X0, X0, X0        // X0 = [sum, sum, sum, sum]
    VHADDPS X0, X0, X0        // X0 = [final_sum, final_sum, final_sum, final_sum]
    VMOVSS X0, ret+32(FP)     // Return sum

remainder_loop_dot_fma32:
    // Handle remaining bytes (< 64 bytes)
    // Calculate remaining bytes: (capacity * size) % 64
    MOVQ CX, AX               // AX = capacity
    MOVQ R9, DX               // Restore size to DX
    MULQ DX                   // AX = capacity * size, DX = high bits
    MOVQ AX, R8               // R8 = total bytes
    ANDQ $63, R8              // R8 = total bytes % 64 (remaining bytes)
    MOVQ R9, DX               // Restore size to DX
    JZ   done_dot_fma32

    // Load current sum into XMM0
    VMOVSS ret+32(FP), X0

    // Check if we have 32 or more remaining bytes
    CMPQ R8, $32
    JL   remainder_lt32_dot_fma32

    // Process 32 bytes at once using YMM registers
    VMOVAPS (SI), Y1          // Y1 = [32 bytes of aData]
    VMOVAPS (DI), Y2          // Y2 = [32 bytes of bData]
    VFMADD231PS Y1, Y2, Y0    // Y0 += Y1 * Y2
    ADDQ $32, SI
    ADDQ $32, DI
    SUBQ $32, R8
    JZ   remainder_done_dot_fma32

remainder_lt32_dot_fma32:
    // For remainder, skip processing (will be handled by Go fallback)
    // Remainder is small and conversion is complex for arbitrary types
    JMP  remainder_done_dot_fma32

remainder_done_dot_fma32:
    // Extract final sum from Y0 if we processed 32 bytes in remainder
    MOVQ CX, AX
    MOVQ R9, DX               // Restore size to DX
    MULQ DX
    MOVQ AX, R8
    ANDQ $63, R8
    CMPQ R8, $32
    JL   remainder_scalar_dot_fma32
    // We processed 32 bytes, need to reduce Y0
    VEXTRACTF128 $1, Y0, X1
    VADDPS X0, X1, X0
    VHADDPS X0, X0, X0
    VHADDPS X0, X0, X0
    JMP  remainder_store_dot_fma32

remainder_scalar_dot_fma32:
    // Only scalar operations were used, X0 already has the sum

remainder_store_dot_fma32:
    VMOVSS X0, ret+32(FP)

done_dot_fma32:
    VZEROUPPER
    RET

