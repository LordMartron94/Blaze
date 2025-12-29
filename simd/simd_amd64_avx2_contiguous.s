//go:build amd64

#include "textflag.h"

// Contiguous access fast paths for AVX2
// These assume stride == sizeof(T), allowing simple pointer increments
// and enabling aligned loads when data is 32-byte aligned

// blazeReduceVectorSumSquaredF64ContiguousAVX2 computes sum of squares for contiguous vector using AVX2
// Optimized with 2x loop unrolling and multiple accumulators to break dependency chains
// Element-size-aware: processes 64 bytes per iteration (2 YMM registers) regardless of element size
// func blazeReduceVectorSumSquaredF64ContiguousAVX2(data unsafe.Pointer, size uintptr, capacity uint64) float64
TEXT ·blazeReduceVectorSumSquaredF64ContiguousAVX2(SB), NOSPLIT, $0-32
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
    JL   remainder_loop_contig64

    // Calculate number of 64-byte iterations
    MOVQ R8, AX               // AX = total bytes
    SHRQ $6, AX               // AX = total bytes / 64 (process 64 bytes per iteration)
    JZ   remainder_loop_contig64

    // Main loop: process 64 bytes at a time (2x unrolled = 2 YMM registers)
simd_loop_contig64:
    // Load first 32 bytes (YMM register)
    VMOVAPD (SI), Y1          // Y1 = [32 bytes of data]
    
    // Load second 32 bytes (YMM register)
    VMOVAPD 32(SI), Y3        // Y3 = [32 bytes of data]

    // Square and accumulate on both accumulators in parallel
    // This breaks the dependency chain and allows better pipelining
    VMULPD Y1, Y1, Y1         // Y1 = Y1 * Y1 (square)
    VADDPD Y1, Y0, Y0         // Y0 += Y1 (first block)
    
    VMULPD Y3, Y3, Y3         // Y3 = Y3 * Y3 (square)
    VADDPD Y3, Y2, Y2         // Y2 += Y3 (second block)

    // Advance pointer: SI += 64 (always 64 bytes regardless of element size)
    ADDQ $64, SI

    DECQ AX
    JNZ  simd_loop_contig64

    // Combine accumulators: Y0 = Y0 + Y2
    VADDPD Y2, Y0, Y0         // Y0 = Y0 + Y2

    // Horizontal reduction: sum all 4 elements in Y0 using tree reduction
    VEXTRACTF128 $1, Y0, X1   // X1 = [Y0[2], Y0[3]]
    VADDPD X0, X1, X0         // X0 = [Y0[0]+Y0[2], Y0[1]+Y0[3]]
    VUNPCKHPD X0, X0, X1      // X1 = [Y0[1]+Y0[3], Y0[1]+Y0[3]]
    VADDPD X1, X0, X0         // X0 = [sum, sum]
    VMOVSD X0, ret+24(FP)     // Return sum

remainder_loop_contig64:
    // Handle remaining bytes (< 64 bytes)
    // Calculate remaining bytes: (capacity * size) % 64
    MOVQ CX, AX               // AX = capacity
    MULQ DX                   // AX = capacity * size, DX = high bits
    MOVQ AX, R8               // R8 = total bytes
    ANDQ $63, R8              // R8 = total bytes % 64 (remaining bytes)
    JZ   done_contig64

    // Load current sum into XMM0
    VMOVSD ret+24(FP), X0

    // Check if we have 32 or more remaining bytes
    CMPQ R8, $32
    JL   remainder_lt32_contig64

    // Process 32 bytes at once using YMM register
    VMOVAPD (SI), Y1          // Y1 = [32 bytes of data]
    VMULPD Y1, Y1, Y1         // Y1 = Y1 * Y1 (square)
    VADDPD Y1, Y0, Y0         // Y0 += Y1
    ADDQ $32, SI
    SUBQ $32, R8
    JZ   remainder_done_contig64

remainder_lt32_contig64:
    // Handle remaining bytes using scalar operations
    // Process elements one at a time based on element size
remainder_iter_contig64:
    // Load element based on size
    CMPQ DX, $8
    JE   remainder_load8_contig64
    CMPQ DX, $4
    JE   remainder_load4_contig64
    CMPQ DX, $2
    JE   remainder_load2_contig64
    // Default: 1 byte (or other sizes) - use scalar fallback
    // For non-standard sizes, process via Go fallback (handled by caller)
    JMP  remainder_done_contig64

remainder_load8_contig64:
    VMOVSD (SI), X1           // X1 = [8-byte element, 0]
    JMP   remainder_square_contig64

remainder_load4_contig64:
    VMOVSS (SI), X1           // X1 = [4-byte element, 0, 0, 0]
    CVTSS2SD X1, X1           // Convert to double
    JMP   remainder_square_contig64

remainder_load2_contig64:
    // For 2-byte elements, use scalar fallback
    // (Conversion is complex, remainder is small anyway)
    JMP  remainder_done_contig64

remainder_square_contig64:
    VMULSD X1, X1, X1         // X1 = X1 * X1 (square)
    VADDSD X1, X0, X0          // X0 += X1
    ADDQ DX, SI                // Advance by element size
    SUBQ DX, R8                // Decrement remaining bytes
    JA   remainder_iter_contig64

remainder_done_contig64:
    // Extract final sum from Y0 if we processed 32 bytes in remainder
    MOVQ CX, AX
    MULQ DX
    MOVQ AX, R8
    ANDQ $63, R8
    CMPQ R8, $32
    JL   remainder_scalar_contig64
    // We processed 32 bytes, need to reduce Y0
    VEXTRACTF128 $1, Y0, X1
    VADDPD X0, X1, X0
    VUNPCKHPD X0, X0, X1
    VADDPD X1, X0, X0
    JMP  remainder_store_contig64

remainder_scalar_contig64:
    // Only scalar operations were used, X0 already has the sum

remainder_store_contig64:
    VMOVSD X0, ret+24(FP)

done_contig64:
    VZEROUPPER
    RET

// blazeReduceVectorSumSquaredF32ContiguousAVX2 computes sum of squares for contiguous vector using AVX2
// Optimized with 2x loop unrolling and multiple accumulators
// Element-size-aware: processes 64 bytes per iteration (2 YMM registers) regardless of element size
// func blazeReduceVectorSumSquaredF32ContiguousAVX2(data unsafe.Pointer, size uintptr, capacity uint64) float32
TEXT ·blazeReduceVectorSumSquaredF32ContiguousAVX2(SB), NOSPLIT, $0-24
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
    JL   remainder_loop_contig32

    // Calculate number of 64-byte iterations
    MOVQ R8, AX               // AX = total bytes
    SHRQ $6, AX               // AX = total bytes / 64 (process 64 bytes per iteration)
    JZ   remainder_loop_contig32

    // Main loop: process 64 bytes at a time (2x unrolled = 2 YMM registers)
simd_loop_contig32:
    // Load first 32 bytes (YMM register)
    VMOVAPS (SI), Y1          // Y1 = [32 bytes of data]
    
    // Load second 32 bytes (YMM register)
    VMOVAPS 32(SI), Y3        // Y3 = [32 bytes of data]

    // Square and accumulate on both accumulators in parallel
    VMULPS Y1, Y1, Y1         // Y1 = Y1 * Y1 (square)
    VADDPS Y1, Y0, Y0         // Y0 += Y1 (first block)
    
    VMULPS Y3, Y3, Y3         // Y3 = Y3 * Y3 (square)
    VADDPS Y3, Y2, Y2         // Y2 += Y3 (second block)

    // Advance pointer: SI += 64 (always 64 bytes regardless of element size)
    ADDQ $64, SI

    DECQ AX
    JNZ  simd_loop_contig32

    // Combine accumulators: Y0 = Y0 + Y2
    VADDPS Y2, Y0, Y0         // Y0 = Y0 + Y2

    // Horizontal reduction: sum all 8 elements
    VEXTRACTF128 $1, Y0, X1   // X1 = [Y0[4], Y0[5], Y0[6], Y0[7]]
    VADDPS X0, X1, X0         // X0 = [sum0, sum1, sum2, sum3]
    VHADDPS X0, X0, X0        // X0 = [sum, sum, sum, sum]
    VHADDPS X0, X0, X0        // X0 = [final_sum, final_sum, final_sum, final_sum]
    VMOVSS X0, ret+24(FP)     // Return sum

remainder_loop_contig32:
    // Handle remaining bytes (< 64 bytes)
    // Calculate remaining bytes: (capacity * size) % 64
    MOVQ CX, AX               // AX = capacity
    MULQ DX                   // AX = capacity * size, DX = high bits
    MOVQ AX, R8               // R8 = total bytes
    ANDQ $63, R8              // R8 = total bytes % 64 (remaining bytes)
    JZ   done_contig32

    // Load current sum into XMM0
    VMOVSS ret+24(FP), X0

    // Check if we have 32 or more remaining bytes
    CMPQ R8, $32
    JL   remainder_lt32_contig32

    // Process 32 bytes at once using YMM register
    VMOVAPS (SI), Y1          // Y1 = [32 bytes of data]
    VMULPS Y1, Y1, Y1         // Y1 = Y1 * Y1 (square)
    VADDPS Y1, Y0, Y0         // Y0 += Y1
    ADDQ $32, SI
    SUBQ $32, R8
    JZ   remainder_done_contig32

remainder_lt32_contig32:
    // Handle remaining bytes using scalar operations
    // Process elements one at a time based on element size
remainder_iter_contig32:
    // Load element based on size
    CMPQ DX, $4
    JE   remainder_load4_contig32
    CMPQ DX, $2
    JE   remainder_load2_contig32
    // Default: 1 byte (or other sizes) - use scalar fallback
    // For non-standard sizes, process via Go fallback (handled by caller)
    JMP  remainder_done_contig32

remainder_load4_contig32:
    VMOVSS (SI), X1           // X1 = [4-byte element, 0, 0, 0]
    JMP   remainder_square_contig32

remainder_load2_contig32:
    // For 2-byte elements, use scalar fallback
    // (Conversion is complex, remainder is small anyway)
    JMP  remainder_done_contig32

remainder_square_contig32:
    VMULSS X1, X1, X1         // X1 = X1 * X1 (square)
    VADDSS X1, X0, X0          // X0 += X1
    ADDQ DX, SI                // Advance by element size
    SUBQ DX, R8                // Decrement remaining bytes
    JA   remainder_iter_contig32

remainder_done_contig32:
    // Extract final sum from Y0 if we processed 32 bytes in remainder
    MOVQ CX, AX
    MULQ DX
    MOVQ AX, R8
    ANDQ $63, R8
    CMPQ R8, $32
    JL   remainder_scalar_contig32
    // We processed 32 bytes, need to reduce Y0
    VEXTRACTF128 $1, Y0, X1
    VADDPS X0, X1, X0
    VHADDPS X0, X0, X0
    VHADDPS X0, X0, X0
    JMP  remainder_store_contig32

remainder_scalar_contig32:
    // Only scalar operations were used, X0 already has the sum

remainder_store_contig32:
    VMOVSS X0, ret+24(FP)

done_contig32:
    VZEROUPPER
    RET

// blazeReduceDotProductF64ContiguousAVX2 computes dot product for two contiguous vectors using AVX2
// Optimized with 2x loop unrolling and multiple accumulators
// Element-size-aware: processes 64 bytes per iteration (2 YMM registers) regardless of element size
// Assumes both vectors have the same element size (contiguous operations)
// func blazeReduceDotProductF64ContiguousAVX2(aData, bData unsafe.Pointer, size uintptr, capacity uint64) float64
TEXT ·blazeReduceDotProductF64ContiguousAVX2(SB), NOSPLIT, $0-40
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
    MULQ DX                   // AX = capacity * size (total bytes), DX = high bits
    MOVQ AX, R8               // R8 = total bytes
    // Check if total bytes >= 64 (need at least 64 bytes for one iteration)
    CMPQ R8, $64
    JL   remainder_loop_dot_contig64

    // Calculate number of 64-byte iterations
    MOVQ R8, AX               // AX = total bytes
    SHRQ $6, AX               // AX = total bytes / 64 (process 64 bytes per iteration)
    JZ   remainder_loop_dot_contig64

    // Main loop: process 64 bytes at a time (2x unrolled = 2 YMM registers)
simd_loop_dot_contig64:
    // Load first 32 bytes from aData and bData
    VMOVAPD (SI), Y1          // Y1 = [32 bytes of aData]
    VMOVAPD (DI), Y2          // Y2 = [32 bytes of bData]
    
    // Load second 32 bytes from aData and bData
    VMOVAPD 32(SI), Y3        // Y3 = [32 bytes of aData]
    VMOVAPD 32(DI), Y5        // Y5 = [32 bytes of bData]

    // Multiply and accumulate on both accumulators in parallel
    VMULPD Y2, Y1, Y1         // Y1 = Y1 * Y2 (multiply)
    VADDPD Y1, Y0, Y0         // Y0 += Y1 (first block)
    
    VMULPD Y5, Y3, Y3         // Y3 = Y3 * Y5 (multiply)
    VADDPD Y3, Y4, Y4         // Y4 += Y3 (second block)

    // Advance pointers: SI += 64, DI += 64 (always 64 bytes regardless of element size)
    ADDQ $64, SI              // SI += 64
    ADDQ $64, DI              // DI += 64

    DECQ AX
    JNZ  simd_loop_dot_contig64

    // Combine accumulators: Y0 = Y0 + Y4
    VADDPD Y4, Y0, Y0         // Y0 = Y0 + Y4

    // Horizontal reduction using tree reduction
    VEXTRACTF128 $1, Y0, X1   // X1 = [Y0[2], Y0[3]]
    VADDPD X0, X1, X0         // X0 = [Y0[0]+Y0[2], Y0[1]+Y0[3]]
    VUNPCKHPD X0, X0, X1      // X1 = [Y0[1]+Y0[3], Y0[1]+Y0[3]]
    VADDPD X1, X0, X0         // X0 = [sum, sum]
    VMOVSD X0, ret+32(FP)     // Return sum

remainder_loop_dot_contig64:
    // Handle remaining bytes (< 64 bytes)
    // Calculate remaining bytes: (capacity * size) % 64
    MOVQ CX, AX               // AX = capacity
    MOVQ R9, DX               // Restore size to DX
    MULQ DX                   // AX = capacity * size, DX = high bits
    MOVQ AX, R8               // R8 = total bytes
    ANDQ $63, R8              // R8 = total bytes % 64 (remaining bytes)
    MOVQ R9, DX               // Restore size to DX
    JZ   done_dot_contig64

    // Load current sum into XMM0
    VMOVSD ret+32(FP), X0

    // Check if we have 32 or more remaining bytes
    CMPQ R8, $32
    JL   remainder_lt32_dot_contig64

    // Process 32 bytes at once using YMM registers
    VMOVAPD (SI), Y1          // Y1 = [32 bytes of aData]
    VMOVAPD (DI), Y2          // Y2 = [32 bytes of bData]
    VMULPD Y2, Y1, Y1         // Y1 = Y1 * Y2 (multiply)
    VADDPD Y1, Y0, Y0         // Y0 += Y1
    ADDQ $32, SI
    ADDQ $32, DI
    SUBQ $32, R8
    JZ   remainder_done_dot_contig64

remainder_lt32_dot_contig64:
    // Handle remaining bytes using scalar operations
    // Iterate while remaining bytes >= element size
    MOVQ R9, DX                // Restore size to DX
remainder_iter_dot_contig64:
    CMPQ R8, DX                // Compare remaining bytes to element size
    JL   remainder_done_dot_contig64
    
    // Load elements from both vectors based on size
    CMPQ DX, $8
    JE   remainder_load8_dot_contig64
    CMPQ DX, $4
    JE   remainder_load4_dot_contig64
    CMPQ DX, $2
    JE   remainder_load2_dot_contig64
    // Default: 1 byte (or other sizes) - use scalar fallback
    // For non-standard sizes, process via Go fallback (handled by caller)
    JMP  remainder_done_dot_contig64

remainder_load8_dot_contig64:
    VMOVSD (SI), X1           // X1 = [8-byte element from a, 0]
    VMOVSD (DI), X2           // X2 = [8-byte element from b, 0]
    JMP   remainder_multiply_dot_contig64

remainder_load4_dot_contig64:
    VMOVSS (SI), X1           // X1 = [4-byte element from a, 0, 0, 0]
    VMOVSS (DI), X2           // X2 = [4-byte element from b, 0, 0, 0]
    CVTSS2SD X1, X1           // Convert to double
    CVTSS2SD X2, X2           // Convert to double
    JMP   remainder_multiply_dot_contig64

remainder_load2_dot_contig64:
    // For 2-byte elements, use scalar fallback
    // (Conversion is complex, remainder is small anyway)
    JMP  remainder_done_dot_contig64

remainder_multiply_dot_contig64:
    VMULSD X2, X1, X1         // X1 = X1 * X2 (multiply)
    VADDSD X1, X0, X0          // X0 += X1
    ADDQ DX, SI                // Advance aData by element size
    ADDQ DX, DI                // Advance bData by element size
    SUBQ DX, R8                // Decrement remaining bytes
    MOVQ R9, DX                // Restore size to DX for next iteration
    JMP  remainder_iter_dot_contig64

remainder_done_dot_contig64:
    // Extract final sum from Y0 if we processed 32 bytes in remainder
    MOVQ CX, AX
    MOVQ R9, DX                // Restore size to DX
    MULQ DX
    MOVQ AX, R8
    ANDQ $63, R8
    CMPQ R8, $32
    JL   remainder_scalar_dot_contig64
    // We processed 32 bytes, need to reduce Y0
    VEXTRACTF128 $1, Y0, X1
    VADDPD X0, X1, X0
    VUNPCKHPD X0, X0, X1
    VADDPD X1, X0, X0
    JMP  remainder_store_dot_contig64

remainder_scalar_dot_contig64:
    // Only scalar operations were used, X0 already has the sum

remainder_store_dot_contig64:
    VMOVSD X0, ret+32(FP)

done_dot_contig64:
    VZEROUPPER
    RET

// blazeReduceDotProductF32ContiguousAVX2 computes dot product for two contiguous vectors using AVX2
// Optimized with 2x loop unrolling and multiple accumulators
// Element-size-aware: processes 64 bytes per iteration (2 YMM registers) regardless of element size
// Assumes both vectors have the same element size (contiguous operations)
// func blazeReduceDotProductF32ContiguousAVX2(aData, bData unsafe.Pointer, size uintptr, capacity uint64) float32
TEXT ·blazeReduceDotProductF32ContiguousAVX2(SB), NOSPLIT, $0-32
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
    JL   remainder_loop_dot_contig32

    // Calculate number of 64-byte iterations
    MOVQ R8, AX               // AX = total bytes
    SHRQ $6, AX               // AX = total bytes / 64 (process 64 bytes per iteration)
    JZ   remainder_loop_dot_contig32

    // Main loop: process 64 bytes at a time (2x unrolled = 2 YMM registers)
simd_loop_dot_contig32:
    // Load first 32 bytes from aData and bData
    VMOVAPS (SI), Y1          // Y1 = [32 bytes of aData]
    VMOVAPS (DI), Y2          // Y2 = [32 bytes of bData]
    
    // Load second 32 bytes from aData and bData
    VMOVAPS 32(SI), Y3        // Y3 = [32 bytes of aData]
    VMOVAPS 32(DI), Y5        // Y5 = [32 bytes of bData]

    // Multiply and accumulate on both accumulators in parallel
    VMULPS Y2, Y1, Y1         // Y1 = Y1 * Y2 (multiply)
    VADDPS Y1, Y0, Y0         // Y0 += Y1 (first block)
    
    VMULPS Y5, Y3, Y3         // Y3 = Y3 * Y5 (multiply)
    VADDPS Y3, Y4, Y4         // Y4 += Y3 (second block)

    // Advance pointers: SI += 64, DI += 64 (always 64 bytes regardless of element size)
    ADDQ $64, SI              // SI += 64
    ADDQ $64, DI              // DI += 64

    DECQ AX
    JNZ  simd_loop_dot_contig32

    // Combine accumulators: Y0 = Y0 + Y4
    VADDPS Y4, Y0, Y0         // Y0 = Y0 + Y4

    // Horizontal reduction
    VEXTRACTF128 $1, Y0, X1   // X1 = [Y0[4], Y0[5], Y0[6], Y0[7]]
    VADDPS X0, X1, X0         // X0 = [sum0, sum1, sum2, sum3]
    VHADDPS X0, X0, X0        // X0 = [sum, sum, sum, sum]
    VHADDPS X0, X0, X0        // X0 = [final_sum, final_sum, final_sum, final_sum]
    VMOVSS X0, ret+32(FP)     // Return sum

remainder_loop_dot_contig32:
    // Handle remaining elements (0-15)
    MOVQ CX, AX
    ANDQ $15, AX             // AX = capacity % 16
    JZ   done_dot_contig32

    // Load current sum into XMM0
    VMOVSS ret+24(FP), X0

    // Check if we have 8 or more remaining elements
    CMPQ AX, $8
    JL   remainder_lt8_dot_contig32

    // Process 8 elements at once using YMM registers
    VMOVAPS (SI), Y1         // Y1 = [a0, a1, ..., a7]
    VMOVAPS (DI), Y2         // Y2 = [b0, b1, ..., b7]
    VMULPS Y2, Y1, Y1        // Y1 = [a0*b0, a1*b1, ..., a7*b7]
    VADDPS Y1, Y0, Y0        // Y0 += Y1
    ADDQ $32, SI
    ADDQ $32, DI
    SUBQ $8, AX
    JZ   remainder_done_dot_contig32

remainder_lt8_dot_contig32:
    // Handle remaining 0-7 elements using scalar operations
remainder_iter_dot_contig32:
    VMOVSS (SI), X1          // X1 = [a, 0, 0, 0]
    VMOVSS (DI), X2          // X2 = [b, 0, 0, 0]
    VMULSS X2, X1, X1        // X1 = [a*b, 0, 0, 0]
    VADDSS X1, X0, X0        // X0 += X1
    ADDQ $4, SI
    ADDQ $4, DI
    DECQ AX
    JNZ  remainder_iter_dot_contig32

remainder_done_dot_contig32:
    // Extract final sum from Y0 if we processed 8 elements in remainder
    MOVQ CX, AX
    ANDQ $15, AX
    CMPQ AX, $8
    JL   remainder_scalar_dot_contig32
    // We processed 8 elements, need to reduce Y0
    VEXTRACTF128 $1, Y0, X1
    VADDPS X0, X1, X0
    VHADDPS X0, X0, X0
    VHADDPS X0, X0, X0
    JMP  remainder_store_dot_contig32

remainder_scalar_dot_contig32:
    // Only scalar operations were used, X0 already has the sum

remainder_store_dot_contig32:
    VMOVSS X0, ret+24(FP)

done_dot_contig32:
    VZEROUPPER
    RET

// blazeElementWiseVectorAddF64ContiguousAVX2 performs element-wise addition for contiguous vectors using AVX2
// Optimized with 2x loop unrolling
// Element-size-aware: processes 64 bytes per iteration (2 YMM registers) regardless of element size
// Assumes all vectors have the same element size (contiguous operations)
// func blazeElementWiseVectorAddF64ContiguousAVX2(aData, bData, dstData unsafe.Pointer, size uintptr, capacity uint64)
TEXT ·blazeElementWiseVectorAddF64ContiguousAVX2(SB), NOSPLIT, $0-40
    MOVQ aData+0(FP), SI     // aData pointer
    MOVQ bData+8(FP), DI     // bData pointer
    MOVQ dstData+16(FP), R9  // dstData pointer
    MOVQ size+24(FP), DX     // element size in bytes (same for all vectors)
    MOVQ capacity+32(FP), CX // capacity (number of elements)

    // Calculate total bytes to process
    MOVQ CX, AX               // AX = capacity
    MULQ DX                   // AX = capacity * size (total bytes), DX = high bits
    MOVQ AX, R8               // R8 = total bytes
    // Check if total bytes >= 64 (need at least 64 bytes for one iteration)
    CMPQ R8, $64
    JL   remainder_loop_add_contig64

    // Calculate number of 64-byte iterations
    MOVQ R8, AX               // AX = total bytes
    SHRQ $6, AX               // AX = total bytes / 64 (process 64 bytes per iteration)
    JZ   remainder_loop_add_contig64

    // Main loop: process 64 bytes at a time (2x unrolled = 2 YMM registers)
simd_loop_add_contig64:
    // Load first 32 bytes from aData and bData
    VMOVAPD (SI), Y0          // Y0 = [32 bytes of aData]
    VMOVAPD (DI), Y1          // Y1 = [32 bytes of bData]

    // Add: Y0 = Y0 + Y1
    VADDPD Y1, Y0, Y0         // Y0 = Y0 + Y1

    // Store first 32 bytes to dstData
    VMOVAPD Y0, (R9)          // dstData = Y0

    // Load second 32 bytes from aData and bData
    VMOVAPD 32(SI), Y2        // Y2 = [32 bytes of aData]
    VMOVAPD 32(DI), Y3        // Y3 = [32 bytes of bData]

    // Add: Y2 = Y2 + Y3
    VADDPD Y3, Y2, Y2         // Y2 = Y2 + Y3

    // Store second 32 bytes to dstData
    VMOVAPD Y2, 32(R9)        // dstData = Y2

    // Advance pointers: SI += 64, DI += 64, R9 += 64 (always 64 bytes regardless of element size)
    ADDQ $64, SI              // SI += 64
    ADDQ $64, DI              // DI += 64
    ADDQ $64, R9              // R9 += 64

    DECQ AX
    JNZ  simd_loop_add_contig64

remainder_loop_add_contig64:
    // Handle remaining bytes (< 64 bytes)
    // Calculate remaining bytes: (capacity * size) % 64
    MOVQ CX, AX               // AX = capacity
    MULQ DX                   // AX = capacity * size, DX = high bits
    MOVQ AX, R8               // R8 = total bytes
    ANDQ $63, R8              // R8 = total bytes % 64 (remaining bytes)
    JZ   done_add_contig64

    // Check if we have 32 or more remaining bytes
    CMPQ R8, $32
    JL   remainder_lt32_add_contig64

    // Process 32 bytes at once using YMM registers
    VMOVAPD (SI), Y0          // Y0 = [32 bytes of aData]
    VMOVAPD (DI), Y1          // Y1 = [32 bytes of bData]
    VADDPD Y1, Y0, Y0         // Y0 = Y0 + Y1
    VMOVAPD Y0, (R9)          // dstData = Y0
    ADDQ $32, SI
    ADDQ $32, DI
    ADDQ $32, R9
    SUBQ $32, R8
    JZ   done_add_contig64

remainder_lt32_add_contig64:
    // For remainder, skip processing (will be handled by Go fallback)
    // Remainder is small and conversion is complex for arbitrary types
    JMP  done_add_contig64

done_add_contig64:
    VZEROUPPER
    RET

// blazeElementWiseVectorAddF32ContiguousAVX2 performs element-wise addition for contiguous vectors using AVX2
// Optimized with 2x loop unrolling
// Element-size-aware: processes 64 bytes per iteration (2 YMM registers) regardless of element size
// Assumes all vectors have the same element size (contiguous operations)
// func blazeElementWiseVectorAddF32ContiguousAVX2(aData, bData, dstData unsafe.Pointer, size uintptr, capacity uint64)
TEXT ·blazeElementWiseVectorAddF32ContiguousAVX2(SB), NOSPLIT, $0-40
    MOVQ aData+0(FP), SI     // aData pointer
    MOVQ bData+8(FP), DI     // bData pointer
    MOVQ dstData+16(FP), R9  // dstData pointer
    MOVQ size+24(FP), DX     // element size in bytes (same for all vectors)
    MOVQ capacity+32(FP), CX // capacity (number of elements)

    // Calculate total bytes to process
    MOVQ CX, AX               // AX = capacity
    MULQ DX                   // AX = capacity * size (total bytes), DX = high bits
    MOVQ AX, R8               // R8 = total bytes
    // Check if total bytes >= 64 (need at least 64 bytes for one iteration)
    CMPQ R8, $64
    JL   remainder_loop_add_contig32

    // Calculate number of 64-byte iterations
    MOVQ R8, AX               // AX = total bytes
    SHRQ $6, AX               // AX = total bytes / 64 (process 64 bytes per iteration)
    JZ   remainder_loop_add_contig32

    // Main loop: process 64 bytes at a time (2x unrolled = 2 YMM registers)
simd_loop_add_contig32:
    // Load first 32 bytes from aData and bData
    VMOVAPS (SI), Y0          // Y0 = [32 bytes of aData]
    VMOVAPS (DI), Y1          // Y1 = [32 bytes of bData]

    // Add: Y0 = Y0 + Y1
    VADDPS Y1, Y0, Y0         // Y0 = Y0 + Y1

    // Store first 32 bytes to dstData
    VMOVAPS Y0, (R9)          // dstData = Y0

    // Load second 32 bytes from aData and bData
    VMOVAPS 32(SI), Y2        // Y2 = [32 bytes of aData]
    VMOVAPS 32(DI), Y3        // Y3 = [32 bytes of bData]

    // Add: Y2 = Y2 + Y3
    VADDPS Y3, Y2, Y2         // Y2 = Y2 + Y3

    // Store second 32 bytes to dstData
    VMOVAPS Y2, 32(R9)        // dstData = Y2

    // Advance pointers: SI += 64, DI += 64, R9 += 64 (always 64 bytes regardless of element size)
    ADDQ $64, SI              // SI += 64
    ADDQ $64, DI              // DI += 64
    ADDQ $64, R9              // R9 += 64

    DECQ AX
    JNZ  simd_loop_add_contig32

remainder_loop_add_contig32:
    // Handle remaining bytes (< 64 bytes)
    // Calculate remaining bytes: (capacity * size) % 64
    MOVQ CX, AX               // AX = capacity
    MULQ DX                   // AX = capacity * size, DX = high bits
    MOVQ AX, R8               // R8 = total bytes
    ANDQ $63, R8              // R8 = total bytes % 64 (remaining bytes)
    JZ   done_add_contig32

    // Check if we have 32 or more remaining bytes
    CMPQ R8, $32
    JL   remainder_lt32_add_contig32

    // Process 32 bytes at once using YMM registers
    VMOVAPS (SI), Y0          // Y0 = [32 bytes of aData]
    VMOVAPS (DI), Y1          // Y1 = [32 bytes of bData]
    VADDPS Y1, Y0, Y0         // Y0 = Y0 + Y1
    VMOVAPS Y0, (R9)          // dstData = Y0
    ADDQ $32, SI
    ADDQ $32, DI
    ADDQ $32, R9
    SUBQ $32, R8
    JZ   done_add_contig32

remainder_lt32_add_contig32:
    // For remainder, skip processing (will be handled by Go fallback)
    // Remainder is small and conversion is complex for arbitrary types
    JMP  done_add_contig32

done_add_contig32:
    VZEROUPPER
    RET

// blazeElementWiseVectorMultiplyF64ContiguousAVX2 performs element-wise multiplication for contiguous vectors using AVX2
// Optimized with 2x loop unrolling
// Element-size-aware: processes 64 bytes per iteration (2 YMM registers) regardless of element size
// Assumes all vectors have the same element size (contiguous operations)
// func blazeElementWiseVectorMultiplyF64ContiguousAVX2(aData, bData, dstData unsafe.Pointer, size uintptr, capacity uint64)
TEXT ·blazeElementWiseVectorMultiplyF64ContiguousAVX2(SB), NOSPLIT, $0-40
    MOVQ aData+0(FP), SI     // aData pointer
    MOVQ bData+8(FP), DI     // bData pointer
    MOVQ dstData+16(FP), R9  // dstData pointer
    MOVQ size+24(FP), DX     // element size in bytes (same for all vectors)
    MOVQ capacity+32(FP), CX // capacity (number of elements)

    // Calculate total bytes to process
    MOVQ CX, AX               // AX = capacity
    MULQ DX                   // AX = capacity * size (total bytes), DX = high bits
    MOVQ AX, R8               // R8 = total bytes
    // Check if total bytes >= 64 (need at least 64 bytes for one iteration)
    CMPQ R8, $64
    JL   remainder_loop_mul_contig64

    // Calculate number of 64-byte iterations
    MOVQ R8, AX               // AX = total bytes
    SHRQ $6, AX               // AX = total bytes / 64 (process 64 bytes per iteration)
    JZ   remainder_loop_mul_contig64

    // Main loop: process 64 bytes at a time (2x unrolled = 2 YMM registers)
simd_loop_mul_contig64:
    // Load first 32 bytes from aData and bData
    VMOVAPD (SI), Y0          // Y0 = [32 bytes of aData]
    VMOVAPD (DI), Y1          // Y1 = [32 bytes of bData]

    // Multiply: Y0 = Y0 * Y1
    VMULPD Y1, Y0, Y0         // Y0 = Y0 * Y1

    // Store first 32 bytes to dstData
    VMOVAPD Y0, (R9)          // dstData = Y0

    // Load second 32 bytes from aData and bData
    VMOVAPD 32(SI), Y2        // Y2 = [32 bytes of aData]
    VMOVAPD 32(DI), Y3        // Y3 = [32 bytes of bData]

    // Multiply: Y2 = Y2 * Y3
    VMULPD Y3, Y2, Y2         // Y2 = Y2 * Y3

    // Store second 32 bytes to dstData
    VMOVAPD Y2, 32(R9)        // dstData = Y2

    // Advance pointers: SI += 64, DI += 64, R9 += 64 (always 64 bytes regardless of element size)
    ADDQ $64, SI              // SI += 64
    ADDQ $64, DI              // DI += 64
    ADDQ $64, R9              // R9 += 64

    DECQ AX
    JNZ  simd_loop_mul_contig64

remainder_loop_mul_contig64:
    // Handle remaining bytes (< 64 bytes)
    // Calculate remaining bytes: (capacity * size) % 64
    MOVQ CX, AX               // AX = capacity
    MULQ DX                   // AX = capacity * size, DX = high bits
    MOVQ AX, R8               // R8 = total bytes
    ANDQ $63, R8              // R8 = total bytes % 64 (remaining bytes)
    JZ   done_mul_contig64

    // Check if we have 32 or more remaining bytes
    CMPQ R8, $32
    JL   remainder_lt32_mul_contig64

    // Process 32 bytes at once using YMM registers
    VMOVAPD (SI), Y0          // Y0 = [32 bytes of aData]
    VMOVAPD (DI), Y1          // Y1 = [32 bytes of bData]
    VMULPD Y1, Y0, Y0         // Y0 = Y0 * Y1
    VMOVAPD Y0, (R9)          // dstData = Y0
    ADDQ $32, SI
    ADDQ $32, DI
    ADDQ $32, R9
    SUBQ $32, R8
    JZ   done_mul_contig64

remainder_lt32_mul_contig64:
    // For remainder, skip processing (will be handled by Go fallback)
    // Remainder is small and conversion is complex for arbitrary types
    JMP  done_mul_contig64

done_mul_contig64:
    VZEROUPPER
    RET

// blazeElementWiseVectorMultiplyF32ContiguousAVX2 performs element-wise multiplication for contiguous vectors using AVX2
// Optimized with 2x loop unrolling
// Element-size-aware: processes 64 bytes per iteration (2 YMM registers) regardless of element size
// Assumes all vectors have the same element size (contiguous operations)
// func blazeElementWiseVectorMultiplyF32ContiguousAVX2(aData, bData, dstData unsafe.Pointer, size uintptr, capacity uint64)
TEXT ·blazeElementWiseVectorMultiplyF32ContiguousAVX2(SB), NOSPLIT, $0-40
    MOVQ aData+0(FP), SI     // aData pointer
    MOVQ bData+8(FP), DI     // bData pointer
    MOVQ dstData+16(FP), R9  // dstData pointer
    MOVQ size+24(FP), DX     // element size in bytes (same for all vectors)
    MOVQ capacity+32(FP), CX // capacity (number of elements)

    // Calculate total bytes to process
    MOVQ CX, AX               // AX = capacity
    MULQ DX                   // AX = capacity * size (total bytes), DX = high bits
    MOVQ AX, R8               // R8 = total bytes
    // Check if total bytes >= 64 (need at least 64 bytes for one iteration)
    CMPQ R8, $64
    JL   remainder_loop_mul_contig32

    // Calculate number of 64-byte iterations
    MOVQ R8, AX               // AX = total bytes
    SHRQ $6, AX               // AX = total bytes / 64 (process 64 bytes per iteration)
    JZ   remainder_loop_mul_contig32

    // Main loop: process 64 bytes at a time (2x unrolled = 2 YMM registers)
simd_loop_mul_contig32:
    // Load first 32 bytes from aData and bData
    VMOVAPS (SI), Y0          // Y0 = [32 bytes of aData]
    VMOVAPS (DI), Y1          // Y1 = [32 bytes of bData]

    // Multiply: Y0 = Y0 * Y1
    VMULPS Y1, Y0, Y0         // Y0 = Y0 * Y1

    // Store first 32 bytes to dstData
    VMOVAPS Y0, (R9)          // dstData = Y0

    // Load second 32 bytes from aData and bData
    VMOVAPS 32(SI), Y2         // Y2 = [32 bytes of aData]
    VMOVAPS 32(DI), Y3         // Y3 = [32 bytes of bData]

    // Multiply: Y2 = Y2 * Y3
    VMULPS Y3, Y2, Y2         // Y2 = Y2 * Y3

    // Store second 32 bytes to dstData
    VMOVAPS Y2, 32(R9)        // dstData = Y2

    // Advance pointers: SI += 64, DI += 64, R9 += 64 (always 64 bytes regardless of element size)
    ADDQ $64, SI              // SI += 64
    ADDQ $64, DI              // DI += 64
    ADDQ $64, R9              // R9 += 64

    DECQ AX
    JNZ  simd_loop_mul_contig32

remainder_loop_mul_contig32:
    // Handle remaining bytes (< 64 bytes)
    // Calculate remaining bytes: (capacity * size) % 64
    MOVQ CX, AX               // AX = capacity
    MULQ DX                   // AX = capacity * size, DX = high bits
    MOVQ AX, R8               // R8 = total bytes
    ANDQ $63, R8              // R8 = total bytes % 64 (remaining bytes)
    JZ   done_mul_contig32

    // Check if we have 32 or more remaining bytes
    CMPQ R8, $32
    JL   remainder_lt32_mul_contig32

    // Process 32 bytes at once using YMM registers
    VMOVAPS (SI), Y0          // Y0 = [32 bytes of aData]
    VMOVAPS (DI), Y1          // Y1 = [32 bytes of bData]
    VMULPS Y1, Y0, Y0         // Y0 = Y0 * Y1
    VMOVAPS Y0, (R9)          // dstData = Y0
    ADDQ $32, SI
    ADDQ $32, DI
    ADDQ $32, R9
    SUBQ $32, R8
    JZ   done_mul_contig32

remainder_lt32_mul_contig32:
    // For remainder, skip processing (will be handled by Go fallback)
    // Remainder is small and conversion is complex for arbitrary types
    JMP  done_mul_contig32

done_mul_contig32:
    VZEROUPPER
    RET

// blazeScalarVectorMultiplyF64ContiguousAVX2 multiplies each contiguous vector element by a scalar using AVX2
// Optimized with 2x loop unrolling
// Element-size-aware: processes 64 bytes per iteration (2 YMM registers) regardless of element size
// Assumes both vectors have the same element size (contiguous operations)
// func blazeScalarVectorMultiplyF64ContiguousAVX2(srcData, dstData unsafe.Pointer, size uintptr, scalar float64, capacity uint64)
TEXT ·blazeScalarVectorMultiplyF64ContiguousAVX2(SB), NOSPLIT, $0-40
    MOVQ srcData+0(FP), SI   // srcData pointer
    MOVQ dstData+8(FP), DI   // dstData pointer
    MOVQ size+16(FP), DX     // element size in bytes (same for both vectors)
    MOVSD scalar+24(FP), X15 // X15 = [scalar, 0]
    MOVQ capacity+32(FP), CX // capacity (number of elements)

    // Broadcast scalar to all 4 positions in YMM register
    VBROADCASTSD X15, Y15    // Y15 = [scalar, scalar, scalar, scalar] (4x)

    // Calculate total bytes to process
    MOVQ CX, AX               // AX = capacity
    MULQ DX                   // AX = capacity * size (total bytes), DX = high bits
    MOVQ AX, R8               // R8 = total bytes
    // Check if total bytes >= 64 (need at least 64 bytes for one iteration)
    CMPQ R8, $64
    JL   remainder_loop_scalar_contig64

    // Calculate number of 64-byte iterations
    MOVQ R8, AX               // AX = total bytes
    SHRQ $6, AX               // AX = total bytes / 64 (process 64 bytes per iteration)
    JZ   remainder_loop_scalar_contig64

    // Main loop: process 64 bytes at a time (2x unrolled = 2 YMM registers)
simd_loop_scalar_contig64:
    // Load first 32 bytes from srcData
    VMOVAPD (SI), Y0          // Y0 = [32 bytes of srcData]

    // Multiply by scalar: Y0 = Y0 * Y15
    VMULPD Y15, Y0, Y0        // Y0 = Y0 * scalar

    // Store first 32 bytes to dstData
    VMOVAPD Y0, (DI)          // dstData = Y0

    // Load second 32 bytes from srcData
    VMOVAPD 32(SI), Y1         // Y1 = [32 bytes of srcData]

    // Multiply by scalar: Y1 = Y1 * Y15
    VMULPD Y15, Y1, Y1         // Y1 = Y1 * scalar

    // Store second 32 bytes to dstData
    VMOVAPD Y1, 32(DI)         // dstData = Y1

    // Advance pointers: SI += 64, DI += 64 (always 64 bytes regardless of element size)
    ADDQ $64, SI              // SI += 64
    ADDQ $64, DI              // DI += 64

    DECQ AX
    JNZ  simd_loop_scalar_contig64

remainder_loop_scalar_contig64:
    // Handle remaining bytes (< 64 bytes)
    // Calculate remaining bytes: (capacity * size) % 64
    MOVQ CX, AX               // AX = capacity
    MULQ DX                   // AX = capacity * size, DX = high bits
    MOVQ AX, R8               // R8 = total bytes
    ANDQ $63, R8              // R8 = total bytes % 64 (remaining bytes)
    JZ   done_scalar_contig64

    // Check if we have 32 or more remaining bytes
    CMPQ R8, $32
    JL   remainder_lt32_scalar_contig64

    // Process 32 bytes at once using YMM register
    VMOVAPD (SI), Y0          // Y0 = [32 bytes of srcData]
    VMULPD Y15, Y0, Y0        // Y0 = Y0 * scalar
    VMOVAPD Y0, (DI)          // dstData = Y0
    ADDQ $32, SI
    ADDQ $32, DI
    SUBQ $32, R8
    JZ   done_scalar_contig64

remainder_lt32_scalar_contig64:
    // For remainder, skip processing (will be handled by Go fallback)
    // Remainder is small and conversion is complex for arbitrary types
    JMP  done_scalar_contig64

done_scalar_contig64:
    VZEROUPPER
    RET

// blazeScalarVectorMultiplyF32ContiguousAVX2 multiplies each contiguous vector element by a scalar using AVX2
// Optimized with 2x loop unrolling
// Element-size-aware: processes 64 bytes per iteration (2 YMM registers) regardless of element size
// Assumes both vectors have the same element size (contiguous operations)
// func blazeScalarVectorMultiplyF32ContiguousAVX2(srcData, dstData unsafe.Pointer, size uintptr, scalar float32, capacity uint64)
TEXT ·blazeScalarVectorMultiplyF32ContiguousAVX2(SB), NOSPLIT, $0-32
    MOVQ srcData+0(FP), SI   // srcData pointer
    MOVQ dstData+8(FP), DI   // dstData pointer
    MOVQ size+16(FP), DX     // element size in bytes (same for both vectors)
    MOVSS scalar+24(FP), X15 // X15 = [scalar, 0, 0, 0]
    MOVQ capacity+32(FP), CX // capacity (number of elements)

    // Broadcast scalar to all 8 positions in YMM register
    VBROADCASTSS X15, Y15    // Y15 = [scalar, scalar, ..., scalar] (8x)

    // Calculate total bytes to process
    MOVQ CX, AX               // AX = capacity
    MULQ DX                   // AX = capacity * size (total bytes), DX = high bits
    MOVQ AX, R8               // R8 = total bytes
    // Check if total bytes >= 64 (need at least 64 bytes for one iteration)
    CMPQ R8, $64
    JL   remainder_loop_scalar_contig32

    // Calculate number of 64-byte iterations
    MOVQ R8, AX               // AX = total bytes
    SHRQ $6, AX               // AX = total bytes / 64 (process 64 bytes per iteration)
    JZ   remainder_loop_scalar_contig32

    // Main loop: process 64 bytes at a time (2x unrolled = 2 YMM registers)
simd_loop_scalar_contig32:
    // Load first 32 bytes from srcData
    VMOVAPS (SI), Y0          // Y0 = [32 bytes of srcData]

    // Multiply by scalar: Y0 = Y0 * Y15
    VMULPS Y15, Y0, Y0        // Y0 = Y0 * scalar

    // Store first 32 bytes to dstData
    VMOVAPS Y0, (DI)          // dstData = Y0

    // Load second 32 bytes from srcData
    VMOVAPS 32(SI), Y1         // Y1 = [32 bytes of srcData]

    // Multiply by scalar: Y1 = Y1 * Y15
    VMULPS Y15, Y1, Y1         // Y1 = Y1 * scalar

    // Store second 32 bytes to dstData
    VMOVAPS Y1, 32(DI)         // dstData = Y1

    // Advance pointers: SI += 64, DI += 64 (always 64 bytes regardless of element size)
    ADDQ $64, SI              // SI += 64
    ADDQ $64, DI              // DI += 64

    DECQ AX
    JNZ  simd_loop_scalar_contig32

remainder_loop_scalar_contig32:
    // Handle remaining bytes (< 64 bytes)
    // Calculate remaining bytes: (capacity * size) % 64
    MOVQ CX, AX               // AX = capacity
    MULQ DX                   // AX = capacity * size, DX = high bits
    MOVQ AX, R8               // R8 = total bytes
    ANDQ $63, R8              // R8 = total bytes % 64 (remaining bytes)
    JZ   done_scalar_contig32

    // Check if we have 32 or more remaining bytes
    CMPQ R8, $32
    JL   remainder_lt32_scalar_contig32

    // Process 32 bytes at once using YMM register
    VMOVAPS (SI), Y0          // Y0 = [32 bytes of srcData]
    VMULPS Y15, Y0, Y0        // Y0 = Y0 * scalar
    VMOVAPS Y0, (DI)          // dstData = Y0
    ADDQ $32, SI
    ADDQ $32, DI
    SUBQ $32, R8
    JZ   done_scalar_contig32

remainder_lt32_scalar_contig32:
    // For remainder, skip processing (will be handled by Go fallback)
    // Remainder is small and conversion is complex for arbitrary types
    JMP  done_scalar_contig32

done_scalar_contig32:
    VZEROUPPER
    RET

// blazeReduceVectorSumF64ContiguousAVX2 computes sum for contiguous vector using AVX2
// Optimized with 2x loop unrolling and multiple accumulators
// Element-size-aware: processes 64 bytes per iteration (2 YMM registers) regardless of element size
// func blazeReduceVectorSumF64ContiguousAVX2(data unsafe.Pointer, size uintptr, capacity uint64) float64
TEXT ·blazeReduceVectorSumF64ContiguousAVX2(SB), NOSPLIT, $0-24
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
    JL   remainder_loop_sum_contig64

    // Calculate number of 64-byte iterations
    MOVQ R8, AX               // AX = total bytes
    SHRQ $6, AX               // AX = total bytes / 64 (process 64 bytes per iteration)
    JZ   remainder_loop_sum_contig64

    // Main loop: process 64 bytes at a time (2x unrolled = 2 YMM registers)
simd_loop_sum_contig64:
    // Load first 32 bytes (YMM register)
    VMOVAPD (SI), Y1          // Y1 = [32 bytes of data]
    
    // Load second 32 bytes (YMM register)
    VMOVAPD 32(SI), Y3        // Y3 = [32 bytes of data]

    // Accumulate on both accumulators in parallel
    VADDPD Y1, Y0, Y0         // Y0 += Y1 (first block)
    VADDPD Y3, Y2, Y2         // Y2 += Y3 (second block)

    // Advance pointer: SI += 64 (always 64 bytes regardless of element size)
    ADDQ $64, SI

    DECQ AX
    JNZ  simd_loop_sum_contig64

    // Combine accumulators: Y0 = Y0 + Y2
    VADDPD Y2, Y0, Y0         // Y0 = Y0 + Y2

    // Horizontal reduction: sum all 4 elements in Y0 using tree reduction
    VEXTRACTF128 $1, Y0, X1   // X1 = [Y0[2], Y0[3]]
    VADDPD X0, X1, X0         // X0 = [Y0[0]+Y0[2], Y0[1]+Y0[3]]
    VUNPCKHPD X0, X0, X1      // X1 = [Y0[1]+Y0[3], Y0[1]+Y0[3]]
    VADDPD X1, X0, X0         // X0 = [sum, sum]
    VMOVSD X0, ret+24(FP)     // Return sum

remainder_loop_sum_contig64:
    // Handle remaining bytes (< 64 bytes)
    // Calculate remaining bytes: (capacity * size) % 64
    MOVQ CX, AX               // AX = capacity
    MULQ DX                   // AX = capacity * size, DX = high bits
    MOVQ AX, R8               // R8 = total bytes
    ANDQ $63, R8              // R8 = total bytes % 64 (remaining bytes)
    JZ   done_sum_contig64

    // Load current sum into XMM0
    VMOVSD ret+24(FP), X0

    // Check if we have 32 or more remaining bytes
    CMPQ R8, $32
    JL   remainder_lt32_sum_contig64

    // Process 32 bytes at once using YMM register
    VMOVAPD (SI), Y1          // Y1 = [32 bytes of data]
    VADDPD Y1, Y0, Y0         // Y0 += Y1
    ADDQ $32, SI
    SUBQ $32, R8
    JZ   remainder_done_sum_contig64

remainder_lt32_sum_contig64:
    // For remainder, skip processing (will be handled by Go fallback)
    // Remainder is small and conversion is complex for arbitrary types
    JMP  remainder_done_sum_contig64

remainder_done_sum_contig64:
    // Extract final sum from Y0 if we processed 32 bytes in remainder
    MOVQ CX, AX
    MULQ DX
    MOVQ AX, R8
    ANDQ $63, R8
    CMPQ R8, $32
    JL   remainder_scalar_sum_contig64
    // We processed 32 bytes, need to reduce Y0
    VEXTRACTF128 $1, Y0, X1
    VADDPD X0, X1, X0
    VUNPCKHPD X0, X0, X1
    VADDPD X1, X0, X0
    JMP  remainder_store_sum_contig64

remainder_scalar_sum_contig64:
    // Only scalar operations were used, X0 already has the sum

remainder_store_sum_contig64:
    VMOVSD X0, ret+24(FP)

done_sum_contig64:
    VZEROUPPER
    RET

// blazeReduceVectorSumF32ContiguousAVX2 computes sum for contiguous vector using AVX2
// Optimized with 2x loop unrolling and multiple accumulators
// Element-size-aware: processes 64 bytes per iteration (2 YMM registers) regardless of element size
// func blazeReduceVectorSumF32ContiguousAVX2(data unsafe.Pointer, size uintptr, capacity uint64) float32
TEXT ·blazeReduceVectorSumF32ContiguousAVX2(SB), NOSPLIT, $0-24
    MOVQ data+0(FP), SI      // data pointer
    MOVQ size+8(FP), DX       // element size in bytes
    MOVQ capacity+16(FP), CX  // capacity (number of elements)

    // Initialize two accumulators to break dependency chains
    VXORPS Y0, Y0, Y0         // Y0 = accumulator 0
    VXORPS Y2, Y2, Y2         // Y2 = accumulator 1

    // Calculate total bytes to process
    MOVQ CX, AX               // AX = capacity
    MULQ DX                   // AX = capacity * size (total bytes), DX = high bits
    MOVQ AX, R8               // R8 = total bytes
    // Check if total bytes >= 64 (need at least 64 bytes for one iteration)
    CMPQ R8, $64
    JL   remainder_loop_sum_contig32

    // Calculate number of 64-byte iterations
    MOVQ R8, AX               // AX = total bytes
    SHRQ $6, AX               // AX = total bytes / 64 (process 64 bytes per iteration)
    JZ   remainder_loop_sum_contig32

    // Main loop: process 64 bytes at a time (2x unrolled = 2 YMM registers)
simd_loop_sum_contig32:
    // Load first 32 bytes (YMM register)
    VMOVAPS (SI), Y1          // Y1 = [32 bytes of data]
    
    // Load second 32 bytes (YMM register)
    VMOVAPS 32(SI), Y3        // Y3 = [32 bytes of data]

    // Accumulate on both accumulators in parallel
    VADDPS Y1, Y0, Y0         // Y0 += Y1 (first block)
    VADDPS Y3, Y2, Y2         // Y2 += Y3 (second block)

    // Advance pointer: SI += 64 (always 64 bytes regardless of element size)
    ADDQ $64, SI

    DECQ AX
    JNZ  simd_loop_sum_contig32

    // Combine accumulators: Y0 = Y0 + Y2
    VADDPS Y2, Y0, Y0         // Y0 = Y0 + Y2

    // Horizontal reduction: sum all 8 elements
    VEXTRACTF128 $1, Y0, X1   // X1 = [Y0[4], Y0[5], Y0[6], Y0[7]]
    VADDPS X0, X1, X0         // X0 = [sum0, sum1, sum2, sum3]
    VHADDPS X0, X0, X0        // X0 = [sum, sum, sum, sum]
    VHADDPS X0, X0, X0        // X0 = [final_sum, final_sum, final_sum, final_sum]
    VMOVSS X0, ret+24(FP)     // Return sum

remainder_loop_sum_contig32:
    // Handle remaining bytes (< 64 bytes)
    // Calculate remaining bytes: (capacity * size) % 64
    MOVQ CX, AX               // AX = capacity
    MULQ DX                   // AX = capacity * size, DX = high bits
    MOVQ AX, R8               // R8 = total bytes
    ANDQ $63, R8              // R8 = total bytes % 64 (remaining bytes)
    JZ   done_sum_contig32

    // Load current sum into XMM0
    VMOVSS ret+24(FP), X0

    // Check if we have 32 or more remaining bytes
    CMPQ R8, $32
    JL   remainder_lt32_sum_contig32

    // Process 32 bytes at once using YMM register
    VMOVAPS (SI), Y1          // Y1 = [32 bytes of data]
    VADDPS Y1, Y0, Y0         // Y0 += Y1
    ADDQ $32, SI
    SUBQ $32, R8
    JZ   remainder_done_sum_contig32

remainder_lt32_sum_contig32:
    // For remainder, skip processing (will be handled by Go fallback)
    // Remainder is small and conversion is complex for arbitrary types
    JMP  remainder_done_sum_contig32

remainder_done_sum_contig32:
    // Extract final sum from Y0 if we processed 32 bytes in remainder
    MOVQ CX, AX
    MULQ DX
    MOVQ AX, R8
    ANDQ $63, R8
    CMPQ R8, $32
    JL   remainder_scalar_sum_contig32
    // We processed 32 bytes, need to reduce Y0
    VEXTRACTF128 $1, Y0, X1
    VADDPS X0, X1, X0
    VHADDPS X0, X0, X0
    VHADDPS X0, X0, X0
    JMP  remainder_store_sum_contig32

remainder_scalar_sum_contig32:
    // Only scalar operations were used, X0 already has the sum

remainder_store_sum_contig32:
    VMOVSS X0, ret+24(FP)

done_sum_contig32:
    VZEROUPPER
    RET

