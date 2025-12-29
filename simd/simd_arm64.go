//go:build arm64

package simd

import "unsafe"

// Assembly function declarations for ARM64 (NEON)
// These are implemented in simd_arm64_neon.s

// Note: ARM64 SIMD implementations will be added in a future phase
// For now, these are placeholders that will fall back to Go implementations

// Reduction operations - SumSquared
//go:noescape
func blazeReduceVectorSumSquaredF64NEON(data unsafe.Pointer, size uintptr, capacity uint64) float64

//go:noescape
func blazeReduceVectorSumSquaredF32NEON(data unsafe.Pointer, size uintptr, capacity uint64) float32

// Reduction operations - DotProduct
//go:noescape
func blazeReduceDotProductF64NEON(aData, bData unsafe.Pointer, aSize, bSize uintptr, capacity uint64) float64

//go:noescape
func blazeReduceDotProductF32NEON(aData, bData unsafe.Pointer, aSize, bSize uintptr, capacity uint64) float32

// Reduction operations - Sum
//go:noescape
func blazeReduceVectorSumF64NEON(data unsafe.Pointer, size uintptr, capacity uint64) float64

//go:noescape
func blazeReduceVectorSumF32NEON(data unsafe.Pointer, size uintptr, capacity uint64) float32

// Elementwise operations - Add
//go:noescape
func blazeElementWiseVectorAddF32NEON(aData, bData, dstData unsafe.Pointer, aSize, bSize, dstSize uintptr, capacity uint64)

//go:noescape
func blazeElementWiseVectorAddF64NEON(aData, bData, dstData unsafe.Pointer, aSize, bSize, dstSize uintptr, capacity uint64)

// Elementwise operations - Multiply
//go:noescape
func blazeElementWiseVectorMultiplyF32NEON(aData, bData, dstData unsafe.Pointer, aSize, bSize, dstSize uintptr, capacity uint64)

//go:noescape
func blazeElementWiseVectorMultiplyF64NEON(aData, bData, dstData unsafe.Pointer, aSize, bSize, dstSize uintptr, capacity uint64)

// Scalar operations - Multiply
//go:noescape
func blazeScalarVectorMultiplyF32NEON(srcData, dstData unsafe.Pointer, srcSize, dstSize uintptr, scalar float32, capacity uint64)

//go:noescape
func blazeScalarVectorMultiplyF64NEON(srcData, dstData unsafe.Pointer, srcSize, dstSize uintptr, scalar float64, capacity uint64)

