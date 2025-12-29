//go:build amd64

package simd

import "unsafe"

// Assembly function declarations for x86_64
// These are implemented in simd_amd64_avx2.s and simd_amd64_avx.s

// Reduction operations - SumSquared
//go:noescape
func blazeReduceVectorSumSquaredF64AVX2(data unsafe.Pointer, size uintptr, capacity uint64) float64

//go:noescape
func blazeReduceVectorSumSquaredF64AVX(data unsafe.Pointer, size uintptr, capacity uint64) float64

//go:noescape
func blazeReduceVectorSumSquaredF64SSE41(data unsafe.Pointer, size uintptr, capacity uint64) float64

//go:noescape
func blazeReduceVectorSumSquaredF32AVX2(data unsafe.Pointer, size uintptr, capacity uint64) float32

//go:noescape
func blazeReduceVectorSumSquaredF32AVX(data unsafe.Pointer, size uintptr, capacity uint64) float32

//go:noescape
func blazeReduceVectorSumSquaredF32SSE41(data unsafe.Pointer, size uintptr, capacity uint64) float32

// Reduction operations - DotProduct
//go:noescape
func blazeReduceDotProductF64AVX2(aData, bData unsafe.Pointer, aSize, bSize uintptr, capacity uint64) float64

//go:noescape
func blazeReduceDotProductF64AVX(aData, bData unsafe.Pointer, aSize, bSize uintptr, capacity uint64) float64

//go:noescape
func blazeReduceDotProductF64SSE41(aData, bData unsafe.Pointer, aSize, bSize uintptr, capacity uint64) float64

//go:noescape
func blazeReduceDotProductF32AVX2(aData, bData unsafe.Pointer, aSize, bSize uintptr, capacity uint64) float32

//go:noescape
func blazeReduceDotProductF32AVX(aData, bData unsafe.Pointer, aSize, bSize uintptr, capacity uint64) float32

//go:noescape
func blazeReduceDotProductF32SSE41(aData, bData unsafe.Pointer, aSize, bSize uintptr, capacity uint64) float32

// Reduction operations - Sum
//go:noescape
func blazeReduceVectorSumF64AVX2(data unsafe.Pointer, size uintptr, capacity uint64) float64

//go:noescape
func blazeReduceVectorSumF64AVX(data unsafe.Pointer, size uintptr, capacity uint64) float64

//go:noescape
func blazeReduceVectorSumF64SSE41(data unsafe.Pointer, size uintptr, capacity uint64) float64

//go:noescape
func blazeReduceVectorSumF32AVX2(data unsafe.Pointer, size uintptr, capacity uint64) float32

//go:noescape
func blazeReduceVectorSumF32AVX(data unsafe.Pointer, size uintptr, capacity uint64) float32

//go:noescape
func blazeReduceVectorSumF32SSE41(data unsafe.Pointer, size uintptr, capacity uint64) float32

// Elementwise operations - Add
//go:noescape
func blazeElementWiseVectorAddF32AVX2(aData, bData, dstData unsafe.Pointer, aSize, bSize, dstSize uintptr, capacity uint64)

//go:noescape
func blazeElementWiseVectorAddF32AVX(aData, bData, dstData unsafe.Pointer, aSize, bSize, dstSize uintptr, capacity uint64)

//go:noescape
func blazeElementWiseVectorAddF32SSE41(aData, bData, dstData unsafe.Pointer, aSize, bSize, dstSize uintptr, capacity uint64)

//go:noescape
func blazeElementWiseVectorAddF64AVX2(aData, bData, dstData unsafe.Pointer, aSize, bSize, dstSize uintptr, capacity uint64)

//go:noescape
func blazeElementWiseVectorAddF64AVX(aData, bData, dstData unsafe.Pointer, aSize, bSize, dstSize uintptr, capacity uint64)

//go:noescape
func blazeElementWiseVectorAddF64SSE41(aData, bData, dstData unsafe.Pointer, aSize, bSize, dstSize uintptr, capacity uint64)

// Elementwise operations - Multiply
//go:noescape
func blazeElementWiseVectorMultiplyF32AVX2(aData, bData, dstData unsafe.Pointer, aSize, bSize, dstSize uintptr, capacity uint64)

//go:noescape
func blazeElementWiseVectorMultiplyF32AVX(aData, bData, dstData unsafe.Pointer, aSize, bSize, dstSize uintptr, capacity uint64)

//go:noescape
func blazeElementWiseVectorMultiplyF32SSE41(aData, bData, dstData unsafe.Pointer, aSize, bSize, dstSize uintptr, capacity uint64)

//go:noescape
func blazeElementWiseVectorMultiplyF64AVX2(aData, bData, dstData unsafe.Pointer, aSize, bSize, dstSize uintptr, capacity uint64)

//go:noescape
func blazeElementWiseVectorMultiplyF64AVX(aData, bData, dstData unsafe.Pointer, aSize, bSize, dstSize uintptr, capacity uint64)

//go:noescape
func blazeElementWiseVectorMultiplyF64SSE41(aData, bData, dstData unsafe.Pointer, aSize, bSize, dstSize uintptr, capacity uint64)

// Scalar operations - Multiply
//go:noescape
func blazeScalarVectorMultiplyF32AVX2(srcData, dstData unsafe.Pointer, srcSize, dstSize uintptr, scalar float32, capacity uint64)

//go:noescape
func blazeScalarVectorMultiplyF32AVX(srcData, dstData unsafe.Pointer, srcSize, dstSize uintptr, scalar float32, capacity uint64)

//go:noescape
func blazeScalarVectorMultiplyF32SSE41(srcData, dstData unsafe.Pointer, srcSize, dstSize uintptr, scalar float32, capacity uint64)

//go:noescape
func blazeScalarVectorMultiplyF64AVX2(srcData, dstData unsafe.Pointer, srcSize, dstSize uintptr, scalar float64, capacity uint64)

//go:noescape
func blazeScalarVectorMultiplyF64AVX(srcData, dstData unsafe.Pointer, srcSize, dstSize uintptr, scalar float64, capacity uint64)

//go:noescape
func blazeScalarVectorMultiplyF64SSE41(srcData, dstData unsafe.Pointer, srcSize, dstSize uintptr, scalar float64, capacity uint64)

// Contiguous access fast paths - these assume stride == sizeof(T) for optimal performance
// Element-size-aware: process 64 bytes per iteration regardless of element size
// Reduction operations - SumSquared (contiguous)
//go:noescape
func blazeReduceVectorSumSquaredF64ContiguousAVX2(data unsafe.Pointer, size uintptr, capacity uint64) float64

//go:noescape
func blazeReduceVectorSumSquaredF32ContiguousAVX2(data unsafe.Pointer, size uintptr, capacity uint64) float32

// Reduction operations - DotProduct (contiguous)
//go:noescape
func blazeReduceDotProductF64ContiguousAVX2(aData, bData unsafe.Pointer, size uintptr, capacity uint64) float64

//go:noescape
func blazeReduceDotProductF32ContiguousAVX2(aData, bData unsafe.Pointer, size uintptr, capacity uint64) float32

// Reduction operations - Sum (contiguous)
//go:noescape
func blazeReduceVectorSumF64ContiguousAVX2(data unsafe.Pointer, size uintptr, capacity uint64) float64

//go:noescape
func blazeReduceVectorSumF32ContiguousAVX2(data unsafe.Pointer, size uintptr, capacity uint64) float32

// Elementwise operations - Add (contiguous)
//go:noescape
func blazeElementWiseVectorAddF64ContiguousAVX2(aData, bData, dstData unsafe.Pointer, size uintptr, capacity uint64)

//go:noescape
func blazeElementWiseVectorAddF32ContiguousAVX2(aData, bData, dstData unsafe.Pointer, size uintptr, capacity uint64)

// Elementwise operations - Multiply (contiguous)
//go:noescape
func blazeElementWiseVectorMultiplyF64ContiguousAVX2(aData, bData, dstData unsafe.Pointer, size uintptr, capacity uint64)

//go:noescape
func blazeElementWiseVectorMultiplyF32ContiguousAVX2(aData, bData, dstData unsafe.Pointer, size uintptr, capacity uint64)

// Scalar operations - Multiply (contiguous)
//go:noescape
func blazeScalarVectorMultiplyF64ContiguousAVX2(srcData, dstData unsafe.Pointer, size uintptr, scalar float64, capacity uint64)

//go:noescape
func blazeScalarVectorMultiplyF32ContiguousAVX2(srcData, dstData unsafe.Pointer, size uintptr, scalar float32, capacity uint64)

// FMA-optimized operations (require AVX2 + FMA)
// Element-size-aware: process 64 bytes per iteration regardless of element size
// Reduction operations - SumSquared (FMA)
//go:noescape
func blazeReduceVectorSumSquaredF64FMA(data unsafe.Pointer, size uintptr, capacity uint64) float64

//go:noescape
func blazeReduceVectorSumSquaredF32FMA(data unsafe.Pointer, size uintptr, capacity uint64) float32

// Reduction operations - DotProduct (FMA)
//go:noescape
func blazeReduceDotProductF64FMA(aData, bData unsafe.Pointer, size uintptr, capacity uint64) float64

//go:noescape
func blazeReduceDotProductF32FMA(aData, bData unsafe.Pointer, size uintptr, capacity uint64) float32

