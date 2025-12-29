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

