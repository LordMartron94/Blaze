//go:build !amd64

package simd

import "unsafe"

// dispatchTable holds function pointers for all SIMD operations
// On non-amd64 platforms, all function pointers are nil and Go fallback is used
type dispatchTable struct {
	// Reduction - SumSquared F64
	sumSquaredF64Contiguous func(unsafe.Pointer, uint64) float64
	sumSquaredF64Strided   func(unsafe.Pointer, uintptr, uint64) float64

	// Reduction - SumSquared F32
	sumSquaredF32Contiguous func(unsafe.Pointer, uint64) float32
	sumSquaredF32Strided     func(unsafe.Pointer, uintptr, uint64) float32

	// Reduction - DotProduct F64
	dotProductF64Contiguous func(unsafe.Pointer, unsafe.Pointer, uint64) float64
	dotProductF64Strided    func(unsafe.Pointer, unsafe.Pointer, uintptr, uintptr, uint64) float64

	// Reduction - DotProduct F32
	dotProductF32Contiguous func(unsafe.Pointer, unsafe.Pointer, uint64) float32
	dotProductF32Strided    func(unsafe.Pointer, unsafe.Pointer, uintptr, uintptr, uint64) float32

	// Reduction - Sum F64
	sumF64Contiguous func(unsafe.Pointer, uint64) float64
	sumF64Strided    func(unsafe.Pointer, uintptr, uint64) float64

	// Reduction - Sum F32
	sumF32Contiguous func(unsafe.Pointer, uint64) float32
	sumF32Strided    func(unsafe.Pointer, uintptr, uint64) float32

	// Elementwise - Add F32
	addF32Contiguous func(unsafe.Pointer, unsafe.Pointer, unsafe.Pointer, uint64)
	addF32Strided   func(unsafe.Pointer, unsafe.Pointer, unsafe.Pointer, uintptr, uintptr, uintptr, uint64)

	// Elementwise - Add F64
	addF64Contiguous func(unsafe.Pointer, unsafe.Pointer, unsafe.Pointer, uint64)
	addF64Strided    func(unsafe.Pointer, unsafe.Pointer, unsafe.Pointer, uintptr, uintptr, uintptr, uint64)

	// Elementwise - Multiply F32
	multiplyF32Contiguous func(unsafe.Pointer, unsafe.Pointer, unsafe.Pointer, uint64)
	multiplyF32Strided    func(unsafe.Pointer, unsafe.Pointer, unsafe.Pointer, uintptr, uintptr, uintptr, uint64)

	// Elementwise - Multiply F64
	multiplyF64Contiguous func(unsafe.Pointer, unsafe.Pointer, unsafe.Pointer, uint64)
	multiplyF64Strided    func(unsafe.Pointer, unsafe.Pointer, unsafe.Pointer, uintptr, uintptr, uintptr, uint64)

	// Scalar - Multiply F32
	scalarMultiplyF32Contiguous func(unsafe.Pointer, unsafe.Pointer, float32, uint64)
	scalarMultiplyF32Strided     func(unsafe.Pointer, unsafe.Pointer, uintptr, uintptr, float32, uint64)

	// Scalar - Multiply F64
	scalarMultiplyF64Contiguous func(unsafe.Pointer, unsafe.Pointer, float64, uint64)
	scalarMultiplyF64Strided     func(unsafe.Pointer, unsafe.Pointer, uintptr, uintptr, float64, uint64)
}

var defaultDispatchTable dispatchTable

// OverrideDispatchTable allows overriding the dispatch table for testing/benchmarking
// On non-amd64 platforms, this is a no-op
func OverrideDispatchTable(override func(*dispatchTable)) {
	// No-op on non-amd64 platforms
}

// ResetDispatchTable resets the dispatch table to the default
// On non-amd64 platforms, this is a no-op
func ResetDispatchTable() {
	// No-op on non-amd64 platforms
}

