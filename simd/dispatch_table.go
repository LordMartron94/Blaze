//go:build amd64

package simd

import (
	"unsafe"

	"golang.org/x/sys/cpu"
)

// dispatchTable holds function pointers for all SIMD operations
// These are initialized once at package init based on CPU features
type dispatchTable struct {
	// Reduction - SumSquared F64
	sumSquaredF64Contiguous func(unsafe.Pointer, uintptr, uint64) float64
	sumSquaredF64Strided    func(unsafe.Pointer, uintptr, uint64) float64

	// Reduction - SumSquared F32
	sumSquaredF32Contiguous func(unsafe.Pointer, uintptr, uint64) float32
	sumSquaredF32Strided    func(unsafe.Pointer, uintptr, uint64) float32

	// Reduction - DotProduct F64
	dotProductF64Contiguous func(unsafe.Pointer, unsafe.Pointer, uintptr, uint64) float64
	dotProductF64Strided    func(unsafe.Pointer, unsafe.Pointer, uintptr, uintptr, uint64) float64

	// Reduction - DotProduct F32
	dotProductF32Contiguous func(unsafe.Pointer, unsafe.Pointer, uintptr, uint64) float32
	dotProductF32Strided    func(unsafe.Pointer, unsafe.Pointer, uintptr, uintptr, uint64) float32

	// Reduction - Sum F64
	sumF64Contiguous func(unsafe.Pointer, uintptr, uint64) float64
	sumF64Strided    func(unsafe.Pointer, uintptr, uint64) float64

	// Reduction - Sum F32
	sumF32Contiguous func(unsafe.Pointer, uintptr, uint64) float32
	sumF32Strided    func(unsafe.Pointer, uintptr, uint64) float32

	// Elementwise - Add F32
	addF32Contiguous func(unsafe.Pointer, unsafe.Pointer, unsafe.Pointer, uintptr, uint64)
	addF32Strided    func(unsafe.Pointer, unsafe.Pointer, unsafe.Pointer, uintptr, uintptr, uintptr, uint64)

	// Elementwise - Add F64
	addF64Contiguous func(unsafe.Pointer, unsafe.Pointer, unsafe.Pointer, uintptr, uint64)
	addF64Strided    func(unsafe.Pointer, unsafe.Pointer, unsafe.Pointer, uintptr, uintptr, uintptr, uint64)

	// Elementwise - Multiply F32
	multiplyF32Contiguous func(unsafe.Pointer, unsafe.Pointer, unsafe.Pointer, uintptr, uint64)
	multiplyF32Strided    func(unsafe.Pointer, unsafe.Pointer, unsafe.Pointer, uintptr, uintptr, uintptr, uint64)

	// Elementwise - Multiply F64
	multiplyF64Contiguous func(unsafe.Pointer, unsafe.Pointer, unsafe.Pointer, uintptr, uint64)
	multiplyF64Strided    func(unsafe.Pointer, unsafe.Pointer, unsafe.Pointer, uintptr, uintptr, uintptr, uint64)

	// Scalar - Multiply F32
	scalarMultiplyF32Contiguous func(unsafe.Pointer, unsafe.Pointer, uintptr, float32, uint64)
	scalarMultiplyF32Strided    func(unsafe.Pointer, unsafe.Pointer, uintptr, uintptr, float32, uint64)

	// Scalar - Multiply F64
	scalarMultiplyF64Contiguous func(unsafe.Pointer, unsafe.Pointer, uintptr, float64, uint64)
	scalarMultiplyF64Strided    func(unsafe.Pointer, unsafe.Pointer, uintptr, uintptr, float64, uint64)
}

var defaultDispatchTable dispatchTable

// init initializes the dispatch table based on CPU features
func init() {
	initDispatchTable(&defaultDispatchTable)
}

// initDispatchTable initializes a dispatch table based on CPU features
func initDispatchTable(table *dispatchTable) {
	// Initialize based on best available SIMD instruction set
	if cpu.X86.HasFMA && cpu.X86.HasAVX2 {
		// FMA + AVX2 (best performance)
		initDispatchTableFMA(table)
	} else if cpu.X86.HasAVX2 {
		// AVX2 (very good performance)
		initDispatchTableAVX2(table)
	} else if cpu.X86.HasAVX {
		// AVX (good performance)
		initDispatchTableAVX(table)
	} else if cpu.X86.HasSSE41 {
		// SSE4.1 (decent performance)
		initDispatchTableSSE41(table)
	} else {
		// No SIMD support - will use Go fallback
		initDispatchTableGo(table)
	}
}

func initDispatchTableFMA(table *dispatchTable) {
	// FMA-optimized contiguous paths
	table.sumSquaredF64Contiguous = blazeReduceVectorSumSquaredF64FMA
	table.sumSquaredF32Contiguous = blazeReduceVectorSumSquaredF32FMA
	table.dotProductF64Contiguous = blazeReduceDotProductF64FMA
	table.dotProductF32Contiguous = blazeReduceDotProductF32FMA

	// AVX2 contiguous paths (for operations without FMA)
	table.sumF64Contiguous = blazeReduceVectorSumF64ContiguousAVX2
	table.sumF32Contiguous = blazeReduceVectorSumF32ContiguousAVX2
	table.addF32Contiguous = blazeElementWiseVectorAddF32ContiguousAVX2
	table.addF64Contiguous = blazeElementWiseVectorAddF64ContiguousAVX2
	table.multiplyF32Contiguous = blazeElementWiseVectorMultiplyF32ContiguousAVX2
	table.multiplyF64Contiguous = blazeElementWiseVectorMultiplyF64ContiguousAVX2
	table.scalarMultiplyF32Contiguous = blazeScalarVectorMultiplyF32ContiguousAVX2
	table.scalarMultiplyF64Contiguous = blazeScalarVectorMultiplyF64ContiguousAVX2

	// AVX2 strided paths
	table.sumSquaredF64Strided = blazeReduceVectorSumSquaredF64AVX2
	table.sumSquaredF32Strided = blazeReduceVectorSumSquaredF32AVX2
	table.dotProductF64Strided = blazeReduceDotProductF64AVX2
	table.dotProductF32Strided = blazeReduceDotProductF32AVX2
	table.sumF64Strided = blazeReduceVectorSumF64AVX2
	table.sumF32Strided = blazeReduceVectorSumF32AVX2
	table.addF32Strided = blazeElementWiseVectorAddF32AVX2
	table.addF64Strided = blazeElementWiseVectorAddF64AVX2
	table.multiplyF32Strided = blazeElementWiseVectorMultiplyF32AVX2
	table.multiplyF64Strided = blazeElementWiseVectorMultiplyF64AVX2
	table.scalarMultiplyF32Strided = blazeScalarVectorMultiplyF32AVX2
	table.scalarMultiplyF64Strided = blazeScalarVectorMultiplyF64AVX2
}

func initDispatchTableAVX2(table *dispatchTable) {
	// AVX2 contiguous paths
	table.sumSquaredF64Contiguous = blazeReduceVectorSumSquaredF64ContiguousAVX2
	table.sumSquaredF32Contiguous = blazeReduceVectorSumSquaredF32ContiguousAVX2
	table.dotProductF64Contiguous = blazeReduceDotProductF64ContiguousAVX2
	table.dotProductF32Contiguous = blazeReduceDotProductF32ContiguousAVX2
	table.sumF64Contiguous = blazeReduceVectorSumF64ContiguousAVX2
	table.sumF32Contiguous = blazeReduceVectorSumF32ContiguousAVX2
	table.addF32Contiguous = blazeElementWiseVectorAddF32ContiguousAVX2
	table.addF64Contiguous = blazeElementWiseVectorAddF64ContiguousAVX2
	table.multiplyF32Contiguous = blazeElementWiseVectorMultiplyF32ContiguousAVX2
	table.multiplyF64Contiguous = blazeElementWiseVectorMultiplyF64ContiguousAVX2
	table.scalarMultiplyF32Contiguous = blazeScalarVectorMultiplyF32ContiguousAVX2
	table.scalarMultiplyF64Contiguous = blazeScalarVectorMultiplyF64ContiguousAVX2

	// AVX2 strided paths
	table.sumSquaredF64Strided = blazeReduceVectorSumSquaredF64AVX2
	table.sumSquaredF32Strided = blazeReduceVectorSumSquaredF32AVX2
	table.dotProductF64Strided = blazeReduceDotProductF64AVX2
	table.dotProductF32Strided = blazeReduceDotProductF32AVX2
	table.sumF64Strided = blazeReduceVectorSumF64AVX2
	table.sumF32Strided = blazeReduceVectorSumF32AVX2
	table.addF32Strided = blazeElementWiseVectorAddF32AVX2
	table.addF64Strided = blazeElementWiseVectorAddF64AVX2
	table.multiplyF32Strided = blazeElementWiseVectorMultiplyF32AVX2
	table.multiplyF64Strided = blazeElementWiseVectorMultiplyF64AVX2
	table.scalarMultiplyF32Strided = blazeScalarVectorMultiplyF32AVX2
	table.scalarMultiplyF64Strided = blazeScalarVectorMultiplyF64AVX2
}

func initDispatchTableAVX(table *dispatchTable) {
	// AVX doesn't have contiguous paths, use strided for both
	table.sumSquaredF64Contiguous = func(data unsafe.Pointer, size uintptr, capacity uint64) float64 {
		return blazeReduceVectorSumSquaredF64AVX(data, size, capacity)
	}
	table.sumSquaredF32Contiguous = func(data unsafe.Pointer, size uintptr, capacity uint64) float32 {
		return blazeReduceVectorSumSquaredF32AVX(data, size, capacity)
	}
	table.dotProductF64Contiguous = func(aData, bData unsafe.Pointer, size uintptr, capacity uint64) float64 {
		return blazeReduceDotProductF64AVX(aData, bData, size, size, capacity)
	}
	table.dotProductF32Contiguous = func(aData, bData unsafe.Pointer, size uintptr, capacity uint64) float32 {
		return blazeReduceDotProductF32AVX(aData, bData, size, size, capacity)
	}
	table.sumF64Contiguous = func(data unsafe.Pointer, size uintptr, capacity uint64) float64 {
		return blazeReduceVectorSumF64AVX(data, size, capacity)
	}
	table.sumF32Contiguous = func(data unsafe.Pointer, size uintptr, capacity uint64) float32 {
		return blazeReduceVectorSumF32AVX(data, size, capacity)
	}
	table.addF32Contiguous = func(aData, bData, dstData unsafe.Pointer, size uintptr, capacity uint64) {
		blazeElementWiseVectorAddF32AVX(aData, bData, dstData, size, size, size, capacity)
	}
	table.addF64Contiguous = func(aData, bData, dstData unsafe.Pointer, size uintptr, capacity uint64) {
		blazeElementWiseVectorAddF64AVX(aData, bData, dstData, size, size, size, capacity)
	}
	table.multiplyF32Contiguous = func(aData, bData, dstData unsafe.Pointer, size uintptr, capacity uint64) {
		blazeElementWiseVectorMultiplyF32AVX(aData, bData, dstData, size, size, size, capacity)
	}
	table.multiplyF64Contiguous = func(aData, bData, dstData unsafe.Pointer, size uintptr, capacity uint64) {
		blazeElementWiseVectorMultiplyF64AVX(aData, bData, dstData, size, size, size, capacity)
	}
	table.scalarMultiplyF32Contiguous = func(srcData, dstData unsafe.Pointer, size uintptr, scalar float32, capacity uint64) {
		blazeScalarVectorMultiplyF32AVX(srcData, dstData, size, size, scalar, capacity)
	}
	table.scalarMultiplyF64Contiguous = func(srcData, dstData unsafe.Pointer, size uintptr, scalar float64, capacity uint64) {
		blazeScalarVectorMultiplyF64AVX(srcData, dstData, size, size, scalar, capacity)
	}

	// AVX strided paths
	table.sumSquaredF64Strided = blazeReduceVectorSumSquaredF64AVX
	table.sumSquaredF32Strided = blazeReduceVectorSumSquaredF32AVX
	table.dotProductF64Strided = blazeReduceDotProductF64AVX
	table.dotProductF32Strided = blazeReduceDotProductF32AVX
	table.sumF64Strided = blazeReduceVectorSumF64AVX
	table.sumF32Strided = blazeReduceVectorSumF32AVX
	table.addF32Strided = blazeElementWiseVectorAddF32AVX
	table.addF64Strided = blazeElementWiseVectorAddF64AVX
	table.multiplyF32Strided = blazeElementWiseVectorMultiplyF32AVX
	table.multiplyF64Strided = blazeElementWiseVectorMultiplyF64AVX
	table.scalarMultiplyF32Strided = blazeScalarVectorMultiplyF32AVX
	table.scalarMultiplyF64Strided = blazeScalarVectorMultiplyF64AVX
}

func initDispatchTableSSE41(table *dispatchTable) {
	// SSE4.1 doesn't have contiguous paths, use strided for both
	table.sumSquaredF64Contiguous = func(data unsafe.Pointer, size uintptr, capacity uint64) float64 {
		return blazeReduceVectorSumSquaredF64SSE41(data, size, capacity)
	}
	table.sumSquaredF32Contiguous = func(data unsafe.Pointer, size uintptr, capacity uint64) float32 {
		return blazeReduceVectorSumSquaredF32SSE41(data, size, capacity)
	}
	table.dotProductF64Contiguous = func(aData, bData unsafe.Pointer, size uintptr, capacity uint64) float64 {
		return blazeReduceDotProductF64SSE41(aData, bData, size, size, capacity)
	}
	table.dotProductF32Contiguous = func(aData, bData unsafe.Pointer, size uintptr, capacity uint64) float32 {
		return blazeReduceDotProductF32SSE41(aData, bData, size, size, capacity)
	}
	table.sumF64Contiguous = func(data unsafe.Pointer, size uintptr, capacity uint64) float64 {
		return blazeReduceVectorSumF64SSE41(data, size, capacity)
	}
	table.sumF32Contiguous = func(data unsafe.Pointer, size uintptr, capacity uint64) float32 {
		return blazeReduceVectorSumF32SSE41(data, size, capacity)
	}
	table.addF32Contiguous = func(aData, bData, dstData unsafe.Pointer, size uintptr, capacity uint64) {
		blazeElementWiseVectorAddF32SSE41(aData, bData, dstData, size, size, size, capacity)
	}
	table.addF64Contiguous = func(aData, bData, dstData unsafe.Pointer, size uintptr, capacity uint64) {
		blazeElementWiseVectorAddF64SSE41(aData, bData, dstData, size, size, size, capacity)
	}
	table.multiplyF32Contiguous = func(aData, bData, dstData unsafe.Pointer, size uintptr, capacity uint64) {
		blazeElementWiseVectorMultiplyF32SSE41(aData, bData, dstData, size, size, size, capacity)
	}
	table.multiplyF64Contiguous = func(aData, bData, dstData unsafe.Pointer, size uintptr, capacity uint64) {
		blazeElementWiseVectorMultiplyF64SSE41(aData, bData, dstData, size, size, size, capacity)
	}
	table.scalarMultiplyF32Contiguous = func(srcData, dstData unsafe.Pointer, size uintptr, scalar float32, capacity uint64) {
		blazeScalarVectorMultiplyF32SSE41(srcData, dstData, size, size, scalar, capacity)
	}
	table.scalarMultiplyF64Contiguous = func(srcData, dstData unsafe.Pointer, size uintptr, scalar float64, capacity uint64) {
		blazeScalarVectorMultiplyF64SSE41(srcData, dstData, size, size, scalar, capacity)
	}

	// SSE4.1 strided paths
	table.sumSquaredF64Strided = blazeReduceVectorSumSquaredF64SSE41
	table.sumSquaredF32Strided = blazeReduceVectorSumSquaredF32SSE41
	table.dotProductF64Strided = blazeReduceDotProductF64SSE41
	table.dotProductF32Strided = blazeReduceDotProductF32SSE41
	table.sumF64Strided = blazeReduceVectorSumF64SSE41
	table.sumF32Strided = blazeReduceVectorSumF32SSE41
	table.addF32Strided = blazeElementWiseVectorAddF32SSE41
	table.addF64Strided = blazeElementWiseVectorAddF64SSE41
	table.multiplyF32Strided = blazeElementWiseVectorMultiplyF32SSE41
	table.multiplyF64Strided = blazeElementWiseVectorMultiplyF64SSE41
	table.scalarMultiplyF32Strided = blazeScalarVectorMultiplyF32SSE41
	table.scalarMultiplyF64Strided = blazeScalarVectorMultiplyF64SSE41
}

func initDispatchTableGo(table *dispatchTable) {
	// No SIMD - all functions will use Go fallback (passed as parameter)
	// These will be nil and dispatch functions will check for nil
}

// OverrideDispatchTable allows overriding the dispatch table for testing/benchmarking
// This is useful for comparing different implementations or forcing specific code paths
// The override is not thread-safe and should only be used in single-threaded contexts (tests/benchmarks)
func OverrideDispatchTable(override func(*dispatchTable)) {
	override(&defaultDispatchTable)
}

// ResetDispatchTable resets the dispatch table to the default based on CPU features
func ResetDispatchTable() {
	initDispatchTable(&defaultDispatchTable)
}
