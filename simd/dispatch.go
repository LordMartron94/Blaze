package simd

import (
	"unsafe"

	"golang.org/x/sys/cpu"
)

// Dispatch functions select the best available SIMD implementation
// They fall back to Go implementations (passed as function parameters) when SIMD is unavailable

func BlazeReduceVectorSumSquaredF64Dispatch(
	data unsafe.Pointer, size uintptr, capacity uint64,
	goImpl func(unsafe.Pointer, uintptr, uint64) float64,
) float64 {
	if cpu.X86.HasAVX2 {
		return blazeReduceVectorSumSquaredF64AVX2(data, size, capacity)
	} else if cpu.X86.HasAVX {
		return blazeReduceVectorSumSquaredF64AVX(data, size, capacity)
	} else if cpu.X86.HasSSE41 {
		return blazeReduceVectorSumSquaredF64SSE41(data, size, capacity)
	}
	return goImpl(data, size, capacity)
}

func BlazeReduceVectorSumSquaredF32Dispatch(
	data unsafe.Pointer, size uintptr, capacity uint64,
	goImpl func(unsafe.Pointer, uintptr, uint64) float32,
) float32 {
	if cpu.X86.HasAVX2 {
		return blazeReduceVectorSumSquaredF32AVX2(data, size, capacity)
	} else if cpu.X86.HasAVX {
		return blazeReduceVectorSumSquaredF32AVX(data, size, capacity)
	} else if cpu.X86.HasSSE41 {
		return blazeReduceVectorSumSquaredF32SSE41(data, size, capacity)
	}
	return goImpl(data, size, capacity)
}

func BlazeReduceDotProductF64Dispatch(
	aData, bData unsafe.Pointer, aSize, bSize uintptr, capacity uint64,
	goImpl func(unsafe.Pointer, unsafe.Pointer, uintptr, uintptr, uint64) float64,
) float64 {
	if cpu.X86.HasAVX2 {
		return blazeReduceDotProductF64AVX2(aData, bData, aSize, bSize, capacity)
	} else if cpu.X86.HasAVX {
		return blazeReduceDotProductF64AVX(aData, bData, aSize, bSize, capacity)
	} else if cpu.X86.HasSSE41 {
		return blazeReduceDotProductF64SSE41(aData, bData, aSize, bSize, capacity)
	}
	return goImpl(aData, bData, aSize, bSize, capacity)
}

func BlazeReduceDotProductF32Dispatch(
	aData, bData unsafe.Pointer, aSize, bSize uintptr, capacity uint64,
	goImpl func(unsafe.Pointer, unsafe.Pointer, uintptr, uintptr, uint64) float32,
) float32 {
	if cpu.X86.HasAVX2 {
		return blazeReduceDotProductF32AVX2(aData, bData, aSize, bSize, capacity)
	} else if cpu.X86.HasAVX {
		return blazeReduceDotProductF32AVX(aData, bData, aSize, bSize, capacity)
	} else if cpu.X86.HasSSE41 {
		return blazeReduceDotProductF32SSE41(aData, bData, aSize, bSize, capacity)
	}
	return goImpl(aData, bData, aSize, bSize, capacity)
}

func BlazeElementWiseVectorAddF32Dispatch(
	aData, bData, dstData unsafe.Pointer, aSize, bSize, dstSize uintptr, capacity uint64,
	goImpl func(unsafe.Pointer, unsafe.Pointer, unsafe.Pointer, uintptr, uintptr, uintptr, uint64),
) {
	if cpu.X86.HasAVX2 {
		blazeElementWiseVectorAddF32AVX2(aData, bData, dstData, aSize, bSize, dstSize, capacity)
	} else if cpu.X86.HasAVX {
		blazeElementWiseVectorAddF32AVX(aData, bData, dstData, aSize, bSize, dstSize, capacity)
	} else if cpu.X86.HasSSE41 {
		blazeElementWiseVectorAddF32SSE41(aData, bData, dstData, aSize, bSize, dstSize, capacity)
	} else {
		goImpl(aData, bData, dstData, aSize, bSize, dstSize, capacity)
	}
}

func BlazeElementWiseVectorAddF64Dispatch(
	aData, bData, dstData unsafe.Pointer, aSize, bSize, dstSize uintptr, capacity uint64,
	goImpl func(unsafe.Pointer, unsafe.Pointer, unsafe.Pointer, uintptr, uintptr, uintptr, uint64),
) {
	if cpu.X86.HasAVX2 {
		blazeElementWiseVectorAddF64AVX2(aData, bData, dstData, aSize, bSize, dstSize, capacity)
	} else if cpu.X86.HasAVX {
		blazeElementWiseVectorAddF64AVX(aData, bData, dstData, aSize, bSize, dstSize, capacity)
	} else if cpu.X86.HasSSE41 {
		blazeElementWiseVectorAddF64SSE41(aData, bData, dstData, aSize, bSize, dstSize, capacity)
	} else {
		goImpl(aData, bData, dstData, aSize, bSize, dstSize, capacity)
	}
}

func BlazeElementWiseVectorMultiplyF32Dispatch(
	aData, bData, dstData unsafe.Pointer, aSize, bSize, dstSize uintptr, capacity uint64,
	goImpl func(unsafe.Pointer, unsafe.Pointer, unsafe.Pointer, uintptr, uintptr, uintptr, uint64),
) {
	if cpu.X86.HasAVX2 {
		blazeElementWiseVectorMultiplyF32AVX2(aData, bData, dstData, aSize, bSize, dstSize, capacity)
	} else if cpu.X86.HasAVX {
		blazeElementWiseVectorMultiplyF32AVX(aData, bData, dstData, aSize, bSize, dstSize, capacity)
	} else if cpu.X86.HasSSE41 {
		blazeElementWiseVectorMultiplyF32SSE41(aData, bData, dstData, aSize, bSize, dstSize, capacity)
	} else {
		goImpl(aData, bData, dstData, aSize, bSize, dstSize, capacity)
	}
}

func BlazeElementWiseVectorMultiplyF64Dispatch(
	aData, bData, dstData unsafe.Pointer, aSize, bSize, dstSize uintptr, capacity uint64,
	goImpl func(unsafe.Pointer, unsafe.Pointer, unsafe.Pointer, uintptr, uintptr, uintptr, uint64),
) {
	if cpu.X86.HasAVX2 {
		blazeElementWiseVectorMultiplyF64AVX2(aData, bData, dstData, aSize, bSize, dstSize, capacity)
	} else if cpu.X86.HasAVX {
		blazeElementWiseVectorMultiplyF64AVX(aData, bData, dstData, aSize, bSize, dstSize, capacity)
	} else if cpu.X86.HasSSE41 {
		blazeElementWiseVectorMultiplyF64SSE41(aData, bData, dstData, aSize, bSize, dstSize, capacity)
	} else {
		goImpl(aData, bData, dstData, aSize, bSize, dstSize, capacity)
	}
}

func BlazeScalarVectorMultiplyF32Dispatch(
	srcData, dstData unsafe.Pointer, srcSize, dstSize uintptr, scalar float32, capacity uint64,
	goImpl func(unsafe.Pointer, unsafe.Pointer, uintptr, uintptr, float32, uint64),
) {
	if cpu.X86.HasAVX2 {
		blazeScalarVectorMultiplyF32AVX2(srcData, dstData, srcSize, dstSize, scalar, capacity)
	} else if cpu.X86.HasAVX {
		blazeScalarVectorMultiplyF32AVX(srcData, dstData, srcSize, dstSize, scalar, capacity)
	} else if cpu.X86.HasSSE41 {
		blazeScalarVectorMultiplyF32SSE41(srcData, dstData, srcSize, dstSize, scalar, capacity)
	} else {
		goImpl(srcData, dstData, srcSize, dstSize, scalar, capacity)
	}
}

func BlazeScalarVectorMultiplyF64Dispatch(
	srcData, dstData unsafe.Pointer, srcSize, dstSize uintptr, scalar float64, capacity uint64,
	goImpl func(unsafe.Pointer, unsafe.Pointer, uintptr, uintptr, float64, uint64),
) {
	if cpu.X86.HasAVX2 {
		blazeScalarVectorMultiplyF64AVX2(srcData, dstData, srcSize, dstSize, scalar, capacity)
	} else if cpu.X86.HasAVX {
		blazeScalarVectorMultiplyF64AVX(srcData, dstData, srcSize, dstSize, scalar, capacity)
	} else if cpu.X86.HasSSE41 {
		blazeScalarVectorMultiplyF64SSE41(srcData, dstData, srcSize, dstSize, scalar, capacity)
	} else {
		goImpl(srcData, dstData, srcSize, dstSize, scalar, capacity)
	}
}

func BlazeReduceVectorSumF64Dispatch(
	data unsafe.Pointer, size uintptr, capacity uint64,
	goImpl func(unsafe.Pointer, uintptr, uint64) float64,
) float64 {
	if cpu.X86.HasAVX2 {
		return blazeReduceVectorSumF64AVX2(data, size, capacity)
	} else if cpu.X86.HasAVX {
		return blazeReduceVectorSumF64AVX(data, size, capacity)
	} else if cpu.X86.HasSSE41 {
		return blazeReduceVectorSumF64SSE41(data, size, capacity)
	}
	return goImpl(data, size, capacity)
}

func BlazeReduceVectorSumF32Dispatch(
	data unsafe.Pointer, size uintptr, capacity uint64,
	goImpl func(unsafe.Pointer, uintptr, uint64) float32,
) float32 {
	if cpu.X86.HasAVX2 {
		return blazeReduceVectorSumF32AVX2(data, size, capacity)
	} else if cpu.X86.HasAVX {
		return blazeReduceVectorSumF32AVX(data, size, capacity)
	} else if cpu.X86.HasSSE41 {
		return blazeReduceVectorSumF32SSE41(data, size, capacity)
	}
	return goImpl(data, size, capacity)
}

