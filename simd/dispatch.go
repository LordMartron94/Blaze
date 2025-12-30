package simd

import (
	"unsafe"
)

// Dispatch functions - all operations use Go implementations
// Assembly implementations removed - will be re-implemented when properly understood

func BlazeReduceVectorSumSquaredF64Dispatch(
	data unsafe.Pointer, size uintptr, capacity uint64, isContiguous bool,
	goImpl func(unsafe.Pointer, uintptr, uint64) float64,
) float64 {
	return goImpl(data, size, capacity)
}

func BlazeReduceVectorSumSquaredF32Dispatch(
	data unsafe.Pointer, size uintptr, capacity uint64, isContiguous bool,
	goImpl func(unsafe.Pointer, uintptr, uint64) float32,
) float32 {
	return goImpl(data, size, capacity)
}

func BlazeReduceDotProductF64Dispatch(
	aData, bData unsafe.Pointer, aSize, bSize uintptr, capacity uint64,
	aIsContiguous, bIsContiguous bool,
	goImpl func(unsafe.Pointer, unsafe.Pointer, uintptr, uintptr, uint64) float64,
) float64 {
	return goImpl(aData, bData, aSize, bSize, capacity)
}

func BlazeReduceDotProductF32Dispatch(
	aData, bData unsafe.Pointer, aSize, bSize uintptr, capacity uint64,
	aIsContiguous, bIsContiguous bool,
	goImpl func(unsafe.Pointer, unsafe.Pointer, uintptr, uintptr, uint64) float32,
) float32 {
	return goImpl(aData, bData, aSize, bSize, capacity)
}

func BlazeElementWiseVectorAddF32Dispatch(
	aData, bData, dstData unsafe.Pointer, aSize, bSize, dstSize uintptr, capacity uint64,
	aIsContiguous, bIsContiguous, dstIsContiguous bool,
	goImpl func(unsafe.Pointer, unsafe.Pointer, unsafe.Pointer, uintptr, uintptr, uintptr, uint64),
) {
	goImpl(aData, bData, dstData, aSize, bSize, dstSize, capacity)
}

func BlazeElementWiseVectorAddF64Dispatch(
	aData, bData, dstData unsafe.Pointer, aSize, bSize, dstSize uintptr, capacity uint64,
	aIsContiguous, bIsContiguous, dstIsContiguous bool,
	goImpl func(unsafe.Pointer, unsafe.Pointer, unsafe.Pointer, uintptr, uintptr, uintptr, uint64),
) {
	goImpl(aData, bData, dstData, aSize, bSize, dstSize, capacity)
}

func BlazeElementWiseVectorMultiplyF32Dispatch(
	aData, bData, dstData unsafe.Pointer, aSize, bSize, dstSize uintptr, capacity uint64,
	aIsContiguous, bIsContiguous, dstIsContiguous bool,
	goImpl func(unsafe.Pointer, unsafe.Pointer, unsafe.Pointer, uintptr, uintptr, uintptr, uint64),
) {
	goImpl(aData, bData, dstData, aSize, bSize, dstSize, capacity)
}

func BlazeElementWiseVectorMultiplyF64Dispatch(
	aData, bData, dstData unsafe.Pointer, aSize, bSize, dstSize uintptr, capacity uint64,
	aIsContiguous, bIsContiguous, dstIsContiguous bool,
	goImpl func(unsafe.Pointer, unsafe.Pointer, unsafe.Pointer, uintptr, uintptr, uintptr, uint64),
) {
	goImpl(aData, bData, dstData, aSize, bSize, dstSize, capacity)
}

func BlazeScalarVectorMultiplyF32Dispatch(
	srcData, dstData unsafe.Pointer, srcSize, dstSize uintptr, scalar float32, capacity uint64,
	srcIsContiguous, dstIsContiguous bool,
	goImpl func(unsafe.Pointer, unsafe.Pointer, uintptr, uintptr, float32, uint64),
) {
	goImpl(srcData, dstData, srcSize, dstSize, scalar, capacity)
}

func BlazeScalarVectorMultiplyF64Dispatch(
	srcData, dstData unsafe.Pointer, srcSize, dstSize uintptr, scalar float64, capacity uint64,
	srcIsContiguous, dstIsContiguous bool,
	goImpl func(unsafe.Pointer, unsafe.Pointer, uintptr, uintptr, float64, uint64),
) {
	goImpl(srcData, dstData, srcSize, dstSize, scalar, capacity)
}

func BlazeReduceVectorSumF64Dispatch(
	data unsafe.Pointer, size uintptr, capacity uint64, isContiguous bool,
	goImpl func(unsafe.Pointer, uintptr, uint64) float64,
) float64 {
	return goImpl(data, size, capacity)
}

func BlazeReduceVectorSumF32Dispatch(
	data unsafe.Pointer, size uintptr, capacity uint64, isContiguous bool,
	goImpl func(unsafe.Pointer, uintptr, uint64) float32,
) float32 {
	return goImpl(data, size, capacity)
}
