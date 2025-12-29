package simd

import (
	"unsafe"
)

// isAligned checks if a pointer is aligned to the specified alignment boundary
// Alignment must be a power of two
//
//go:inline
func isAligned(ptr unsafe.Pointer, alignment uintptr) bool {
	return uintptr(ptr)%alignment == 0
}

// Dispatch functions select the best available SIMD implementation
// They fall back to Go implementations (passed as function parameters) when SIMD is unavailable
// CPU feature detection happens once at package init, eliminating branch overhead in hot paths

func BlazeReduceVectorSumSquaredF64Dispatch(
	data unsafe.Pointer, size uintptr, capacity uint64, isContiguous bool,
	goImpl func(unsafe.Pointer, uintptr, uint64) float64,
) float64 {
	// Handle nil pointer or empty vector
	if data == nil || capacity == 0 {
		return goImpl(data, size, capacity)
	}

	// Check if data is contiguous and can use optimized SIMD path
	if isContiguous {
		if fn := defaultDispatchTable.sumSquaredF64Contiguous; fn != nil {
			// Verify 32-byte alignment for SIMD (AVX2 requirement)
			if isAligned(data, 32) {
				return fn(data, size, capacity)
			}
			// Fall back to strided or Go implementation if misaligned
			panic("unaligned")
		}
	}

	if fn := defaultDispatchTable.sumSquaredF64Strided; fn != nil {
		return fn(data, size, capacity)
	}
	return goImpl(data, size, capacity)
}

func BlazeReduceVectorSumSquaredF32Dispatch(
	data unsafe.Pointer, size uintptr, capacity uint64, isContiguous bool,
	goImpl func(unsafe.Pointer, uintptr, uint64) float32,
) float32 {
	// Handle nil pointer or empty vector
	if data == nil || capacity == 0 {
		return goImpl(data, size, capacity)
	}

	// Check if data is contiguous and can use optimized SIMD path
	if isContiguous {
		if fn := defaultDispatchTable.sumSquaredF32Contiguous; fn != nil {
			// Verify 32-byte alignment for SIMD (AVX2 requirement)
			if isAligned(data, 32) {
				return fn(data, size, capacity)
			}
			// Fall back to strided or Go implementation if misaligned
		}
	}

	if fn := defaultDispatchTable.sumSquaredF32Strided; fn != nil {
		return fn(data, size, capacity)
	}
	return goImpl(data, size, capacity)
}

func BlazeReduceDotProductF64Dispatch(
	aData, bData unsafe.Pointer, aSize, bSize uintptr, capacity uint64,
	aIsContiguous, bIsContiguous bool,
	goImpl func(unsafe.Pointer, unsafe.Pointer, uintptr, uintptr, uint64) float64,
) float64 {
	// Handle nil pointers or empty vector
	if aData == nil || bData == nil || capacity == 0 {
		return goImpl(aData, bData, aSize, bSize, capacity)
	}

	// Check if both vectors are contiguous and can use optimized SIMD path
	// For contiguous operations, both vectors must have the same element size
	if aIsContiguous && bIsContiguous && aSize == bSize {
		if fn := defaultDispatchTable.dotProductF64Contiguous; fn != nil {
			// Verify 32-byte alignment for SIMD (AVX2 requirement) for both pointers
			if isAligned(aData, 32) && isAligned(bData, 32) {
				return fn(aData, bData, aSize, capacity)
			}
			// Fall back to strided or Go implementation if misaligned
		}
	}

	if fn := defaultDispatchTable.dotProductF64Strided; fn != nil {
		return fn(aData, bData, aSize, bSize, capacity)
	}
	return goImpl(aData, bData, aSize, bSize, capacity)
}

func BlazeReduceDotProductF32Dispatch(
	aData, bData unsafe.Pointer, aSize, bSize uintptr, capacity uint64,
	aIsContiguous, bIsContiguous bool,
	goImpl func(unsafe.Pointer, unsafe.Pointer, uintptr, uintptr, uint64) float32,
) float32 {
	// Handle nil pointers or empty vector
	if aData == nil || bData == nil || capacity == 0 {
		return goImpl(aData, bData, aSize, bSize, capacity)
	}

	// Check if both vectors are contiguous and can use optimized SIMD path
	// For contiguous operations, both vectors must have the same element size
	if aIsContiguous && bIsContiguous && aSize == bSize {
		if fn := defaultDispatchTable.dotProductF32Contiguous; fn != nil {
			// Verify 32-byte alignment for SIMD (AVX2 requirement) for both pointers
			if isAligned(aData, 32) && isAligned(bData, 32) {
				return fn(aData, bData, aSize, capacity)
			}
			// Fall back to strided or Go implementation if misaligned
		}
	}

	if fn := defaultDispatchTable.dotProductF32Strided; fn != nil {
		return fn(aData, bData, aSize, bSize, capacity)
	}
	return goImpl(aData, bData, aSize, bSize, capacity)
}

func BlazeElementWiseVectorAddF32Dispatch(
	aData, bData, dstData unsafe.Pointer, aSize, bSize, dstSize uintptr, capacity uint64,
	aIsContiguous, bIsContiguous, dstIsContiguous bool,
	goImpl func(unsafe.Pointer, unsafe.Pointer, unsafe.Pointer, uintptr, uintptr, uintptr, uint64),
) {
	// Handle nil pointers or empty vector
	if aData == nil || bData == nil || dstData == nil || capacity == 0 {
		goImpl(aData, bData, dstData, aSize, bSize, dstSize, capacity)
		return
	}

	// Check if all vectors are contiguous and can use optimized SIMD path
	// For contiguous operations, all vectors must have the same element size
	if aIsContiguous && bIsContiguous && dstIsContiguous && aSize == bSize && aSize == dstSize {
		if fn := defaultDispatchTable.addF32Contiguous; fn != nil {
			// Verify 32-byte alignment for SIMD (AVX2 requirement) for all pointers
			if isAligned(aData, 32) && isAligned(bData, 32) && isAligned(dstData, 32) {
				fn(aData, bData, dstData, aSize, capacity)
				return
			}
			// Fall back to strided or Go implementation if misaligned
		}
	}

	if fn := defaultDispatchTable.addF32Strided; fn != nil {
		fn(aData, bData, dstData, aSize, bSize, dstSize, capacity)
	} else {
		goImpl(aData, bData, dstData, aSize, bSize, dstSize, capacity)
	}
}

func BlazeElementWiseVectorAddF64Dispatch(
	aData, bData, dstData unsafe.Pointer, aSize, bSize, dstSize uintptr, capacity uint64,
	aIsContiguous, bIsContiguous, dstIsContiguous bool,
	goImpl func(unsafe.Pointer, unsafe.Pointer, unsafe.Pointer, uintptr, uintptr, uintptr, uint64),
) {
	// Handle nil pointers or empty vector
	if aData == nil || bData == nil || dstData == nil || capacity == 0 {
		goImpl(aData, bData, dstData, aSize, bSize, dstSize, capacity)
		return
	}

	// Check if all vectors are contiguous and can use optimized SIMD path
	if aIsContiguous && bIsContiguous && dstIsContiguous {
		if fn := defaultDispatchTable.addF64Contiguous; fn != nil {
			// Verify 32-byte alignment for SIMD (AVX2 requirement) for all pointers
			if isAligned(aData, 32) && isAligned(bData, 32) && isAligned(dstData, 32) {
				fn(aData, bData, dstData, aSize, capacity)
				return
			}
			// Fall back to strided or Go implementation if misaligned
		}
	}

	if fn := defaultDispatchTable.addF64Strided; fn != nil {
		fn(aData, bData, dstData, aSize, bSize, dstSize, capacity)
	} else {
		goImpl(aData, bData, dstData, aSize, bSize, dstSize, capacity)
	}
}

func BlazeElementWiseVectorMultiplyF32Dispatch(
	aData, bData, dstData unsafe.Pointer, aSize, bSize, dstSize uintptr, capacity uint64,
	aIsContiguous, bIsContiguous, dstIsContiguous bool,
	goImpl func(unsafe.Pointer, unsafe.Pointer, unsafe.Pointer, uintptr, uintptr, uintptr, uint64),
) {
	// Handle nil pointers or empty vector
	if aData == nil || bData == nil || dstData == nil || capacity == 0 {
		goImpl(aData, bData, dstData, aSize, bSize, dstSize, capacity)
		return
	}

	// Check if all vectors are contiguous and can use optimized SIMD path
	// For contiguous operations, all vectors must have the same element size
	if aIsContiguous && bIsContiguous && dstIsContiguous && aSize == bSize && aSize == dstSize {
		if fn := defaultDispatchTable.multiplyF32Contiguous; fn != nil {
			// Verify 32-byte alignment for SIMD (AVX2 requirement) for all pointers
			if isAligned(aData, 32) && isAligned(bData, 32) && isAligned(dstData, 32) {
				fn(aData, bData, dstData, aSize, capacity)
				return
			}
			// Fall back to strided or Go implementation if misaligned
		}
	}

	if fn := defaultDispatchTable.multiplyF32Strided; fn != nil {
		fn(aData, bData, dstData, aSize, bSize, dstSize, capacity)
	} else {
		goImpl(aData, bData, dstData, aSize, bSize, dstSize, capacity)
	}
}

func BlazeElementWiseVectorMultiplyF64Dispatch(
	aData, bData, dstData unsafe.Pointer, aSize, bSize, dstSize uintptr, capacity uint64,
	aIsContiguous, bIsContiguous, dstIsContiguous bool,
	goImpl func(unsafe.Pointer, unsafe.Pointer, unsafe.Pointer, uintptr, uintptr, uintptr, uint64),
) {
	// Handle nil pointers or empty vector
	if aData == nil || bData == nil || dstData == nil || capacity == 0 {
		goImpl(aData, bData, dstData, aSize, bSize, dstSize, capacity)
		return
	}

	// Check if all vectors are contiguous and can use optimized SIMD path
	// For contiguous operations, all vectors must have the same element size
	if aIsContiguous && bIsContiguous && dstIsContiguous && aSize == bSize && aSize == dstSize {
		if fn := defaultDispatchTable.multiplyF64Contiguous; fn != nil {
			// Verify 32-byte alignment for SIMD (AVX2 requirement) for all pointers
			if isAligned(aData, 32) && isAligned(bData, 32) && isAligned(dstData, 32) {
				fn(aData, bData, dstData, aSize, capacity)
				return
			}
			// Fall back to strided or Go implementation if misaligned
		}
	}

	if fn := defaultDispatchTable.multiplyF64Strided; fn != nil {
		fn(aData, bData, dstData, aSize, bSize, dstSize, capacity)
	} else {
		goImpl(aData, bData, dstData, aSize, bSize, dstSize, capacity)
	}
}

func BlazeScalarVectorMultiplyF32Dispatch(
	srcData, dstData unsafe.Pointer, srcSize, dstSize uintptr, scalar float32, capacity uint64,
	srcIsContiguous, dstIsContiguous bool,
	goImpl func(unsafe.Pointer, unsafe.Pointer, uintptr, uintptr, float32, uint64),
) {
	// Handle nil pointers or empty vector
	if srcData == nil || dstData == nil || capacity == 0 {
		goImpl(srcData, dstData, srcSize, dstSize, scalar, capacity)
		return
	}

	// Check if both vectors are contiguous and can use optimized SIMD path
	if srcIsContiguous && dstIsContiguous {
		if fn := defaultDispatchTable.scalarMultiplyF32Contiguous; fn != nil {
			// Verify 32-byte alignment for SIMD (AVX2 requirement) for both pointers
			if isAligned(srcData, 32) && isAligned(dstData, 32) {
				fn(srcData, dstData, srcSize, scalar, capacity)
				return
			}
			// Fall back to strided or Go implementation if misaligned
		}
	}

	if fn := defaultDispatchTable.scalarMultiplyF32Strided; fn != nil {
		fn(srcData, dstData, srcSize, dstSize, scalar, capacity)
	} else {
		goImpl(srcData, dstData, srcSize, dstSize, scalar, capacity)
	}
}

func BlazeScalarVectorMultiplyF64Dispatch(
	srcData, dstData unsafe.Pointer, srcSize, dstSize uintptr, scalar float64, capacity uint64,
	srcIsContiguous, dstIsContiguous bool,
	goImpl func(unsafe.Pointer, unsafe.Pointer, uintptr, uintptr, float64, uint64),
) {
	// Handle nil pointers or empty vector
	if srcData == nil || dstData == nil || capacity == 0 {
		goImpl(srcData, dstData, srcSize, dstSize, scalar, capacity)
		return
	}

	// Check if both vectors are contiguous and can use optimized SIMD path
	if srcIsContiguous && dstIsContiguous {
		if fn := defaultDispatchTable.scalarMultiplyF64Contiguous; fn != nil {
			// Verify 32-byte alignment for SIMD (AVX2 requirement) for both pointers
			if isAligned(srcData, 32) && isAligned(dstData, 32) {
				fn(srcData, dstData, srcSize, scalar, capacity)
				return
			}
			// Fall back to strided or Go implementation if misaligned
		}
	}

	if fn := defaultDispatchTable.scalarMultiplyF64Strided; fn != nil {
		fn(srcData, dstData, srcSize, dstSize, scalar, capacity)
	} else {
		goImpl(srcData, dstData, srcSize, dstSize, scalar, capacity)
	}
}

func BlazeReduceVectorSumF64Dispatch(
	data unsafe.Pointer, size uintptr, capacity uint64, isContiguous bool,
	goImpl func(unsafe.Pointer, uintptr, uint64) float64,
) float64 {
	// Handle nil pointer or empty vector
	if data == nil || capacity == 0 {
		return goImpl(data, size, capacity)
	}

	// Check if data is contiguous and can use optimized SIMD path
	if isContiguous {
		if fn := defaultDispatchTable.sumF64Contiguous; fn != nil {
			// Verify 32-byte alignment for SIMD (AVX2 requirement)
			if isAligned(data, 32) {
				return fn(data, size, capacity)
			}
			// Fall back to strided or Go implementation if misaligned
		}
	}

	if fn := defaultDispatchTable.sumF64Strided; fn != nil {
		return fn(data, size, capacity)
	}
	return goImpl(data, size, capacity)
}

func BlazeReduceVectorSumF32Dispatch(
	data unsafe.Pointer, size uintptr, capacity uint64, isContiguous bool,
	goImpl func(unsafe.Pointer, uintptr, uint64) float32,
) float32 {
	// Handle nil pointer or empty vector
	if data == nil || capacity == 0 {
		return goImpl(data, size, capacity)
	}

	// Check if data is contiguous and can use optimized SIMD path
	if isContiguous {
		if fn := defaultDispatchTable.sumF32Contiguous; fn != nil {
			// Verify 32-byte alignment for SIMD (AVX2 requirement)
			if isAligned(data, 32) {
				return fn(data, size, capacity)
			}
			// Fall back to strided or Go implementation if misaligned
		}
	}

	if fn := defaultDispatchTable.sumF32Strided; fn != nil {
		return fn(data, size, capacity)
	}
	return goImpl(data, size, capacity)
}
