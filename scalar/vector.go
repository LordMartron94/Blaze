package scalar

import (
	"fmt"
	"unsafe"

	"foundation"
	"memcore"
	"memstruct"
)

/*
BlazeScalarVectorSetAllSequence fills a vector with an arithmetic sequence.

Sets each element to: value[i] = initial + (i * step) where i is the element index.

Use cases:
- Creating test data (sequences, ranges)
- Initializing vectors with patterns
- Generating indices or coordinates
- Data preprocessing

Time complexity: O(n) - single pass through vector
Space complexity: O(1) - only accumulator variables used

Prerequisites:
- Vector must be valid and initialized

Edge cases:
- Panics if any computed value exceeds the numeric type's maximum value
- Works with any numeric type (int, float32, float64, etc.)
- No type conversions (uses same type as input)
- Overflow checking prevents silent data corruption
*/
func BlazeScalarVectorSetAllSequence[T foundation.Numeric](vector memcore.MarkRaw, initial, step T) {
	data := memstruct.VectorDataPtrGet[T](vector)
	size := uintptr(memcore.SizeOf[T]())
	capacity := memstruct.VectorCapacityGet[T](vector)
	maxT := float64(foundation.MaxValue[T]())

	var i uint64
	idx := uint64(0)

	// Manually unrolled loop for stride 8
	for ; i+7 < capacity; i += 8 {
		v0 := float64(initial) + (float64(idx+0) * float64(step))
		v1 := float64(initial) + (float64(idx+1) * float64(step))
		v2 := float64(initial) + (float64(idx+2) * float64(step))
		v3 := float64(initial) + (float64(idx+3) * float64(step))
		v4 := float64(initial) + (float64(idx+4) * float64(step))
		v5 := float64(initial) + (float64(idx+5) * float64(step))
		v6 := float64(initial) + (float64(idx+6) * float64(step))
		v7 := float64(initial) + (float64(idx+7) * float64(step))

		if v0 > maxT {
			panic(fmt.Errorf("overflow at idx %d: requested=%f, max=%f", idx+0, v0, maxT))
		}
		if v1 > maxT {
			panic(fmt.Errorf("overflow at idx %d: requested=%f, max=%f", idx+1, v1, maxT))
		}
		if v2 > maxT {
			panic(fmt.Errorf("overflow at idx %d: requested=%f, max=%f", idx+2, v2, maxT))
		}
		if v3 > maxT {
			panic(fmt.Errorf("overflow at idx %d: requested=%f, max=%f", idx+3, v3, maxT))
		}
		if v4 > maxT {
			panic(fmt.Errorf("overflow at idx %d: requested=%f, max=%f", idx+4, v4, maxT))
		}
		if v5 > maxT {
			panic(fmt.Errorf("overflow at idx %d: requested=%f, max=%f", idx+5, v5, maxT))
		}
		if v6 > maxT {
			panic(fmt.Errorf("overflow at idx %d: requested=%f, max=%f", idx+6, v6, maxT))
		}
		if v7 > maxT {
			panic(fmt.Errorf("overflow at idx %d: requested=%f, max=%f", idx+7, v7, maxT))
		}

		*(*T)(unsafe.Add(data, uintptr(i+0)*size)) = T(v0)
		*(*T)(unsafe.Add(data, uintptr(i+1)*size)) = T(v1)
		*(*T)(unsafe.Add(data, uintptr(i+2)*size)) = T(v2)
		*(*T)(unsafe.Add(data, uintptr(i+3)*size)) = T(v3)
		*(*T)(unsafe.Add(data, uintptr(i+4)*size)) = T(v4)
		*(*T)(unsafe.Add(data, uintptr(i+5)*size)) = T(v5)
		*(*T)(unsafe.Add(data, uintptr(i+6)*size)) = T(v6)
		*(*T)(unsafe.Add(data, uintptr(i+7)*size)) = T(v7)

		idx += 8
	}

	// Handle remaining elements
	for ; i < capacity; i++ {
		v := float64(initial) + (float64(idx) * float64(step))
		if v > maxT {
			panic(fmt.Errorf("overflow at idx %d: requested=%f, max=%f", idx, v, maxT))
		}
		*(*T)(unsafe.Add(data, uintptr(i)*size)) = T(v)
		idx++
	}
}

// ────────────────────────────────────────────────────────────────
//   FLOAT32 PRECISION OPERATIONS
// ────────────────────────────────────────────────────────────────

/*
BlazeScalarVectorMultiplyF32 multiplies each vector element by a scalar value in float32 precision.

Computes: result[i] = vector[i] * scalar for all elements i.

Use cases:
- Scaling data (normalization, unit conversion)
- Signal processing (amplification, attenuation)
- Machine learning (feature scaling)
- Numerical simulations (scaling factors)

Time complexity: O(n) - single pass through vector
Space complexity: O(1) - only accumulator variables used

Prerequisites:
- Input vector must be valid and initialized
- Output vector must have the same length as input
- Both vectors must be valid and initialized

Edge cases:
- Works with any numeric type (int, float32, float64, etc.)
- Type conversions occur during computation (input types → float32)
- No overflow checking (relies on Go's numeric behavior)
*/
func BlazeScalarVectorMultiplyF32[T foundation.Numeric](
	currentVectorAddr, newVectorAddr memcore.MarkRaw,
	scalar float32,
) {
	srcData := memstruct.VectorDataPtrGet[T](currentVectorAddr)
	dstData := memstruct.VectorDataPtrGet[float32](newVectorAddr)
	srcSize := uintptr(memcore.SizeOf[T]())
	dstSize := uintptr(memcore.SizeOf[float32]())
	capacity := memstruct.VectorCapacityGet[T](currentVectorAddr)

	var i uint64

	// Manually unrolled loop for stride 8
	for ; i+7 < capacity; i += 8 {
		v0 := float32(*(*T)(unsafe.Add(srcData, uintptr(i+0)*srcSize))) * scalar
		v1 := float32(*(*T)(unsafe.Add(srcData, uintptr(i+1)*srcSize))) * scalar
		v2 := float32(*(*T)(unsafe.Add(srcData, uintptr(i+2)*srcSize))) * scalar
		v3 := float32(*(*T)(unsafe.Add(srcData, uintptr(i+3)*srcSize))) * scalar
		v4 := float32(*(*T)(unsafe.Add(srcData, uintptr(i+4)*srcSize))) * scalar
		v5 := float32(*(*T)(unsafe.Add(srcData, uintptr(i+5)*srcSize))) * scalar
		v6 := float32(*(*T)(unsafe.Add(srcData, uintptr(i+6)*srcSize))) * scalar
		v7 := float32(*(*T)(unsafe.Add(srcData, uintptr(i+7)*srcSize))) * scalar

		*(*float32)(unsafe.Add(dstData, uintptr(i+0)*dstSize)) = v0
		*(*float32)(unsafe.Add(dstData, uintptr(i+1)*dstSize)) = v1
		*(*float32)(unsafe.Add(dstData, uintptr(i+2)*dstSize)) = v2
		*(*float32)(unsafe.Add(dstData, uintptr(i+3)*dstSize)) = v3
		*(*float32)(unsafe.Add(dstData, uintptr(i+4)*dstSize)) = v4
		*(*float32)(unsafe.Add(dstData, uintptr(i+5)*dstSize)) = v5
		*(*float32)(unsafe.Add(dstData, uintptr(i+6)*dstSize)) = v6
		*(*float32)(unsafe.Add(dstData, uintptr(i+7)*dstSize)) = v7
	}

	// Handle remaining elements
	for ; i < capacity; i++ {
		v := float32(*(*T)(unsafe.Add(srcData, uintptr(i)*srcSize))) * scalar
		*(*float32)(unsafe.Add(dstData, uintptr(i)*dstSize)) = v
	}
}

/*
BlazeScalarVectorDivideF32 divides each vector element by a scalar value in float32 precision.

Computes: result[i] = vector[i] / scalar for all elements i.

Use cases:
- Normalization (dividing by sum, mean, etc.)
- Unit conversion (scaling factors)
- Signal processing (attenuation)
- Machine learning (feature normalization)

Time complexity: O(n) - single pass through vector
Space complexity: O(1) - only accumulator variables used

Prerequisites:
- Input vector must be valid and initialized
- Output vector must have the same length as input
- Both vectors must be valid and initialized

Edge cases:
- Works with any numeric type (int, float32, float64, etc.)
- Type conversions occur during computation (input types → float32)
- Division by zero results in ±Inf or NaN (Go's standard behavior)
- No overflow checking (relies on Go's numeric behavior)
*/
func BlazeScalarVectorDivideF32[T foundation.Numeric](
	currentVectorAddr, newVectorAddr memcore.MarkRaw,
	scalar float32,
) {
	srcData := memstruct.VectorDataPtrGet[T](currentVectorAddr)
	dstData := memstruct.VectorDataPtrGet[float32](newVectorAddr)
	srcSize := uintptr(memcore.SizeOf[T]())
	dstSize := uintptr(memcore.SizeOf[float32]())
	capacity := memstruct.VectorCapacityGet[T](currentVectorAddr)

	var i uint64

	// Manually unrolled loop for stride 8
	for ; i+7 < capacity; i += 8 {
		v0 := float32(*(*T)(unsafe.Add(srcData, uintptr(i+0)*srcSize))) / scalar
		v1 := float32(*(*T)(unsafe.Add(srcData, uintptr(i+1)*srcSize))) / scalar
		v2 := float32(*(*T)(unsafe.Add(srcData, uintptr(i+2)*srcSize))) / scalar
		v3 := float32(*(*T)(unsafe.Add(srcData, uintptr(i+3)*srcSize))) / scalar
		v4 := float32(*(*T)(unsafe.Add(srcData, uintptr(i+4)*srcSize))) / scalar
		v5 := float32(*(*T)(unsafe.Add(srcData, uintptr(i+5)*srcSize))) / scalar
		v6 := float32(*(*T)(unsafe.Add(srcData, uintptr(i+6)*srcSize))) / scalar
		v7 := float32(*(*T)(unsafe.Add(srcData, uintptr(i+7)*srcSize))) / scalar

		*(*float32)(unsafe.Add(dstData, uintptr(i+0)*dstSize)) = v0
		*(*float32)(unsafe.Add(dstData, uintptr(i+1)*dstSize)) = v1
		*(*float32)(unsafe.Add(dstData, uintptr(i+2)*dstSize)) = v2
		*(*float32)(unsafe.Add(dstData, uintptr(i+3)*dstSize)) = v3
		*(*float32)(unsafe.Add(dstData, uintptr(i+4)*dstSize)) = v4
		*(*float32)(unsafe.Add(dstData, uintptr(i+5)*dstSize)) = v5
		*(*float32)(unsafe.Add(dstData, uintptr(i+6)*dstSize)) = v6
		*(*float32)(unsafe.Add(dstData, uintptr(i+7)*dstSize)) = v7
	}

	// Handle remaining elements
	for ; i < capacity; i++ {
		v := float32(*(*T)(unsafe.Add(srcData, uintptr(i)*srcSize))) / scalar
		*(*float32)(unsafe.Add(dstData, uintptr(i)*dstSize)) = v
	}
}

/*
BlazeScalarVectorAddF32 adds a scalar value to each vector element in float32 precision.

Computes: result[i] = vector[i] + scalar for all elements i.

Use cases:
- Shifting data (offset adjustments)
- Signal processing (DC bias)
- Data preprocessing (centering, offsetting)
- Numerical simulations (translations)

Time complexity: O(n) - single pass through vector
Space complexity: O(1) - only accumulator variables used

Prerequisites:
- Input vector must be valid and initialized
- Output vector must have the same length as input
- Both vectors must be valid and initialized

Edge cases:
- Works with any numeric type (int, float32, float64, etc.)
- Type conversions occur during computation (input types → float32)
- No overflow checking (relies on Go's numeric behavior)
*/
func BlazeScalarVectorAddF32[T foundation.Numeric](
	currentVectorAddr, newVectorAddr memcore.MarkRaw,
	scalar float32,
) {
	srcData := memstruct.VectorDataPtrGet[T](currentVectorAddr)
	dstData := memstruct.VectorDataPtrGet[float32](newVectorAddr)
	srcSize := uintptr(memcore.SizeOf[T]())
	dstSize := uintptr(memcore.SizeOf[float32]())
	capacity := memstruct.VectorCapacityGet[T](currentVectorAddr)

	var i uint64

	// Manually unrolled loop for stride 8
	for ; i+7 < capacity; i += 8 {
		v0 := float32(*(*T)(unsafe.Add(srcData, uintptr(i+0)*srcSize))) + scalar
		v1 := float32(*(*T)(unsafe.Add(srcData, uintptr(i+1)*srcSize))) + scalar
		v2 := float32(*(*T)(unsafe.Add(srcData, uintptr(i+2)*srcSize))) + scalar
		v3 := float32(*(*T)(unsafe.Add(srcData, uintptr(i+3)*srcSize))) + scalar
		v4 := float32(*(*T)(unsafe.Add(srcData, uintptr(i+4)*srcSize))) + scalar
		v5 := float32(*(*T)(unsafe.Add(srcData, uintptr(i+5)*srcSize))) + scalar
		v6 := float32(*(*T)(unsafe.Add(srcData, uintptr(i+6)*srcSize))) + scalar
		v7 := float32(*(*T)(unsafe.Add(srcData, uintptr(i+7)*srcSize))) + scalar

		*(*float32)(unsafe.Add(dstData, uintptr(i+0)*dstSize)) = v0
		*(*float32)(unsafe.Add(dstData, uintptr(i+1)*dstSize)) = v1
		*(*float32)(unsafe.Add(dstData, uintptr(i+2)*dstSize)) = v2
		*(*float32)(unsafe.Add(dstData, uintptr(i+3)*dstSize)) = v3
		*(*float32)(unsafe.Add(dstData, uintptr(i+4)*dstSize)) = v4
		*(*float32)(unsafe.Add(dstData, uintptr(i+5)*dstSize)) = v5
		*(*float32)(unsafe.Add(dstData, uintptr(i+6)*dstSize)) = v6
		*(*float32)(unsafe.Add(dstData, uintptr(i+7)*dstSize)) = v7
	}

	// Handle remaining elements
	for ; i < capacity; i++ {
		v := float32(*(*T)(unsafe.Add(srcData, uintptr(i)*srcSize))) + scalar
		*(*float32)(unsafe.Add(dstData, uintptr(i)*dstSize)) = v
	}
}

/*
BlazeScalarVectorSubtractF32 subtracts a scalar value from each vector element in float32 precision.

Computes: result[i] = vector[i] - scalar for all elements i.

Use cases:
- Shifting data (offset adjustments)
- Signal processing (DC removal)
- Data preprocessing (centering, mean subtraction)
- Numerical simulations (translations)

Time complexity: O(n) - single pass through vector
Space complexity: O(1) - only accumulator variables used

Prerequisites:
- Input vector must be valid and initialized
- Output vector must have the same length as input
- Both vectors must be valid and initialized

Edge cases:
- Works with any numeric type (int, float32, float64, etc.)
- Type conversions occur during computation (input types → float32)
- No overflow checking (relies on Go's numeric behavior)
*/
func BlazeScalarVectorSubtractF32[T foundation.Numeric](
	currentVectorAddr, newVectorAddr memcore.MarkRaw,
	scalar float32,
) {
	srcData := memstruct.VectorDataPtrGet[T](currentVectorAddr)
	dstData := memstruct.VectorDataPtrGet[float32](newVectorAddr)
	srcSize := uintptr(memcore.SizeOf[T]())
	dstSize := uintptr(memcore.SizeOf[float32]())
	capacity := memstruct.VectorCapacityGet[T](currentVectorAddr)

	var i uint64

	// Manually unrolled loop for stride 8
	for ; i+7 < capacity; i += 8 {
		v0 := float32(*(*T)(unsafe.Add(srcData, uintptr(i+0)*srcSize))) - scalar
		v1 := float32(*(*T)(unsafe.Add(srcData, uintptr(i+1)*srcSize))) - scalar
		v2 := float32(*(*T)(unsafe.Add(srcData, uintptr(i+2)*srcSize))) - scalar
		v3 := float32(*(*T)(unsafe.Add(srcData, uintptr(i+3)*srcSize))) - scalar
		v4 := float32(*(*T)(unsafe.Add(srcData, uintptr(i+4)*srcSize))) - scalar
		v5 := float32(*(*T)(unsafe.Add(srcData, uintptr(i+5)*srcSize))) - scalar
		v6 := float32(*(*T)(unsafe.Add(srcData, uintptr(i+6)*srcSize))) - scalar
		v7 := float32(*(*T)(unsafe.Add(srcData, uintptr(i+7)*srcSize))) - scalar

		*(*float32)(unsafe.Add(dstData, uintptr(i+0)*dstSize)) = v0
		*(*float32)(unsafe.Add(dstData, uintptr(i+1)*dstSize)) = v1
		*(*float32)(unsafe.Add(dstData, uintptr(i+2)*dstSize)) = v2
		*(*float32)(unsafe.Add(dstData, uintptr(i+3)*dstSize)) = v3
		*(*float32)(unsafe.Add(dstData, uintptr(i+4)*dstSize)) = v4
		*(*float32)(unsafe.Add(dstData, uintptr(i+5)*dstSize)) = v5
		*(*float32)(unsafe.Add(dstData, uintptr(i+6)*dstSize)) = v6
		*(*float32)(unsafe.Add(dstData, uintptr(i+7)*dstSize)) = v7
	}

	// Handle remaining elements
	for ; i < capacity; i++ {
		v := float32(*(*T)(unsafe.Add(srcData, uintptr(i)*srcSize))) - scalar
		*(*float32)(unsafe.Add(dstData, uintptr(i)*dstSize)) = v
	}
}

// ────────────────────────────────────────────────────────────────
//   FLOAT64 PRECISION OPERATIONS
// ────────────────────────────────────────────────────────────────

/*
BlazeScalarVectorMultiplyF64 multiplies each vector element by a scalar value in float64 precision.

Computes: result[i] = vector[i] * scalar for all elements i.

Use cases:
- Scaling data (normalization, unit conversion)
- Signal processing (amplification, attenuation)
- Machine learning (feature scaling)
- Numerical simulations (scaling factors)

Time complexity: O(n) - single pass through vector
Space complexity: O(1) - only accumulator variables used

Prerequisites:
- Input vector must be valid and initialized
- Output vector must have the same length as input
- Both vectors must be valid and initialized

Edge cases:
- Works with any numeric type (int, float32, float64, etc.)
- Type conversions occur during computation (input types → float64)
- Better precision than F32 variant
- No overflow checking (relies on Go's numeric behavior)
*/
func BlazeScalarVectorMultiplyF64[T foundation.Numeric](
	currentVectorAddr, newVectorAddr memcore.MarkRaw,
	scalar float64,
) {
	srcData := memstruct.VectorDataPtrGet[T](currentVectorAddr)
	dstData := memstruct.VectorDataPtrGet[float64](newVectorAddr)
	srcSize := uintptr(memcore.SizeOf[T]())
	dstSize := uintptr(memcore.SizeOf[float64]())
	capacity := memstruct.VectorCapacityGet[T](currentVectorAddr)

	var i uint64

	// Manually unrolled loop for stride 8
	for ; i+7 < capacity; i += 8 {
		v0 := float64(*(*T)(unsafe.Add(srcData, uintptr(i+0)*srcSize))) * scalar
		v1 := float64(*(*T)(unsafe.Add(srcData, uintptr(i+1)*srcSize))) * scalar
		v2 := float64(*(*T)(unsafe.Add(srcData, uintptr(i+2)*srcSize))) * scalar
		v3 := float64(*(*T)(unsafe.Add(srcData, uintptr(i+3)*srcSize))) * scalar
		v4 := float64(*(*T)(unsafe.Add(srcData, uintptr(i+4)*srcSize))) * scalar
		v5 := float64(*(*T)(unsafe.Add(srcData, uintptr(i+5)*srcSize))) * scalar
		v6 := float64(*(*T)(unsafe.Add(srcData, uintptr(i+6)*srcSize))) * scalar
		v7 := float64(*(*T)(unsafe.Add(srcData, uintptr(i+7)*srcSize))) * scalar

		*(*float64)(unsafe.Add(dstData, uintptr(i+0)*dstSize)) = v0
		*(*float64)(unsafe.Add(dstData, uintptr(i+1)*dstSize)) = v1
		*(*float64)(unsafe.Add(dstData, uintptr(i+2)*dstSize)) = v2
		*(*float64)(unsafe.Add(dstData, uintptr(i+3)*dstSize)) = v3
		*(*float64)(unsafe.Add(dstData, uintptr(i+4)*dstSize)) = v4
		*(*float64)(unsafe.Add(dstData, uintptr(i+5)*dstSize)) = v5
		*(*float64)(unsafe.Add(dstData, uintptr(i+6)*dstSize)) = v6
		*(*float64)(unsafe.Add(dstData, uintptr(i+7)*dstSize)) = v7
	}

	// Handle remaining elements
	for ; i < capacity; i++ {
		v := float64(*(*T)(unsafe.Add(srcData, uintptr(i)*srcSize))) * scalar
		*(*float64)(unsafe.Add(dstData, uintptr(i)*dstSize)) = v
	}
}

/*
BlazeScalarVectorDivideF64 divides each vector element by a scalar value in float64 precision.

Computes: result[i] = vector[i] / scalar for all elements i.

Use cases:
- Normalization (dividing by sum, mean, etc.)
- Unit conversion (scaling factors)
- Signal processing (attenuation)
- Machine learning (feature normalization)

Time complexity: O(n) - single pass through vector
Space complexity: O(1) - only accumulator variables used

Prerequisites:
- Input vector must be valid and initialized
- Output vector must have the same length as input
- Both vectors must be valid and initialized

Edge cases:
- Works with any numeric type (int, float32, float64, etc.)
- Type conversions occur during computation (input types → float64)
- Division by zero results in ±Inf or NaN (Go's standard behavior)
- Better precision than F32 variant
- No overflow checking (relies on Go's numeric behavior)
*/
func BlazeScalarVectorDivideF64[T foundation.Numeric](
	currentVectorAddr, newVectorAddr memcore.MarkRaw,
	scalar float64,
) {
	srcData := memstruct.VectorDataPtrGet[T](currentVectorAddr)
	dstData := memstruct.VectorDataPtrGet[float64](newVectorAddr)
	srcSize := uintptr(memcore.SizeOf[T]())
	dstSize := uintptr(memcore.SizeOf[float64]())
	capacity := memstruct.VectorCapacityGet[T](currentVectorAddr)

	var i uint64

	// Manually unrolled loop for stride 8
	for ; i+7 < capacity; i += 8 {
		v0 := float64(*(*T)(unsafe.Add(srcData, uintptr(i+0)*srcSize))) / scalar
		v1 := float64(*(*T)(unsafe.Add(srcData, uintptr(i+1)*srcSize))) / scalar
		v2 := float64(*(*T)(unsafe.Add(srcData, uintptr(i+2)*srcSize))) / scalar
		v3 := float64(*(*T)(unsafe.Add(srcData, uintptr(i+3)*srcSize))) / scalar
		v4 := float64(*(*T)(unsafe.Add(srcData, uintptr(i+4)*srcSize))) / scalar
		v5 := float64(*(*T)(unsafe.Add(srcData, uintptr(i+5)*srcSize))) / scalar
		v6 := float64(*(*T)(unsafe.Add(srcData, uintptr(i+6)*srcSize))) / scalar
		v7 := float64(*(*T)(unsafe.Add(srcData, uintptr(i+7)*srcSize))) / scalar

		*(*float64)(unsafe.Add(dstData, uintptr(i+0)*dstSize)) = v0
		*(*float64)(unsafe.Add(dstData, uintptr(i+1)*dstSize)) = v1
		*(*float64)(unsafe.Add(dstData, uintptr(i+2)*dstSize)) = v2
		*(*float64)(unsafe.Add(dstData, uintptr(i+3)*dstSize)) = v3
		*(*float64)(unsafe.Add(dstData, uintptr(i+4)*dstSize)) = v4
		*(*float64)(unsafe.Add(dstData, uintptr(i+5)*dstSize)) = v5
		*(*float64)(unsafe.Add(dstData, uintptr(i+6)*dstSize)) = v6
		*(*float64)(unsafe.Add(dstData, uintptr(i+7)*dstSize)) = v7
	}

	// Handle remaining elements
	for ; i < capacity; i++ {
		v := float64(*(*T)(unsafe.Add(srcData, uintptr(i)*srcSize))) / scalar
		*(*float64)(unsafe.Add(dstData, uintptr(i)*dstSize)) = v
	}
}

/*
BlazeScalarVectorAddF64 adds a scalar value to each vector element in float64 precision.

Computes: result[i] = vector[i] + scalar for all elements i.

Use cases:
- Shifting data (offset adjustments)
- Signal processing (DC bias)
- Data preprocessing (centering, offsetting)
- Numerical simulations (translations)

Time complexity: O(n) - single pass through vector
Space complexity: O(1) - only accumulator variables used

Prerequisites:
- Input vector must be valid and initialized
- Output vector must have the same length as input
- Both vectors must be valid and initialized

Edge cases:
- Works with any numeric type (int, float32, float64, etc.)
- Type conversions occur during computation (input types → float64)
- Better precision than F32 variant
- No overflow checking (relies on Go's numeric behavior)
*/
func BlazeScalarVectorAddF64[T foundation.Numeric](
	currentVectorAddr, newVectorAddr memcore.MarkRaw,
	scalar float64,
) {
	srcData := memstruct.VectorDataPtrGet[T](currentVectorAddr)
	dstData := memstruct.VectorDataPtrGet[float64](newVectorAddr)
	srcSize := uintptr(memcore.SizeOf[T]())
	dstSize := uintptr(memcore.SizeOf[float64]())
	capacity := memstruct.VectorCapacityGet[T](currentVectorAddr)

	var i uint64

	// Manually unrolled loop for stride 8
	for ; i+7 < capacity; i += 8 {
		v0 := float64(*(*T)(unsafe.Add(srcData, uintptr(i+0)*srcSize))) + scalar
		v1 := float64(*(*T)(unsafe.Add(srcData, uintptr(i+1)*srcSize))) + scalar
		v2 := float64(*(*T)(unsafe.Add(srcData, uintptr(i+2)*srcSize))) + scalar
		v3 := float64(*(*T)(unsafe.Add(srcData, uintptr(i+3)*srcSize))) + scalar
		v4 := float64(*(*T)(unsafe.Add(srcData, uintptr(i+4)*srcSize))) + scalar
		v5 := float64(*(*T)(unsafe.Add(srcData, uintptr(i+5)*srcSize))) + scalar
		v6 := float64(*(*T)(unsafe.Add(srcData, uintptr(i+6)*srcSize))) + scalar
		v7 := float64(*(*T)(unsafe.Add(srcData, uintptr(i+7)*srcSize))) + scalar

		*(*float64)(unsafe.Add(dstData, uintptr(i+0)*dstSize)) = v0
		*(*float64)(unsafe.Add(dstData, uintptr(i+1)*dstSize)) = v1
		*(*float64)(unsafe.Add(dstData, uintptr(i+2)*dstSize)) = v2
		*(*float64)(unsafe.Add(dstData, uintptr(i+3)*dstSize)) = v3
		*(*float64)(unsafe.Add(dstData, uintptr(i+4)*dstSize)) = v4
		*(*float64)(unsafe.Add(dstData, uintptr(i+5)*dstSize)) = v5
		*(*float64)(unsafe.Add(dstData, uintptr(i+6)*dstSize)) = v6
		*(*float64)(unsafe.Add(dstData, uintptr(i+7)*dstSize)) = v7
	}

	// Handle remaining elements
	for ; i < capacity; i++ {
		v := float64(*(*T)(unsafe.Add(srcData, uintptr(i)*srcSize))) + scalar
		*(*float64)(unsafe.Add(dstData, uintptr(i)*dstSize)) = v
	}
}

/*
BlazeScalarVectorSubtractF64 subtracts a scalar value from each vector element in float64 precision.

Computes: result[i] = vector[i] - scalar for all elements i.

Use cases:
- Shifting data (offset adjustments)
- Signal processing (DC removal)
- Data preprocessing (centering, mean subtraction)
- Numerical simulations (translations)

Time complexity: O(n) - single pass through vector
Space complexity: O(1) - only accumulator variables used

Prerequisites:
- Input vector must be valid and initialized
- Output vector must have the same length as input
- Both vectors must be valid and initialized

Edge cases:
- Works with any numeric type (int, float32, float64, etc.)
- Type conversions occur during computation (input types → float64)
- Better precision than F32 variant
- No overflow checking (relies on Go's numeric behavior)
*/
func BlazeScalarVectorSubtractF64[T foundation.Numeric](
	currentVectorAddr, newVectorAddr memcore.MarkRaw,
	scalar float64,
) {
	srcData := memstruct.VectorDataPtrGet[T](currentVectorAddr)
	dstData := memstruct.VectorDataPtrGet[float64](newVectorAddr)
	srcSize := uintptr(memcore.SizeOf[T]())
	dstSize := uintptr(memcore.SizeOf[float64]())
	capacity := memstruct.VectorCapacityGet[T](currentVectorAddr)

	var i uint64

	// Manually unrolled loop for stride 8
	for ; i+7 < capacity; i += 8 {
		v0 := float64(*(*T)(unsafe.Add(srcData, uintptr(i+0)*srcSize))) - scalar
		v1 := float64(*(*T)(unsafe.Add(srcData, uintptr(i+1)*srcSize))) - scalar
		v2 := float64(*(*T)(unsafe.Add(srcData, uintptr(i+2)*srcSize))) - scalar
		v3 := float64(*(*T)(unsafe.Add(srcData, uintptr(i+3)*srcSize))) - scalar
		v4 := float64(*(*T)(unsafe.Add(srcData, uintptr(i+4)*srcSize))) - scalar
		v5 := float64(*(*T)(unsafe.Add(srcData, uintptr(i+5)*srcSize))) - scalar
		v6 := float64(*(*T)(unsafe.Add(srcData, uintptr(i+6)*srcSize))) - scalar
		v7 := float64(*(*T)(unsafe.Add(srcData, uintptr(i+7)*srcSize))) - scalar

		*(*float64)(unsafe.Add(dstData, uintptr(i+0)*dstSize)) = v0
		*(*float64)(unsafe.Add(dstData, uintptr(i+1)*dstSize)) = v1
		*(*float64)(unsafe.Add(dstData, uintptr(i+2)*dstSize)) = v2
		*(*float64)(unsafe.Add(dstData, uintptr(i+3)*dstSize)) = v3
		*(*float64)(unsafe.Add(dstData, uintptr(i+4)*dstSize)) = v4
		*(*float64)(unsafe.Add(dstData, uintptr(i+5)*dstSize)) = v5
		*(*float64)(unsafe.Add(dstData, uintptr(i+6)*dstSize)) = v6
		*(*float64)(unsafe.Add(dstData, uintptr(i+7)*dstSize)) = v7
	}

	// Handle remaining elements
	for ; i < capacity; i++ {
		v := float64(*(*T)(unsafe.Add(srcData, uintptr(i)*srcSize))) - scalar
		*(*float64)(unsafe.Add(dstData, uintptr(i)*dstSize)) = v
	}
}

// ────────────────────────────────────────────────────────────────
//   MISC
// ────────────────────────────────────────────────────────────────

/*
BlazeScalarVectorClamp clamps each vector element to lie within [min, max] range.

For each element: if value < min, set to min; if value > max, set to max; otherwise keep value.
This operation mutates the existing vector in place.

Use cases:
- Data validation (enforcing bounds)
- Signal processing (limiting amplitude)
- Image processing (clipping pixel values)
- Numerical stability (preventing overflow)

Time complexity: O(n) - single pass through vector
Space complexity: O(1) - only accumulator variables used

Prerequisites:
- Vector must be valid and initialized
- min must be <= max (undefined behavior if min > max)

Edge cases:
- Works with any numeric type (int, float32, float64, etc.)
- No type conversions (uses same type as input)
- In-place operation (modifies input vector)
- If min > max, elements may be set inconsistently
*/
func BlazeScalarVectorClamp[T foundation.Numeric](
	vector memcore.MarkRaw,
	min, max T,
) {
	data := memstruct.VectorDataPtrGet[T](vector)
	size := uintptr(memcore.SizeOf[T]())
	capacity := memstruct.VectorCapacityGet[T](vector)

	var i uint64

	// Manually unrolled loop for stride 8
	for ; i+7 < capacity; i += 8 {
		v0 := *(*T)(unsafe.Add(data, uintptr(i+0)*size))
		v1 := *(*T)(unsafe.Add(data, uintptr(i+1)*size))
		v2 := *(*T)(unsafe.Add(data, uintptr(i+2)*size))
		v3 := *(*T)(unsafe.Add(data, uintptr(i+3)*size))
		v4 := *(*T)(unsafe.Add(data, uintptr(i+4)*size))
		v5 := *(*T)(unsafe.Add(data, uintptr(i+5)*size))
		v6 := *(*T)(unsafe.Add(data, uintptr(i+6)*size))
		v7 := *(*T)(unsafe.Add(data, uintptr(i+7)*size))

		if v0 < min {
			v0 = min
		} else if v0 > max {
			v0 = max
		}
		if v1 < min {
			v1 = min
		} else if v1 > max {
			v1 = max
		}
		if v2 < min {
			v2 = min
		} else if v2 > max {
			v2 = max
		}
		if v3 < min {
			v3 = min
		} else if v3 > max {
			v3 = max
		}
		if v4 < min {
			v4 = min
		} else if v4 > max {
			v4 = max
		}
		if v5 < min {
			v5 = min
		} else if v5 > max {
			v5 = max
		}
		if v6 < min {
			v6 = min
		} else if v6 > max {
			v6 = max
		}
		if v7 < min {
			v7 = min
		} else if v7 > max {
			v7 = max
		}

		*(*T)(unsafe.Add(data, uintptr(i+0)*size)) = v0
		*(*T)(unsafe.Add(data, uintptr(i+1)*size)) = v1
		*(*T)(unsafe.Add(data, uintptr(i+2)*size)) = v2
		*(*T)(unsafe.Add(data, uintptr(i+3)*size)) = v3
		*(*T)(unsafe.Add(data, uintptr(i+4)*size)) = v4
		*(*T)(unsafe.Add(data, uintptr(i+5)*size)) = v5
		*(*T)(unsafe.Add(data, uintptr(i+6)*size)) = v6
		*(*T)(unsafe.Add(data, uintptr(i+7)*size)) = v7
	}

	// Handle remaining elements
	for ; i < capacity; i++ {
		v := *(*T)(unsafe.Add(data, uintptr(i)*size))
		if v < min {
			v = min
		} else if v > max {
			v = max
		}
		*(*T)(unsafe.Add(data, uintptr(i)*size)) = v
	}
}
