package metric

import (
	"unsafe"

	"blaze/reduce"
	"foundation"
	"memcore"
	"memstruct"
)

/*
BlazeMetricVectorMagnitudeF32 computes the magnitude (L2 norm, Euclidean norm) of a vector in float32 precision.

Computes: magnitude = √(Σ(vector[i]²)) = ||vector||₂

The magnitude represents the length of the vector in Euclidean space.
It is the square root of the sum of squares of all elements.

Use cases:
- Distance calculations (Euclidean distance)
- Vector normalization (unit vectors)
- Signal processing (signal strength, power)
- Machine learning (feature magnitude, regularization)

Time complexity: O(n) - single pass through vector (uses BlazeReduceVectorSumSquaredF32)
Space complexity: O(1) - only accumulator variables used

Prerequisites:
- Vector must be valid and initialized

Edge cases:
- Returns 0.0 for empty vectors
- Works with any numeric type (int, float32, float64, etc.)
- Type conversions occur during computation (input types → float32)
- Potential precision loss for very large values
- Returns 0.0 if all elements are zero
*/
func BlazeMetricVectorMagnitudeF32[T foundation.Numeric](vector memcore.MarkRaw) float32 {
	sqrSum := reduce.BlazeReduceVectorSumSquaredF32[T](vector)
	return foundation.Sqrt32(sqrSum)
}

/*
BlazeMetricVectorMagnitudeF64 computes the magnitude (L2 norm, Euclidean norm) of a vector in float64 precision.

Computes: magnitude = √(Σ(vector[i]²)) = ||vector||₂

The magnitude represents the length of the vector in Euclidean space.
It is the square root of the sum of squares of all elements.

Use cases:
- Distance calculations (Euclidean distance)
- Vector normalization (unit vectors)
- Signal processing (signal strength, power)
- Machine learning (feature magnitude, regularization)

Time complexity: O(n) - single pass through vector (uses BlazeReduceVectorSumSquaredF64)
Space complexity: O(1) - only accumulator variables used

Prerequisites:
- Vector must be valid and initialized

Edge cases:
- Returns 0.0 for empty vectors
- Works with any numeric type (int, float32, float64, etc.)
- Type conversions occur during computation (input types → float64)
- Better precision than F32 variant for large values
- Returns 0.0 if all elements are zero
*/
func BlazeMetricVectorMagnitudeF64[T foundation.Numeric](vector memcore.MarkRaw) float64 {
	sqrSum := reduce.BlazeReduceVectorSumSquaredF64[T](vector)
	return foundation.Sqrt64(sqrSum)
}

/*
BlazeMetricVectorNormalizedF32 normalizes a vector to unit length (magnitude = 1) in float32 precision.

Computes: result[i] = vector[i] / magnitude for all elements i.

Normalization scales the vector so its magnitude becomes 1.0 while preserving direction.
This is useful for direction vectors, unit vectors, and cosine similarity calculations.

Use cases:
- Creating unit vectors (direction vectors)
- Cosine similarity preprocessing
- Machine learning (feature normalization)
- Signal processing (normalized signals)

Time complexity: O(n) - single pass through vector (uses BlazeMetricVectorMagnitudeF32)
Space complexity: O(1) - only accumulator variables used

Prerequisites:
- Input vector must be valid and initialized
- Output vector must have the same length as input
- Both vectors must be valid and initialized

Edge cases:
- Panics if vector magnitude is zero (division by zero)
- Works with any numeric type (int, float32, float64, etc.)
- Type conversions occur during computation (input types → float32)
- Result vector has magnitude = 1.0 (within floating-point precision)
- If input is zero vector, operation will panic
*/
func BlazeMetricVectorNormalizedF32[T foundation.Numeric](
	currentVector memcore.MarkRaw,
	newVectorAddr memcore.MarkRaw,
) {
	magnitude := BlazeMetricVectorMagnitudeF32[T](currentVector)
	inv := float32(1.0 / magnitude)

	srcData := memstruct.VectorDataPtrGet[T](currentVector)
	dstData := memstruct.VectorDataPtrGet[float32](newVectorAddr)
	srcSize := uintptr(memcore.SizeOf[T]())
	dstSize := uintptr(memcore.SizeOf[float32]())
	capacity := memstruct.VectorCapacityGet[T](currentVector)

	var i uint64

	// Manually unrolled loop for stride 8
	for ; i+7 < capacity; i += 8 {
		v0 := float32(*(*T)(unsafe.Add(srcData, uintptr(i+0)*srcSize))) * inv
		v1 := float32(*(*T)(unsafe.Add(srcData, uintptr(i+1)*srcSize))) * inv
		v2 := float32(*(*T)(unsafe.Add(srcData, uintptr(i+2)*srcSize))) * inv
		v3 := float32(*(*T)(unsafe.Add(srcData, uintptr(i+3)*srcSize))) * inv
		v4 := float32(*(*T)(unsafe.Add(srcData, uintptr(i+4)*srcSize))) * inv
		v5 := float32(*(*T)(unsafe.Add(srcData, uintptr(i+5)*srcSize))) * inv
		v6 := float32(*(*T)(unsafe.Add(srcData, uintptr(i+6)*srcSize))) * inv
		v7 := float32(*(*T)(unsafe.Add(srcData, uintptr(i+7)*srcSize))) * inv

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
		v := float32(*(*T)(unsafe.Add(srcData, uintptr(i)*srcSize))) * inv
		*(*float32)(unsafe.Add(dstData, uintptr(i)*dstSize)) = v
	}
}

/*
BlazeMetricVectorNormalizedF64 normalizes a vector to unit length (magnitude = 1) in float64 precision.

Computes: result[i] = vector[i] / magnitude for all elements i.

Normalization scales the vector so its magnitude becomes 1.0 while preserving direction.
This is useful for direction vectors, unit vectors, and cosine similarity calculations.

Use cases:
- Creating unit vectors (direction vectors)
- Cosine similarity preprocessing
- Machine learning (feature normalization)
- Signal processing (normalized signals)

Time complexity: O(n) - single pass through vector (uses BlazeMetricVectorMagnitudeF64)
Space complexity: O(1) - only accumulator variables used

Prerequisites:
- Input vector must be valid and initialized
- Output vector must have the same length as input
- Both vectors must be valid and initialized

Edge cases:
- Panics if vector magnitude is zero (division by zero)
- Works with any numeric type (int, float32, float64, etc.)
- Type conversions occur during computation (input types → float64)
- Better precision than F32 variant
- Result vector has magnitude = 1.0 (within floating-point precision)
- If input is zero vector, operation will panic
*/
func BlazeMetricVectorNormalizedF64[T foundation.Numeric](
	currentVector memcore.MarkRaw,
	newVectorAddr memcore.MarkRaw,
) {
	magnitude := BlazeMetricVectorMagnitudeF64[T](currentVector)
	inv := 1.0 / magnitude

	srcData := memstruct.VectorDataPtrGet[T](currentVector)
	dstData := memstruct.VectorDataPtrGet[float64](newVectorAddr)
	srcSize := uintptr(memcore.SizeOf[T]())
	dstSize := uintptr(memcore.SizeOf[float64]())
	capacity := memstruct.VectorCapacityGet[T](currentVector)

	var i uint64

	// Manually unrolled loop for stride 8
	for ; i+7 < capacity; i += 8 {
		v0 := float64(*(*T)(unsafe.Add(srcData, uintptr(i+0)*srcSize))) * inv
		v1 := float64(*(*T)(unsafe.Add(srcData, uintptr(i+1)*srcSize))) * inv
		v2 := float64(*(*T)(unsafe.Add(srcData, uintptr(i+2)*srcSize))) * inv
		v3 := float64(*(*T)(unsafe.Add(srcData, uintptr(i+3)*srcSize))) * inv
		v4 := float64(*(*T)(unsafe.Add(srcData, uintptr(i+4)*srcSize))) * inv
		v5 := float64(*(*T)(unsafe.Add(srcData, uintptr(i+5)*srcSize))) * inv
		v6 := float64(*(*T)(unsafe.Add(srcData, uintptr(i+6)*srcSize))) * inv
		v7 := float64(*(*T)(unsafe.Add(srcData, uintptr(i+7)*srcSize))) * inv

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
		v := float64(*(*T)(unsafe.Add(srcData, uintptr(i)*srcSize))) * inv
		*(*float64)(unsafe.Add(dstData, uintptr(i)*dstSize)) = v
	}
}
