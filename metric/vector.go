package metric

import (
	"blaze/core"
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
	inv := 1.0 / magnitude
	memstruct.VectorUnaryExecute(currentVector, newVectorAddr, func(a T) float32 {
		return float32(a) * inv
	}, core.BlazeDefaultStride)
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
	memstruct.VectorUnaryExecute(currentVector, newVectorAddr, func(a T) float64 {
		return float64(a) * inv
	}, core.BlazeDefaultStride)
}
