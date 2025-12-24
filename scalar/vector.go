package scalar

import (
	"blaze/core"
	"fmt"
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
	maxT := float64(foundation.MaxValue[T]())
	idx := 0

	memstruct.VectorUnaryExecute(vector, vector, func(_ T) T {
		v := float64(initial) + (float64(idx) * float64(step))
		if v > maxT {
			panic(fmt.Errorf(
				"overflow at idx %d: requested=%f, max=%f",
				idx, v, maxT,
			))
		}
		idx++
		return T(v)
	}, core.BlazeDefaultStride)
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
	memstruct.VectorUnaryExecute(currentVectorAddr, newVectorAddr, func(a T) float32 {
		return float32(a) * scalar
	}, core.BlazeDefaultStride)
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
	memstruct.VectorUnaryExecute(currentVectorAddr, newVectorAddr, func(a T) float32 {
		return float32(a) / scalar
	}, core.BlazeDefaultStride)
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
	memstruct.VectorUnaryExecute(currentVectorAddr, newVectorAddr, func(a T) float32 {
		return float32(a) + scalar
	}, core.BlazeDefaultStride)
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
	memstruct.VectorUnaryExecute(currentVectorAddr, newVectorAddr, func(a T) float32 {
		return float32(a) - scalar
	}, core.BlazeDefaultStride)
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
	memstruct.VectorUnaryExecute(currentVectorAddr, newVectorAddr, func(a T) float64 {
		return float64(a) * scalar
	}, core.BlazeDefaultStride)
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
	memstruct.VectorUnaryExecute(currentVectorAddr, newVectorAddr, func(a T) float64 {
		return float64(a) / scalar
	}, core.BlazeDefaultStride)
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
	memstruct.VectorUnaryExecute(currentVectorAddr, newVectorAddr, func(a T) float64 {
		return float64(a) + scalar
	}, core.BlazeDefaultStride)
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
	memstruct.VectorUnaryExecute(currentVectorAddr, newVectorAddr, func(a T) float64 {
		return float64(a) - scalar
	}, core.BlazeDefaultStride)
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
	memstruct.VectorUnaryExecute(vector, vector, func(a T) T {
		if a < min {
			return min
		} else if a > max {
			return max
		}
		return a
	}, core.BlazeDefaultStride)
}
