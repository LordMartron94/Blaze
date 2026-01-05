package scalar

import (
	"blaze/core"
	"fmt"
	"foundation"
	"memcore"
	"memstruct"
)

/*
BlazeScalarMatrixSetAllSequence fills a matrix with an arithmetic sequence.

Sets each element to: value[i] = initial + (i * step) where i is the element index
(row-major order).

Use cases:
- Creating test data (sequences, ranges)
- Initializing matrices with patterns
- Generating indices or coordinates
- Data preprocessing

Time complexity: O(m*n) - single pass through all matrix elements
Space complexity: O(1) - only accumulator variables used

Prerequisites:
- Matrix must be valid and initialized

Edge cases:
- Panics if any computed value exceeds the numeric type's maximum value
- Works with any numeric type (int, float32, float64, etc.)
- No type conversions (uses same type as input)
- Overflow checking prevents silent data corruption
*/
func BlazeScalarMatrixSetAllSequence[T foundation.Numeric](matrix memcore.MarkRaw, initial, step T) {
	maxT := float64(foundation.MaxValue[T]())
	idx := 0

	memstruct.MatrixUnaryExecute(matrix, matrix, func(_ T) T {
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
BlazeScalarMatrixMultiplyF32 multiplies each matrix element by a scalar value in float32 precision.

Computes: result[i,j] = matrix[i,j] * scalar for all elements.

Use cases:
- Scaling data (normalization, unit conversion)
- Signal processing (amplification, attenuation)
- Machine learning (feature scaling)
- Numerical simulations (scaling factors)

Time complexity: O(m*n) - single pass through all matrix elements
Space complexity: O(1) - only accumulator variables used

Prerequisites:
- Input matrix must be valid and initialized
- Output matrix must have the same dimensions as input
- Both matrices must be valid and initialized

Edge cases:
- Works with any numeric type (int, float32, float64, etc.)
- Type conversions occur during computation (input types → float32)
- No overflow checking (relies on Go's numeric behavior)
*/
func BlazeScalarMatrixMultiplyF32[T foundation.Numeric](
	currentMatrixAddr, newMatrixAddr memcore.MarkRaw,
	scalar float32,
) {
	memstruct.MatrixUnaryExecute(currentMatrixAddr, newMatrixAddr, func(a T) float32 {
		return float32(a) * scalar
	}, core.BlazeDefaultStride)
}

/*
BlazeScalarMatrixDivideF32 divides each matrix element by a scalar value in float32 precision.

Computes: result[i,j] = matrix[i,j] / scalar for all elements.

Use cases:
- Normalization (dividing by sum, mean, etc.)
- Unit conversion (scaling factors)
- Signal processing (attenuation)
- Machine learning (feature normalization)

Time complexity: O(m*n) - single pass through all matrix elements
Space complexity: O(1) - only accumulator variables used

Prerequisites:
- Input matrix must be valid and initialized
- Output matrix must have the same dimensions as input
- Both matrices must be valid and initialized

Edge cases:
- Works with any numeric type (int, float32, float64, etc.)
- Type conversions occur during computation (input types → float32)
- Division by zero results in ±Inf or NaN (Go's standard behavior)
- No overflow checking (relies on Go's numeric behavior)
*/
func BlazeScalarMatrixDivideF32[T foundation.Numeric](
	currentMatrixAddr, newMatrixAddr memcore.MarkRaw,
	scalar float32,
) {
	memstruct.MatrixUnaryExecute(currentMatrixAddr, newMatrixAddr, func(a T) float32 {
		return float32(a) / scalar
	}, core.BlazeDefaultStride)
}

/*
BlazeScalarMatrixAddF32 adds a scalar value to each matrix element in float32 precision.

Computes: result[i,j] = matrix[i,j] + scalar for all elements.

Use cases:
- Shifting data (offset adjustments)
- Signal processing (DC bias)
- Data preprocessing (centering, offsetting)
- Numerical simulations (translations)

Time complexity: O(m*n) - single pass through all matrix elements
Space complexity: O(1) - only accumulator variables used

Prerequisites:
- Input matrix must be valid and initialized
- Output matrix must have the same dimensions as input
- Both matrices must be valid and initialized

Edge cases:
- Works with any numeric type (int, float32, float64, etc.)
- Type conversions occur during computation (input types → float32)
- No overflow checking (relies on Go's numeric behavior)
*/
func BlazeScalarMatrixAddF32[T foundation.Numeric](
	currentMatrixAddr, newMatrixAddr memcore.MarkRaw,
	scalar float32,
) {
	memstruct.MatrixUnaryExecute(currentMatrixAddr, newMatrixAddr, func(a T) float32 {
		return float32(a) + scalar
	}, core.BlazeDefaultStride)
}

/*
BlazeScalarMatrixSubtractF32 subtracts a scalar value from each matrix element in float32 precision.

Computes: result[i,j] = matrix[i,j] - scalar for all elements.

Use cases:
- Shifting data (offset adjustments)
- Signal processing (DC removal)
- Data preprocessing (centering, mean subtraction)
- Numerical simulations (translations)

Time complexity: O(m*n) - single pass through all matrix elements
Space complexity: O(1) - only accumulator variables used

Prerequisites:
- Input matrix must be valid and initialized
- Output matrix must have the same dimensions as input
- Both matrices must be valid and initialized

Edge cases:
- Works with any numeric type (int, float32, float64, etc.)
- Type conversions occur during computation (input types → float32)
- No overflow checking (relies on Go's numeric behavior)
*/
func BlazeScalarMatrixSubtractF32[T foundation.Numeric](
	currentMatrixAddr, newMatrixAddr memcore.MarkRaw,
	scalar float32,
) {
	memstruct.MatrixUnaryExecute(currentMatrixAddr, newMatrixAddr, func(a T) float32 {
		return float32(a) - scalar
	}, core.BlazeDefaultStride)
}

// ────────────────────────────────────────────────────────────────
//   FLOAT64 PRECISION OPERATIONS
// ────────────────────────────────────────────────────────────────

/*
BlazeScalarMatrixMultiplyF64 multiplies each matrix element by a scalar value in float64 precision.

Computes: result[i,j] = matrix[i,j] * scalar for all elements.

Use cases:
- Scaling data (normalization, unit conversion)
- Signal processing (amplification, attenuation)
- Machine learning (feature scaling)
- Numerical simulations (scaling factors)

Time complexity: O(m*n) - single pass through all matrix elements
Space complexity: O(1) - only accumulator variables used

Prerequisites:
- Input matrix must be valid and initialized
- Output matrix must have the same dimensions as input
- Both matrices must be valid and initialized

Edge cases:
- Works with any numeric type (int, float32, float64, etc.)
- Type conversions occur during computation (input types → float64)
- Better precision than F32 variant
- No overflow checking (relies on Go's numeric behavior)
*/
func BlazeScalarMatrixMultiplyF64[T foundation.Numeric](
	currentMatrixAddr, newMatrixAddr memcore.MarkRaw,
	scalar float64,
) {
	memstruct.MatrixUnaryExecute(currentMatrixAddr, newMatrixAddr, func(a T) float64 {
		return float64(a) * scalar
	}, core.BlazeDefaultStride)
}

/*
BlazeScalarMatrixDivideF64 divides each matrix element by a scalar value in float64 precision.

Computes: result[i,j] = matrix[i,j] / scalar for all elements.

Use cases:
- Normalization (dividing by sum, mean, etc.)
- Unit conversion (scaling factors)
- Signal processing (attenuation)
- Machine learning (feature normalization)

Time complexity: O(m*n) - single pass through all matrix elements
Space complexity: O(1) - only accumulator variables used

Prerequisites:
- Input matrix must be valid and initialized
- Output matrix must have the same dimensions as input
- Both matrices must be valid and initialized

Edge cases:
- Works with any numeric type (int, float32, float64, etc.)
- Type conversions occur during computation (input types → float64)
- Division by zero results in ±Inf or NaN (Go's standard behavior)
- Better precision than F32 variant
- No overflow checking (relies on Go's numeric behavior)
*/
func BlazeScalarMatrixDivideF64[T foundation.Numeric](
	currentMatrixAddr, newMatrixAddr memcore.MarkRaw,
	scalar float64,
) {
	memstruct.MatrixUnaryExecute(currentMatrixAddr, newMatrixAddr, func(a T) float64 {
		return float64(a) / scalar
	}, core.BlazeDefaultStride)
}

/*
BlazeScalarMatrixAddF64 adds a scalar value to each matrix element in float64 precision.

Computes: result[i,j] = matrix[i,j] + scalar for all elements.

Use cases:
- Shifting data (offset adjustments)
- Signal processing (DC bias)
- Data preprocessing (centering, offsetting)
- Numerical simulations (translations)

Time complexity: O(m*n) - single pass through all matrix elements
Space complexity: O(1) - only accumulator variables used

Prerequisites:
- Input matrix must be valid and initialized
- Output matrix must have the same dimensions as input
- Both matrices must be valid and initialized

Edge cases:
- Works with any numeric type (int, float32, float64, etc.)
- Type conversions occur during computation (input types → float64)
- Better precision than F32 variant
- No overflow checking (relies on Go's numeric behavior)
*/
func BlazeScalarMatrixAddF64[T foundation.Numeric](
	currentMatrixAddr, newMatrixAddr memcore.MarkRaw,
	scalar float64,
) {
	memstruct.MatrixUnaryExecute(currentMatrixAddr, newMatrixAddr, func(a T) float64 {
		return float64(a) + scalar
	}, core.BlazeDefaultStride)
}

/*
BlazeScalarMatrixSubtractF64 subtracts a scalar value from each matrix element in float64 precision.

Computes: result[i,j] = matrix[i,j] - scalar for all elements.

Use cases:
- Shifting data (offset adjustments)
- Signal processing (DC removal)
- Data preprocessing (centering, mean subtraction)
- Numerical simulations (translations)

Time complexity: O(m*n) - single pass through all matrix elements
Space complexity: O(1) - only accumulator variables used

Prerequisites:
- Input matrix must be valid and initialized
- Output matrix must have the same dimensions as input
- Both matrices must be valid and initialized

Edge cases:
- Works with any numeric type (int, float32, float64, etc.)
- Type conversions occur during computation (input types → float64)
- Better precision than F32 variant
- No overflow checking (relies on Go's numeric behavior)
*/
func BlazeScalarMatrixSubtractF64[T foundation.Numeric](
	currentMatrixAddr, newMatrixAddr memcore.MarkRaw,
	scalar float64,
) {
	memstruct.MatrixUnaryExecute(currentMatrixAddr, newMatrixAddr, func(a T) float64 {
		return float64(a) - scalar
	}, core.BlazeDefaultStride)
}

// ────────────────────────────────────────────────────────────────
//   MISC
// ────────────────────────────────────────────────────────────────

/*
BlazeScalarMatrixClamp clamps each matrix element to lie within [min, max] range.

For each element: if value < min, set to min; if value > max, set to max; otherwise keep value.
This operation mutates the existing matrix in place.

Use cases:
- Data validation (enforcing bounds)
- Signal processing (limiting amplitude)
- Image processing (clipping pixel values)
- Numerical stability (preventing overflow)

Time complexity: O(m*n) - single pass through all matrix elements
Space complexity: O(1) - only accumulator variables used

Prerequisites:
- Matrix must be valid and initialized
- min must be <= max (undefined behavior if min > max)

Edge cases:
- Works with any numeric type (int, float32, float64, etc.)
- No type conversions (uses same type as input)
- In-place operation (modifies input matrix)
- If min > max, elements may be set inconsistently
*/
func BlazeScalarMatrixClamp[T foundation.Numeric](
	matrix memcore.MarkRaw,
	min, max T,
) {
	memstruct.MatrixUnaryExecute(matrix, matrix, func(a T) T {
		if a < min {
			return min
		} else if a > max {
			return max
		}
		return a
	}, core.BlazeDefaultStride)
}
