package elementwise

import (
	"blaze/core"
	"foundation"
	"memcore"
	"memstruct"
)

/*
BlazeElementWiseVectorAddF32 performs element-wise addition of two vectors in float32 precision.

Computes: result[i] = vectorA[i] + vectorB[i] for all elements i.

Use cases:
- Signal processing (combining signals)
- Image processing (blending images)
- Numerical simulations (combining vectors)
- Data transformations

Time complexity: O(n) - single pass through vectors
Space complexity: O(1) - only accumulator variables used

Prerequisites:
- Both input vectors must have the same length
- Output vector must have the same length as inputs
- All vectors must be valid and initialized

Edge cases:
- Works with any numeric type (int, float32, float64, etc.)
- Type conversions occur during computation (input types → float32)
- No overflow checking (relies on Go's numeric behavior)
*/
func BlazeElementWiseVectorAddF32[T, U foundation.Numeric](
	vectorAAddr, vectorBAddr, newVectorAddr memcore.MarkRaw,
) {
	memstruct.VectorBinaryExecute(vectorAAddr, vectorBAddr, newVectorAddr, func(a T, b U) float32 {
		return float32(a) + float32(b)
	}, core.BlazeDefaultStride)
}

/*
BlazeElementWiseVectorAddF64 performs element-wise addition of two vectors in float64 precision.

Computes: result[i] = vectorA[i] + vectorB[i] for all elements i.

Use cases:
- Signal processing (combining signals)
- Image processing (blending images)
- Numerical simulations (combining vectors)
- Data transformations

Time complexity: O(n) - single pass through vectors
Space complexity: O(1) - only accumulator variables used

Prerequisites:
- Both input vectors must have the same length
- Output vector must have the same length as inputs
- All vectors must be valid and initialized

Edge cases:
- Works with any numeric type (int, float32, float64, etc.)
- Type conversions occur during computation (input types → float64)
- No overflow checking (relies on Go's numeric behavior)
*/
func BlazeElementWiseVectorAddF64[T, U foundation.Numeric](
	vectorAAddr, vectorBAddr, newVectorAddr memcore.MarkRaw,
) {
	memstruct.VectorBinaryExecute(vectorAAddr, vectorBAddr, newVectorAddr, func(a T, b U) float64 {
		return float64(a) + float64(b)
	}, core.BlazeDefaultStride)
}

/*
BlazeElementWiseVectorSubtractF32 performs element-wise subtraction of two vectors in float32 precision.

Computes: result[i] = vectorA[i] - vectorB[i] for all elements i.

Use cases:
- Signal processing (difference between signals)
- Image processing (subtracting backgrounds)
- Numerical simulations (computing differences)
- Error calculations

Time complexity: O(n) - single pass through vectors
Space complexity: O(1) - only accumulator variables used

Prerequisites:
- Both input vectors must have the same length
- Output vector must have the same length as inputs
- All vectors must be valid and initialized

Edge cases:
- Works with any numeric type (int, float32, float64, etc.)
- Type conversions occur during computation (input types → float32)
- No overflow checking (relies on Go's numeric behavior)
*/
func BlazeElementWiseVectorSubtractF32[T, U foundation.Numeric](
	vectorAAddr, vectorBAddr, newVectorAddr memcore.MarkRaw,
) {
	memstruct.VectorBinaryExecute(vectorAAddr, vectorBAddr, newVectorAddr, func(a T, b U) float32 {
		return float32(a) - float32(b)
	}, core.BlazeDefaultStride)
}

/*
BlazeElementWiseVectorSubtractF64 performs element-wise subtraction of two vectors in float64 precision.

Computes: result[i] = vectorA[i] - vectorB[i] for all elements i.

Use cases:
- Signal processing (difference between signals)
- Image processing (subtracting backgrounds)
- Numerical simulations (computing differences)
- Error calculations

Time complexity: O(n) - single pass through vectors
Space complexity: O(1) - only accumulator variables used

Prerequisites:
- Both input vectors must have the same length
- Output vector must have the same length as inputs
- All vectors must be valid and initialized

Edge cases:
- Works with any numeric type (int, float32, float64, etc.)
- Type conversions occur during computation (input types → float64)
- No overflow checking (relies on Go's numeric behavior)
*/
func BlazeElementWiseVectorSubtractF64[T, U foundation.Numeric](
	vectorAAddr, vectorBAddr, newVectorAddr memcore.MarkRaw,
) {
	memstruct.VectorBinaryExecute(vectorAAddr, vectorBAddr, newVectorAddr, func(a T, b U) float64 {
		return float64(a) - float64(b)
	}, core.BlazeDefaultStride)
}

/*
BlazeElementWiseVectorMultiplyF32 performs element-wise multiplication of two vectors in float32 precision.

Computes: result[i] = vectorA[i] * vectorB[i] for all elements i.

Use cases:
- Signal processing (modulation, filtering)
- Image processing (masking, blending)
- Numerical simulations (element-wise scaling)
- Weighted operations

Time complexity: O(n) - single pass through vectors
Space complexity: O(1) - only accumulator variables used

Prerequisites:
- Both input vectors must have the same length
- Output vector must have the same length as inputs
- All vectors must be valid and initialized

Edge cases:
- Works with any numeric type (int, float32, float64, etc.)
- Type conversions occur during computation (input types → float32)
- No overflow checking (relies on Go's numeric behavior)
*/
func BlazeElementWiseVectorMultiplyF32[T, U foundation.Numeric](
	vectorAAddr, vectorBAddr, newVectorAddr memcore.MarkRaw,
) {
	memstruct.VectorBinaryExecute(vectorAAddr, vectorBAddr, newVectorAddr, func(a T, b U) float32 {
		return float32(a) * float32(b)
	}, core.BlazeDefaultStride)
}

/*
BlazeElementWiseVectorMultiplyF64 performs element-wise multiplication of two vectors in float64 precision.

Computes: result[i] = vectorA[i] * vectorB[i] for all elements i.

Use cases:
- Signal processing (modulation, filtering)
- Image processing (masking, blending)
- Numerical simulations (element-wise scaling)
- Weighted operations

Time complexity: O(n) - single pass through vectors
Space complexity: O(1) - only accumulator variables used

Prerequisites:
- Both input vectors must have the same length
- Output vector must have the same length as inputs
- All vectors must be valid and initialized

Edge cases:
- Works with any numeric type (int, float32, float64, etc.)
- Type conversions occur during computation (input types → float64)
- No overflow checking (relies on Go's numeric behavior)
*/
func BlazeElementWiseVectorMultiplyF64[T, U foundation.Numeric](
	vectorAAddr, vectorBAddr, newVectorAddr memcore.MarkRaw,
) {
	memstruct.VectorBinaryExecute(vectorAAddr, vectorBAddr, newVectorAddr, func(a T, b U) float64 {
		return float64(a) * float64(b)
	}, core.BlazeDefaultStride)
}

/*
BlazeElementWiseVectorDivideF32 performs element-wise division of two vectors in float32 precision.

Computes: result[i] = vectorA[i] / vectorB[i] for all elements i.

Use cases:
- Signal processing (normalization, deconvolution)
- Image processing (ratio images)
- Numerical simulations (element-wise scaling)
- Normalization operations

Time complexity: O(n) - single pass through vectors
Space complexity: O(1) - only accumulator variables used

Prerequisites:
- Both input vectors must have the same length
- Output vector must have the same length as inputs
- All vectors must be valid and initialized

Edge cases:
- Works with any numeric type (int, float32, float64, etc.)
- Type conversions occur during computation (input types → float32)
- Division by zero results in ±Inf or NaN (Go's standard behavior)
- No overflow checking (relies on Go's numeric behavior)
*/
func BlazeElementWiseVectorDivideF32[T, U foundation.Numeric](
	vectorAAddr, vectorBAddr, newVectorAddr memcore.MarkRaw,
) {
	memstruct.VectorBinaryExecute(vectorAAddr, vectorBAddr, newVectorAddr, func(a T, b U) float32 {
		return float32(a) / float32(b)
	}, core.BlazeDefaultStride)
}

/*
BlazeElementWiseVectorDivideF64 performs element-wise division of two vectors in float64 precision.

Computes: result[i] = vectorA[i] / vectorB[i] for all elements i.

Use cases:
- Signal processing (normalization, deconvolution)
- Image processing (ratio images)
- Numerical simulations (element-wise scaling)
- Normalization operations

Time complexity: O(n) - single pass through vectors
Space complexity: O(1) - only accumulator variables used

Prerequisites:
- Both input vectors must have the same length
- Output vector must have the same length as inputs
- All vectors must be valid and initialized

Edge cases:
- Works with any numeric type (int, float32, float64, etc.)
- Type conversions occur during computation (input types → float64)
- Division by zero results in ±Inf or NaN (Go's standard behavior)
- No overflow checking (relies on Go's numeric behavior)
*/
func BlazeElementWiseVectorDivideF64[T, U foundation.Numeric](
	vectorAAddr, vectorBAddr, newVectorAddr memcore.MarkRaw,
) {
	memstruct.VectorBinaryExecute(vectorAAddr, vectorBAddr, newVectorAddr, func(a T, b U) float64 {
		return float64(a) / float64(b)
	}, core.BlazeDefaultStride)
}
