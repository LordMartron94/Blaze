package elementwise

import (
	"blaze/core"
	"foundation"
	"memcore"
	"memstruct"
)

/*
BlazeElementWiseMatrixAddF32 performs element-wise addition of two matrices in float32 precision.

Computes: result[i,j] = matrixA[i,j] + matrixB[i,j] for all elements.

Use cases:
- Image processing (blending images, combining layers)
- Signal processing (combining multi-channel signals)
- Numerical simulations (combining matrices)
- Data transformations

Time complexity: O(m*n) - single pass through all matrix elements
Space complexity: O(1) - only accumulator variables used

Prerequisites:
- Both input matrices must have the same dimensions (m×n)
- Output matrix must have the same dimensions as inputs
- All matrices must be valid and initialized

Edge cases:
- Works with any numeric type (int, float32, float64, etc.)
- Type conversions occur during computation (input types → float32)
- No overflow checking (relies on Go's numeric behavior)

Note: This is element-wise addition, not standard matrix multiplication.
For standard matrix multiplication, see blaze/structure.
*/
func BlazeElementWiseMatrixAddF32[T, U foundation.Numeric](
	matrixAAddr, matrixBAddr, newMatrixAddr memcore.MarkRaw,
) {
	memstruct.MatrixBinaryExecute(matrixAAddr, matrixBAddr, newMatrixAddr, func(a T, b U) float32 {
		return float32(a) + float32(b)
	}, core.BlazeDefaultStride)
}

/*
BlazeElementWiseMatrixAddF64 performs element-wise addition of two matrices in float64 precision.

Computes: result[i,j] = matrixA[i,j] + matrixB[i,j] for all elements.

Use cases:
- Image processing (blending images, combining layers)
- Signal processing (combining multi-channel signals)
- Numerical simulations (combining matrices)
- Data transformations

Time complexity: O(m*n) - single pass through all matrix elements
Space complexity: O(1) - only accumulator variables used

Prerequisites:
- Both input matrices must have the same dimensions (m×n)
- Output matrix must have the same dimensions as inputs
- All matrices must be valid and initialized

Edge cases:
- Works with any numeric type (int, float32, float64, etc.)
- Type conversions occur during computation (input types → float64)
- No overflow checking (relies on Go's numeric behavior)

Note: This is element-wise addition, not standard matrix multiplication.
For standard matrix multiplication, see blaze/structure.
*/
func BlazeElementWiseMatrixAddF64[T, U foundation.Numeric](
	matrixAAddr, matrixBAddr, newMatrixAddr memcore.MarkRaw,
) {
	memstruct.MatrixBinaryExecute(matrixAAddr, matrixBAddr, newMatrixAddr, func(a T, b U) float64 {
		return float64(a) + float64(b)
	}, core.BlazeDefaultStride)
}

/*
BlazeElementWiseMatrixSubtractF32 performs element-wise subtraction of two matrices in float32 precision.

Computes: result[i,j] = matrixA[i,j] - matrixB[i,j] for all elements.

Use cases:
- Image processing (subtracting backgrounds, difference images)
- Signal processing (difference between signals)
- Numerical simulations (computing differences)
- Error calculations

Time complexity: O(m*n) - single pass through all matrix elements
Space complexity: O(1) - only accumulator variables used

Prerequisites:
- Both input matrices must have the same dimensions (m×n)
- Output matrix must have the same dimensions as inputs
- All matrices must be valid and initialized

Edge cases:
- Works with any numeric type (int, float32, float64, etc.)
- Type conversions occur during computation (input types → float32)
- No overflow checking (relies on Go's numeric behavior)

Note: This is element-wise subtraction, not standard matrix multiplication.
For standard matrix multiplication, see blaze/structure.
*/
func BlazeElementWiseMatrixSubtractF32[T, U foundation.Numeric](
	matrixAAddr, matrixBAddr, newMatrixAddr memcore.MarkRaw,
) {
	memstruct.MatrixBinaryExecute(matrixAAddr, matrixBAddr, newMatrixAddr, func(a T, b U) float32 {
		return float32(a) - float32(b)
	}, core.BlazeDefaultStride)
}

/*
BlazeElementWiseMatrixSubtractF64 performs element-wise subtraction of two matrices in float64 precision.

Computes: result[i,j] = matrixA[i,j] - matrixB[i,j] for all elements.

Use cases:
- Image processing (subtracting backgrounds, difference images)
- Signal processing (difference between signals)
- Numerical simulations (computing differences)
- Error calculations

Time complexity: O(m*n) - single pass through all matrix elements
Space complexity: O(1) - only accumulator variables used

Prerequisites:
- Both input matrices must have the same dimensions (m×n)
- Output matrix must have the same dimensions as inputs
- All matrices must be valid and initialized

Edge cases:
- Works with any numeric type (int, float32, float64, etc.)
- Type conversions occur during computation (input types → float64)
- No overflow checking (relies on Go's numeric behavior)

Note: This is element-wise subtraction, not standard matrix multiplication.
For standard matrix multiplication, see blaze/structure.
*/
func BlazeElementWiseMatrixSubtractF64[T, U foundation.Numeric](
	matrixAAddr, matrixBAddr, newMatrixAddr memcore.MarkRaw,
) {
	memstruct.MatrixBinaryExecute(matrixAAddr, matrixBAddr, newMatrixAddr, func(a T, b U) float64 {
		return float64(a) - float64(b)
	}, core.BlazeDefaultStride)
}

/*
BlazeElementWiseMatrixMultiplyF32 performs element-wise multiplication of two matrices in float32 precision.

Computes: result[i,j] = matrixA[i,j] * matrixB[i,j] for all elements.

Use cases:
- Image processing (masking, blending, filtering)
- Signal processing (modulation, element-wise filtering)
- Numerical simulations (element-wise scaling)
- Weighted operations

Time complexity: O(m*n) - single pass through all matrix elements
Space complexity: O(1) - only accumulator variables used

Prerequisites:
- Both input matrices must have the same dimensions (m×n)
- Output matrix must have the same dimensions as inputs
- All matrices must be valid and initialized

Edge cases:
- Works with any numeric type (int, float32, float64, etc.)
- Type conversions occur during computation (input types → float32)
- No overflow checking (relies on Go's numeric behavior)

Note: This is element-wise multiplication (Hadamard product), not standard matrix multiplication.
For standard matrix multiplication (C = A × B), see blaze/structure.
*/
func BlazeElementWiseMatrixMultiplyF32[T, U foundation.Numeric](
	matrixAAddr, matrixBAddr, newMatrixAddr memcore.MarkRaw,
) {
	memstruct.MatrixBinaryExecute(matrixAAddr, matrixBAddr, newMatrixAddr, func(a T, b U) float32 {
		return float32(a) * float32(b)
	}, core.BlazeDefaultStride)
}

/*
BlazeElementWiseMatrixMultiplyF64 performs element-wise multiplication of two matrices in float64 precision.

Computes: result[i,j] = matrixA[i,j] * matrixB[i,j] for all elements.

Use cases:
- Image processing (masking, blending, filtering)
- Signal processing (modulation, element-wise filtering)
- Numerical simulations (element-wise scaling)
- Weighted operations

Time complexity: O(m*n) - single pass through all matrix elements
Space complexity: O(1) - only accumulator variables used

Prerequisites:
- Both input matrices must have the same dimensions (m×n)
- Output matrix must have the same dimensions as inputs
- All matrices must be valid and initialized

Edge cases:
- Works with any numeric type (int, float32, float64, etc.)
- Type conversions occur during computation (input types → float64)
- No overflow checking (relies on Go's numeric behavior)

Note: This is element-wise multiplication (Hadamard product), not standard matrix multiplication.
For standard matrix multiplication (C = A × B), see blaze/structure.
*/
func BlazeElementWiseMatrixMultiplyF64[T, U foundation.Numeric](
	matrixAAddr, matrixBAddr, newMatrixAddr memcore.MarkRaw,
) {
	memstruct.MatrixBinaryExecute(matrixAAddr, matrixBAddr, newMatrixAddr, func(a T, b U) float64 {
		return float64(a) * float64(b)
	}, core.BlazeDefaultStride)
}

/*
BlazeElementWiseMatrixDivideF32 performs element-wise division of two matrices in float32 precision.

Computes: result[i,j] = matrixA[i,j] / matrixB[i,j] for all elements.

Use cases:
- Image processing (ratio images, normalization)
- Signal processing (normalization, deconvolution)
- Numerical simulations (element-wise scaling)
- Normalization operations

Time complexity: O(m*n) - single pass through all matrix elements
Space complexity: O(1) - only accumulator variables used

Prerequisites:
- Both input matrices must have the same dimensions (m×n)
- Output matrix must have the same dimensions as inputs
- All matrices must be valid and initialized

Edge cases:
- Works with any numeric type (int, float32, float64, etc.)
- Type conversions occur during computation (input types → float32)
- Division by zero results in ±Inf or NaN (Go's standard behavior)
- No overflow checking (relies on Go's numeric behavior)

Note: This is element-wise division, not standard matrix multiplication.
For standard matrix multiplication, see blaze/structure.
*/
func BlazeElementWiseMatrixDivideF32[T, U foundation.Numeric](
	matrixAAddr, matrixBAddr, newMatrixAddr memcore.MarkRaw,
) {
	memstruct.MatrixBinaryExecute(matrixAAddr, matrixBAddr, newMatrixAddr, func(a T, b U) float32 {
		return float32(a) / float32(b)
	}, core.BlazeDefaultStride)
}

/*
BlazeElementWiseMatrixDivideF64 performs element-wise division of two matrices in float64 precision.

Computes: result[i,j] = matrixA[i,j] / matrixB[i,j] for all elements.

Use cases:
- Image processing (ratio images, normalization)
- Signal processing (normalization, deconvolution)
- Numerical simulations (element-wise scaling)
- Normalization operations

Time complexity: O(m*n) - single pass through all matrix elements
Space complexity: O(1) - only accumulator variables used

Prerequisites:
- Both input matrices must have the same dimensions (m×n)
- Output matrix must have the same dimensions as inputs
- All matrices must be valid and initialized

Edge cases:
- Works with any numeric type (int, float32, float64, etc.)
- Type conversions occur during computation (input types → float64)
- Division by zero results in ±Inf or NaN (Go's standard behavior)
- No overflow checking (relies on Go's numeric behavior)

Note: This is element-wise division, not standard matrix multiplication.
For standard matrix multiplication, see blaze/structure.
*/
func BlazeElementWiseMatrixDivideF64[T, U foundation.Numeric](
	matrixAAddr, matrixBAddr, newMatrixAddr memcore.MarkRaw,
) {
	memstruct.MatrixBinaryExecute(matrixAAddr, matrixBAddr, newMatrixAddr, func(a T, b U) float64 {
		return float64(a) / float64(b)
	}, core.BlazeDefaultStride)
}
