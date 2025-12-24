package elementwise

import (
	"blaze/core"
	"fmt"
	"foundation"
	"memcore"
	"memstruct"
)

// BlazeElementWiseMatrixAddF32 adds the values of Matrix B to Matrix A, resulting in Matrix C at newMatrixAddr.
// It does so in float32 precision.
func BlazeElementWiseMatrixAddF32[T, U foundation.Numeric](
	matrixAAddr, matrixBAddr, newMatrixAddr memcore.MarkRaw,
) {
	memstruct.MatrixBinaryExecute(matrixAAddr, matrixBAddr, newMatrixAddr, func(a T, b U) float32 {
		return float32(a) + float32(b)
	}, core.BlazeDefaultStride)
}

// BlazeElementWiseMatrixAddF64 adds the values of Matrix B to Matrix A, resulting in Matrix C at newMatrixAddr.
// It does so in float64 precision.
func BlazeElementWiseMatrixAddF64[T, U foundation.Numeric](
	matrixAAddr, matrixBAddr, newMatrixAddr memcore.MarkRaw,
) {
	memstruct.MatrixBinaryExecute(matrixAAddr, matrixBAddr, newMatrixAddr, func(a T, b U) float64 {
		return float64(a) + float64(b)
	}, core.BlazeDefaultStride)
}

// BlazeElementWiseMatrixSubtractF32 subtracts the values of Matrix B from Matrix A, resulting in Matrix C at newMatrixAddr.
// It does so in float32 precision.
func BlazeElementWiseMatrixSubtractF32[T, U foundation.Numeric](
	matrixAAddr, matrixBAddr, newMatrixAddr memcore.MarkRaw,
) {
	memstruct.MatrixBinaryExecute(matrixAAddr, matrixBAddr, newMatrixAddr, func(a T, b U) float32 {
		return float32(a) - float32(b)
	}, core.BlazeDefaultStride)
}

// BlazeElementWiseMatrixSubtractF64 subtracts the values of Matrix B from Matrix A, resulting in Matrix C at newMatrixAddr.
// It does so in float364precision.
func BlazeElementWiseMatrixSubtractF64[T, U foundation.Numeric](
	matrixAAddr, matrixBAddr, newMatrixAddr memcore.MarkRaw,
) {
	memstruct.MatrixBinaryExecute(matrixAAddr, matrixBAddr, newMatrixAddr, func(a T, b U) float64 {
		return float64(a) - float64(b)
	}, core.BlazeDefaultStride)
}

/*
BlazeElementWiseMatrixMultiplyF32 computes standard matrix multiplication C = A × B in float32 precision.

Standard matrix multiplication computes: C[i,j] = Σ(A[i,k] * B[k,j]) for k in [0, n)
where A is m×n, B is n×p, and C is m×p.

Use cases:
- Linear algebra operations
- Computing X^T X for covariance matrices
- Matrix transformations
- Statistical computations (multivariate analysis)

Time complexity: O(m * n * p) - standard matrix multiplication
Space complexity: O(1) - only accumulator variables used

Prerequisites:
- Matrix A must be m×n
- Matrix B must be n×p
- Matrix C (output) must be m×p
- All matrices must be valid and initialized

Edge cases:
- Panics if dimensions don't match (A.cols != B.rows)
- Panics if output matrix dimensions don't match (C.rows != A.rows or C.cols != B.cols)
- Works with any numeric type (int, float32, float64, etc.)

The implementation uses nested loops with cache-friendly access patterns.
Future optimizations could include SIMD vectorization or BLAS integration.
*/
func BlazeElementWiseMatrixMultiplyF32[T, U foundation.Numeric](
	matrixAAddr, matrixBAddr, matrixCAddr memcore.MarkRaw,
) {
	aRows := memstruct.MatrixRowsGet[T](matrixAAddr)
	aCols := memstruct.MatrixColsGet[T](matrixAAddr)
	bRows := memstruct.MatrixRowsGet[U](matrixBAddr)
	bCols := memstruct.MatrixColsGet[U](matrixBAddr)
	cRows := memstruct.MatrixRowsGet[float32](matrixCAddr)
	cCols := memstruct.MatrixColsGet[float32](matrixCAddr)

	// Validate dimensions: A (m×n) × B (n×p) → C (m×p)
	if aCols != bRows {
		panic(fmt.Errorf(
			"BlazeElementWiseMatrixMultiplyF32: dimension mismatch (A.cols=%d != B.rows=%d)",
			aCols, bRows,
		))
	}
	if cRows != aRows || cCols != bCols {
		panic(fmt.Errorf(
			"BlazeElementWiseMatrixMultiplyF32: output dimension mismatch (C=%dx%d, expected=%dx%d)",
			cRows, cCols, aRows, bCols,
		))
	}

	// Standard matrix multiplication: C[i,j] = Σ(A[i,k] * B[k,j])
	for i := uint64(0); i < aRows; i++ {
		for j := uint64(0); j < bCols; j++ {
			var sum float32 = 0
			for k := uint64(0); k < aCols; k++ {
				aVal := memstruct.MatrixItemGetAtUnsafe[T](matrixAAddr, i, k)
				bVal := memstruct.MatrixItemGetAtUnsafe[U](matrixBAddr, k, j)
				sum += float32(aVal) * float32(bVal)
			}
			memstruct.MatrixSetAtUnsafe(matrixCAddr, i, j, sum)
		}
	}
}

/*
BlazeElementWiseMatrixMultiplyF64 computes standard matrix multiplication C = A × B in float64 precision.

Standard matrix multiplication computes: C[i,j] = Σ(A[i,k] * B[k,j]) for k in [0, n)
where A is m×n, B is n×p, and C is m×p.

Use cases:
- Linear algebra operations
- Computing X^T X for covariance matrices
- Matrix transformations
- Statistical computations (multivariate analysis)

Time complexity: O(m * n * p) - standard matrix multiplication
Space complexity: O(1) - only accumulator variables used

Prerequisites:
- Matrix A must be m×n
- Matrix B must be n×p
- Matrix C (output) must be m×p
- All matrices must be valid and initialized

Edge cases:
- Panics if dimensions don't match (A.cols != B.rows)
- Panics if output matrix dimensions don't match (C.rows != A.rows or C.cols != B.cols)
- Works with any numeric type (int, float32, float64, etc.)

The implementation uses nested loops with cache-friendly access patterns.
Future optimizations could include SIMD vectorization or BLAS integration.
*/
func BlazeElementWiseMatrixMultiplyF64[T, U foundation.Numeric](
	matrixAAddr, matrixBAddr, matrixCAddr memcore.MarkRaw,
) {
	aRows := memstruct.MatrixRowsGet[T](matrixAAddr)
	aCols := memstruct.MatrixColsGet[T](matrixAAddr)
	bRows := memstruct.MatrixRowsGet[U](matrixBAddr)
	bCols := memstruct.MatrixColsGet[U](matrixBAddr)
	cRows := memstruct.MatrixRowsGet[float64](matrixCAddr)
	cCols := memstruct.MatrixColsGet[float64](matrixCAddr)

	// Validate dimensions: A (m×n) × B (n×p) → C (m×p)
	if aCols != bRows {
		panic(fmt.Errorf(
			"BlazeElementWiseMatrixMultiplyF64: dimension mismatch (A.cols=%d != B.rows=%d)",
			aCols, bRows,
		))
	}
	if cRows != aRows || cCols != bCols {
		panic(fmt.Errorf(
			"BlazeElementWiseMatrixMultiplyF64: output dimension mismatch (C=%dx%d, expected=%dx%d)",
			cRows, cCols, aRows, bCols,
		))
	}

	// Standard matrix multiplication: C[i,j] = Σ(A[i,k] * B[k,j])
	for i := uint64(0); i < aRows; i++ {
		for j := uint64(0); j < bCols; j++ {
			var sum float64 = 0
			for k := uint64(0); k < aCols; k++ {
				aVal := memstruct.MatrixItemGetAtUnsafe[T](matrixAAddr, i, k)
				bVal := memstruct.MatrixItemGetAtUnsafe[U](matrixBAddr, k, j)
				sum += float64(aVal) * float64(bVal)
			}
			memstruct.MatrixSetAtUnsafe(matrixCAddr, i, j, sum)
		}
	}
}

// BlazeElementWiseMatrixDivideF32 divides the values of Matrix A by Matrix B, resulting in Matrix C at newMatrixAddr.
// It does so in float32 precision.
func BlazeElementWiseMatrixDivideF32[T, U foundation.Numeric](
	matrixAAddr, matrixBAddr, newMatrixAddr memcore.MarkRaw,
) {
	memstruct.MatrixBinaryExecute(matrixAAddr, matrixBAddr, newMatrixAddr, func(a T, b U) float32 {
		return float32(a) / float32(b)
	}, core.BlazeDefaultStride)
}

// BlazeElementWiseMatrixDivideF64 divides the values of Matrix A by Matrix B, resulting in Matrix C at newMatrixAddr.
// It does so in float64 precision.
func BlazeElementWiseMatrixDivideF64[T, U foundation.Numeric](
	matrixAAddr, matrixBAddr, newMatrixAddr memcore.MarkRaw,
) {
	memstruct.MatrixBinaryExecute(matrixAAddr, matrixBAddr, newMatrixAddr, func(a T, b U) float64 {
		return float64(a) / float64(b)
	}, core.BlazeDefaultStride)
}

// BlazeElementWiseMatrixMultiplyVectorF32 multiplies each row of Matrix A element-wise with Vector B,
// then sums the results, producing Vector C at newVectorAddr.
// This is standard matrix-vector multiplication: result[i] = sum(matrix[i,j] * vector[j] for j in 0..cols)
// It does so in float32 precision.
func BlazeElementWiseMatrixMultiplyVectorF32[T, U foundation.Numeric](
	matrixAddr, vectorAddr, newVectorAddr memcore.MarkRaw,
) error {
	rows := memstruct.MatrixRowsGet[T](matrixAddr)
	cols := memstruct.MatrixColsGet[T](matrixAddr)
	vectorLength := memstruct.VectorCapacityGet[U](vectorAddr)
	outputLength := memstruct.VectorCapacityGet[float32](newVectorAddr)

	if cols != vectorLength {
		return fmt.Errorf("matrix column count (%d) must match vector capacity (%d)", cols, vectorLength)
	}

	if rows != outputLength {
		return fmt.Errorf("matrix row count (%d) must match output vector capacity (%d)", rows, outputLength)
	}

	// For each row in the matrix, compute dot product with the vector
	for row := uint64(0); row < rows; row++ {
		sum := float32(0)

		// Compute dot product: sum of matrix[row, col] * vector[col] for all cols
		for col := uint64(0); col < cols; col++ {
			matrixVal := memstruct.MatrixItemGetAtUnsafe[T](matrixAddr, row, col)
			vectorVal := memstruct.VectorItemGetAtUnsafe[U](vectorAddr, col)
			sum += float32(matrixVal) * float32(vectorVal)
		}

		// Store result in output vector
		memstruct.VectorSetAtUnsafe(newVectorAddr, row, sum)
	}

	return nil
}

// BlazeElementWiseMatrixMultiplyVectorF64 multiplies each row of Matrix A element-wise with Vector B,
// then sums the results, producing Vector C at newVectorAddr.
// This is standard matrix-vector multiplication: result[i] = sum(matrix[i,j] * vector[j] for j in 0..cols)
// It does so in float64 precision.
func BlazeElementWiseMatrixMultiplyVectorF64[T, U foundation.Numeric](
	matrixAddr, vectorAddr, newVectorAddr memcore.MarkRaw,
) error {
	rows := memstruct.MatrixRowsGet[T](matrixAddr)
	cols := memstruct.MatrixColsGet[T](matrixAddr)
	vectorLength := memstruct.VectorCapacityGet[U](vectorAddr)
	outputLength := memstruct.VectorCapacityGet[float64](newVectorAddr)

	if cols != vectorLength {
		return fmt.Errorf("matrix column count (%d) must match vector capacity (%d)", cols, vectorLength)
	}

	if rows != outputLength {
		return fmt.Errorf("matrix row count (%d) must match output vector capacity (%d)", rows, outputLength)
	}

	// For each row in the matrix, compute dot product with the vector
	for row := uint64(0); row < rows; row++ {
		sum := float64(0)

		// Compute dot product: sum of matrix[row, col] * vector[col] for all cols
		for col := uint64(0); col < cols; col++ {
			matrixVal := memstruct.MatrixItemGetAtUnsafe[T](matrixAddr, row, col)
			vectorVal := memstruct.VectorItemGetAtUnsafe[U](vectorAddr, col)
			sum += float64(matrixVal) * float64(vectorVal)
		}

		// Store result in output vector
		memstruct.VectorSetAtUnsafe(newVectorAddr, row, sum)
	}

	return nil
}
