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

// BlazeElementWiseMatrixMultiplyF32 multiplies the values of Matrix A by Matrix B, resulting in Matrix C at newMatrixAddr.
// It does so in float32 precision.
func BlazeElementWiseMatrixMultiplyF32[T, U foundation.Numeric](
	matrixAAddr, matrixBAddr, newMatrixAddr memcore.MarkRaw,
) {
	memstruct.MatrixBinaryExecute(matrixAAddr, matrixBAddr, newMatrixAddr, func(a T, b U) float32 {
		return float32(a) * float32(b)
	}, core.BlazeDefaultStride)
}

// BlazeElementWiseMatrixMultiplyF64 multiplies the values of Matrix A by Matrix B, resulting in Matrix C at newMatrixAddr.
// It does so in float64 precision.
func BlazeElementWiseMatrixMultiplyF64[T, U foundation.Numeric](
	matrixAAddr, matrixBAddr, newMatrixAddr memcore.MarkRaw,
) {
	memstruct.MatrixBinaryExecute(matrixAAddr, matrixBAddr, newMatrixAddr, func(a T, b U) float64 {
		return float64(a) * float64(b)
	}, core.BlazeDefaultStride)
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
