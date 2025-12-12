package elementwise

import (
	"blaze/core"
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
