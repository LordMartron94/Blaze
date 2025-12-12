package scalar

import (
	"blaze/core"
	"fmt"
	"foundation"
	"memcore"
	"memstruct"
)

// BlazeScalarMatrixSetAllSequence sets all values in the Matrix to:
//
//	value[i] = initial + (i * step)
//
// It panics if the resulting value exceeds the numeric type’s limits.
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

// BlazeScalarMatrixMultiplyF32 multiplies each matrix element by a scalar (float32 precision).
func BlazeScalarMatrixMultiplyF32[T foundation.Numeric](
	currentMatrixAddr, newMatrixAddr memcore.MarkRaw,
	scalar float32,
) {
	memstruct.MatrixUnaryExecute(currentMatrixAddr, newMatrixAddr, func(a T) float32 {
		return float32(a) * scalar
	}, core.BlazeDefaultStride)
}

// BlazeScalarMatrixDivideF32 divides each matrix element by a scalar (float32 precision).
func BlazeScalarMatrixDivideF32[T foundation.Numeric](
	currentMatrixAddr, newMatrixAddr memcore.MarkRaw,
	scalar float32,
) {
	memstruct.MatrixUnaryExecute(currentMatrixAddr, newMatrixAddr, func(a T) float32 {
		return float32(a) / scalar
	}, core.BlazeDefaultStride)
}

// BlazeScalarMatrixAddF32 adds a scalar to each matrix element (float32 precision).
func BlazeScalarMatrixAddF32[T foundation.Numeric](
	currentMatrixAddr, newMatrixAddr memcore.MarkRaw,
	scalar float32,
) {
	memstruct.MatrixUnaryExecute(currentMatrixAddr, newMatrixAddr, func(a T) float32 {
		return float32(a) + scalar
	}, core.BlazeDefaultStride)
}

// BlazeScalarMatrixSubtractF32 subtracts a scalar from each matrix element (float32 precision).
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

// BlazeScalarMatrixMultiplyF64 multiplies each matrix element by a scalar (float64 precision).
func BlazeScalarMatrixMultiplyF64[T foundation.Numeric](
	currentMatrixAddr, newMatrixAddr memcore.MarkRaw,
	scalar float64,
) {
	memstruct.MatrixUnaryExecute(currentMatrixAddr, newMatrixAddr, func(a T) float64 {
		return float64(a) * scalar
	}, core.BlazeDefaultStride)
}

// BlazeScalarMatrixDivideF64 divides each matrix element by a scalar (float64 precision).
func BlazeScalarMatrixDivideF64[T foundation.Numeric](
	currentMatrixAddr, newMatrixAddr memcore.MarkRaw,
	scalar float64,
) {
	memstruct.MatrixUnaryExecute(currentMatrixAddr, newMatrixAddr, func(a T) float64 {
		return float64(a) / scalar
	}, core.BlazeDefaultStride)
}

// BlazeScalarMatrixAddF64 adds a scalar to each matrix element (float64 precision).
func BlazeScalarMatrixAddF64[T foundation.Numeric](
	currentMatrixAddr, newMatrixAddr memcore.MarkRaw,
	scalar float64,
) {
	memstruct.MatrixUnaryExecute(currentMatrixAddr, newMatrixAddr, func(a T) float64 {
		return float64(a) + scalar
	}, core.BlazeDefaultStride)
}

// BlazeScalarMatrixSubtractF64 subtracts a scalar from each matrix element (float64 precision).
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

// BlazeScalarMatrixClamp clamps each matrix element so it lies within [min, max].
//
// This operation mutates the existing matrix in place.
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
