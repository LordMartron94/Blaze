package scalar

import (
	"blaze/core"
	"fmt"
	"foundation"
	"memcore"
	"memstruct"
)

// BlazeScalarVectorSetAllSequence sets all values in the Vector to:
//
//	value[i] = initial + (i * step)
//
// It panics if the resulting value exceeds the numeric type’s limits.
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

// BlazeScalarVectorMultiplyF32 multiplies each vector element by a scalar (float32 precision).
func BlazeScalarVectorMultiplyF32[T foundation.Numeric](
	currentVectorAddr, newVectorAddr memcore.MarkRaw,
	scalar float32,
) {
	memstruct.VectorUnaryExecute(currentVectorAddr, newVectorAddr, func(a T) float32 {
		return float32(a) * scalar
	}, core.BlazeDefaultStride)
}

// BlazeScalarVectorDivideF32 divides each vector element by a scalar (float32 precision).
func BlazeScalarVectorDivideF32[T foundation.Numeric](
	currentVectorAddr, newVectorAddr memcore.MarkRaw,
	scalar float32,
) {
	memstruct.VectorUnaryExecute(currentVectorAddr, newVectorAddr, func(a T) float32 {
		return float32(a) / scalar
	}, core.BlazeDefaultStride)
}

// BlazeScalarVectorAddF32 adds a scalar to each vector element (float32 precision).
func BlazeScalarVectorAddF32[T foundation.Numeric](
	currentVectorAddr, newVectorAddr memcore.MarkRaw,
	scalar float32,
) {
	memstruct.VectorUnaryExecute(currentVectorAddr, newVectorAddr, func(a T) float32 {
		return float32(a) + scalar
	}, core.BlazeDefaultStride)
}

// BlazeScalarVectorSubtractF32 subtracts a scalar from each vector element (float32 precision).
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

// BlazeScalarVectorMultiplyF64 multiplies each vector element by a scalar (float64 precision).
func BlazeScalarVectorMultiplyF64[T foundation.Numeric](
	currentVectorAddr, newVectorAddr memcore.MarkRaw,
	scalar float64,
) {
	memstruct.VectorUnaryExecute(currentVectorAddr, newVectorAddr, func(a T) float64 {
		return float64(a) * scalar
	}, core.BlazeDefaultStride)
}

// BlazeScalarVectorDivideF64 divides each vector element by a scalar (float64 precision).
func BlazeScalarVectorDivideF64[T foundation.Numeric](
	currentVectorAddr, newVectorAddr memcore.MarkRaw,
	scalar float64,
) {
	memstruct.VectorUnaryExecute(currentVectorAddr, newVectorAddr, func(a T) float64 {
		return float64(a) / scalar
	}, core.BlazeDefaultStride)
}

// BlazeScalarVectorAddF64 adds a scalar to each vector element (float64 precision).
func BlazeScalarVectorAddF64[T foundation.Numeric](
	currentVectorAddr, newVectorAddr memcore.MarkRaw,
	scalar float64,
) {
	memstruct.VectorUnaryExecute(currentVectorAddr, newVectorAddr, func(a T) float64 {
		return float64(a) + scalar
	}, core.BlazeDefaultStride)
}

// BlazeScalarVectorSubtractF64 subtracts a scalar from each vector element (float64 precision).
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

// BlazeScalarVectorClamp clamps each vector element so it lies within [min, max].
//
// This operation mutates the existing vector in place.
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
