package scalar

import (
	"blaze/core"
	"fmt"
	"foundation"
	"memcore"
	"memstruct"
)

// BlazeScalarVectorSetAllSequence sets all the values in the Vector to a value computed as:
// initial + (idx*step).
// It panics if the resulting value would be bigger than the numeric type.
func BlazeScalarVectorSetAllSequence[T foundation.Numeric](vector memcore.MarkRaw, initial, step T) {
	maxT := float64(foundation.MaxValue[T]())
	idx := 0
	memstruct.VectorUnaryExecute(vector, vector, func(_ T) T {
		v := float64(initial) + (float64(idx) * float64(step))

		if v > maxT {
			panic(fmt.Errorf("cannot set value of idx %d: would result in overflow: requested=%f,max=%f", idx, v, maxT))
		}

		idx++

		return T(v)
	}, core.BlazeDefaultStride)
}

// BlazeScalarVectorMultiplyF32 multiplies the values of the Vector by the scalar in float32 precision.
func BlazeScalarVectorMultiplyF32[T, S foundation.Numeric](
	currentVectorAddr, newVectorAddr memcore.MarkRaw,
	scalar S,
) {
	memstruct.VectorUnaryExecute(currentVectorAddr, newVectorAddr, func(a T) float32 {
		return float32(a) * float32(scalar)
	}, core.BlazeDefaultStride)
}

// BlazeScalarVectorMultiplyF64 multiplies the values of the Vector by the scalar in float64 precision.
func BlazeScalarVectorMultiplyF64[T, S foundation.Numeric](
	currentVectorAddr, newVectorAddr memcore.MarkRaw,
	scalar S,
) {
	memstruct.VectorUnaryExecute(currentVectorAddr, newVectorAddr, func(a T) float64 {
		return float64(a) * float64(scalar)
	}, core.BlazeDefaultStride)
}

// BlazeScalarVectorDivideF32 divides the values of the Vector by the scalar in float32 precision.
func BlazeScalarVectorDivideF32[T, S foundation.Numeric](
	currentVectorAddr, newVectorAddr memcore.MarkRaw,
	scalar S,
) {
	memstruct.VectorUnaryExecute(currentVectorAddr, newVectorAddr, func(a T) float32 {
		return float32(a) / float32(scalar)
	}, core.BlazeDefaultStride)
}

// BlazeScalarVectorDivideF64 divides the values of the Vector by the scalar in float64 precision.
func BlazeScalarVectorDivideF64[T, S foundation.Numeric](
	currentVectorAddr, newVectorAddr memcore.MarkRaw,
	scalar S,
) {
	memstruct.VectorUnaryExecute(currentVectorAddr, newVectorAddr, func(a T) float64 {
		return float64(a) / float64(scalar)
	}, core.BlazeDefaultStride)
}

// BlazeScalarVectorAddF32 adds the values of the Vector by the scalar in float32 precision.
func BlazeScalarVectorAddF32[T, S foundation.Numeric](
	currentVectorAddr, newVectorAddr memcore.MarkRaw,
	scalar S,
) {
	memstruct.VectorUnaryExecute(currentVectorAddr, newVectorAddr, func(a T) float32 {
		return float32(a) + float32(scalar)
	}, core.BlazeDefaultStride)
}

// BlazeScalarVectorAddF64 adds the values of the Vector by the scalar in float64 precision.
func BlazeScalarVectorAddF64[T, S foundation.Numeric](
	currentVectorAddr, newVectorAddr memcore.MarkRaw,
	scalar S,
) {
	memstruct.VectorUnaryExecute(currentVectorAddr, newVectorAddr, func(a T) float64 {
		return float64(a) + float64(scalar)
	}, core.BlazeDefaultStride)
}

// BlazeScalarVectorSubtractF32 subtracts the scalar from the values in the Vector in float32 precision.
func BlazeScalarVectorSubtractF32[T, S foundation.Numeric](
	currentVectorAddr, newVectorAddr memcore.MarkRaw,
	scalar S,
) {
	memstruct.VectorUnaryExecute(currentVectorAddr, newVectorAddr, func(a T) float32 {
		return float32(a) - float32(scalar)
	}, core.BlazeDefaultStride)
}

// BlazeScalarVectorSubtractF64 subtracts the scalar from the values in the Vector in float64 precision.
func BlazeScalarVectorSubtractF64[T, S foundation.Numeric](
	currentVectorAddr, newVectorAddr memcore.MarkRaw,
	scalar S,
) {
	memstruct.VectorUnaryExecute(currentVectorAddr, newVectorAddr, func(a T) float64 {
		return float64(a) - float64(scalar)
	}, core.BlazeDefaultStride)
}

// BlazeScalarVectorClamp transforms the vector in such a way that each element is between
// min and max.
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
