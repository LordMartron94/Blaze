package metric

import (
	"blaze/core"
	"blaze/reduce"
	"foundation"
	"memcore"
	"memstruct"
)

// BlazeMetricVectorMagnitudeF32 computes the magnitude of a given vector in float32 precision.
func BlazeMetricVectorMagnitudeF32[T foundation.Numeric](vector memcore.MarkRaw) float32 {
	sqrSum := reduce.BlazeReduceVectorSumSquaredF32[T](vector)
	return foundation.Sqrt32(sqrSum)
}

// BlazeMetricVectorMagnitudeF64 computes the magnitude of a given vector in float64 precision.
func BlazeMetricVectorMagnitudeF64[T foundation.Numeric](vector memcore.MarkRaw) float64 {
	sqrSum := reduce.BlazeReduceVectorSumSquaredF64[T](vector)
	return foundation.Sqrt64(sqrSum)
}

// BlazeMetricVectorNormalizedF32 normalizes the vector (float32 precision) so its magnitude is 1.
func BlazeMetricVectorNormalizedF32[T foundation.Numeric](
	currentVector memcore.MarkRaw,
	newVectorAddr memcore.MarkRaw,
) {
	magnitude := BlazeMetricVectorMagnitudeF32[T](currentVector)
	inv := 1.0 / magnitude
	memstruct.VectorUnaryExecute(currentVector, newVectorAddr, func(a T) float32 {
		return float32(a) * inv
	}, core.BlazeDefaultStride)
}

// BlazeMetricVectorNormalizedF64 normalizes the vector (float64 precision) so its magnitude is 1.
func BlazeMetricVectorNormalizedF64[T foundation.Numeric](
	currentVector memcore.MarkRaw,
	newVectorAddr memcore.MarkRaw,
) {
	magnitude := BlazeMetricVectorMagnitudeF64[T](currentVector)
	inv := 1.0 / magnitude
	memstruct.VectorUnaryExecute(currentVector, newVectorAddr, func(a T) float64 {
		return float64(a) * inv
	}, core.BlazeDefaultStride)
}
