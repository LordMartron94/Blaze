package reduce

import (
	"blaze/core"
	"foundation"
	"memcore"
	"memstruct"
)

// BlazeReduceVectorMin returns the lowest element within the vector.
func BlazeReduceVectorMin[T foundation.Numeric](vector memcore.MarkRaw) T {
	minValue := foundation.MaxValue[T]()

	memstruct.VectorUnaryReadOnlyExecute(vector, func(item T) {
		if item < minValue {
			minValue = item
		}
	}, core.BlazeDefaultStride)

	return minValue
}

// BlazeReduceVectorMax returns the highest element within the vector.
func BlazeReduceVectorMax[T foundation.Numeric](vector memcore.MarkRaw) T {
	maxValue := foundation.MinValue[T]()

	memstruct.VectorUnaryReadOnlyExecute(vector, func(item T) {
		if item > maxValue {
			maxValue = item
		}
	}, core.BlazeDefaultStride)

	return maxValue
}

// BlazeReduceVectorSumF32 computes the linear sum of the vector’s elements in float32 precision.
func BlazeReduceVectorSumF32[T foundation.Numeric](vector memcore.MarkRaw) float32 {
	sum := float32(0)

	memstruct.VectorUnaryReadOnlyExecute(vector, func(a T) {
		sum += float32(a)
	}, core.BlazeDefaultStride)

	return sum
}

// BlazeReduceVectorSumF64 computes the linear sum of the vector’s elements in float64 precision.
func BlazeReduceVectorSumF64[T foundation.Numeric](vector memcore.MarkRaw) float64 {
	sum := float64(0)

	memstruct.VectorUnaryReadOnlyExecute(vector, func(a T) {
		sum += float64(a)
	}, core.BlazeDefaultStride)

	return sum
}

// BlazeReduceVectorSumSquaredF32 computes the sum of squares (used in magnitude calculation) in float32 precision.
func BlazeReduceVectorSumSquaredF32[T foundation.Numeric](vector memcore.MarkRaw) float32 {
	sqrSum := float32(0)

	memstruct.VectorUnaryReadOnlyExecute(vector, func(a T) {
		sqrSum += float32(a) * float32(a)
	}, core.BlazeDefaultStride)

	return sqrSum
}

// BlazeReduceVectorSumSquaredF64 computes the sum of squares (used in magnitude calculation) in float64 precision.
func BlazeReduceVectorSumSquaredF64[T foundation.Numeric](vector memcore.MarkRaw) float64 {
	sqrSum := float64(0)

	memstruct.VectorUnaryReadOnlyExecute(vector, func(a T) {
		sqrSum += float64(a) * float64(a)
	}, core.BlazeDefaultStride)

	return sqrSum
}

// BlazeReduceDotProductF32 computes the dot product between two vectors (float32 precision).
func BlazeReduceDotProductF32[T, U foundation.Numeric](aAddr, bAddr memcore.MarkRaw) float32 {
	dot := float32(0)
	memstruct.VectorBinaryReadOnlyExecute(aAddr, bAddr, func(a T, b U) {
		dot += float32(a) * float32(b)
	}, core.BlazeDefaultStride)
	return dot
}

// BlazeReduceDotProductF64 computes the dot product between two vectors (float64 precision).
func BlazeReduceDotProductF64[T, U foundation.Numeric](aAddr, bAddr memcore.MarkRaw) float64 {
	dot := float64(0)
	memstruct.VectorBinaryReadOnlyExecute(aAddr, bAddr, func(a T, b U) {
		dot += float64(a) * float64(b)
	}, core.BlazeDefaultStride)
	return dot
}
