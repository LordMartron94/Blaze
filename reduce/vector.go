package reduce

import (
	"blaze/core"
	"foundation"
	"memcore"
	"memstruct"
)

/*BlazeReduceVectorMin returns the lowest element within the vector.*/
func BlazeReduceVectorMin[T foundation.Numeric](vector memcore.MarkRaw) T {
	minValue := foundation.MaxValue[T]()

	memstruct.VectorUnaryReadOnlyExecute(vector, func(item T) {
		if item < minValue {
			minValue = item
		}
	}, core.BlazeDefaultStride)

	return minValue
}

/*BlazeReduceVectorMax returns the highest element within the vector.*/
func BlazeReduceVectorMax[T foundation.Numeric](vector memcore.MarkRaw) T {
	maxValue := foundation.MinValue[T]()

	memstruct.VectorUnaryReadOnlyExecute(vector, func(item T) {
		if item > maxValue {
			maxValue = item
		}
	}, core.BlazeDefaultStride)

	return maxValue
}

/* BlazeReduceVectorMinMax returns the lowest and highest elements within the vector. */
func BlazeReduceVectorMinMax[T foundation.Numeric](vector memcore.MarkRaw) (min T, max T) {
	minValue := foundation.MaxValue[T]()
	maxValue := foundation.MinValue[T]()

	memstruct.VectorUnaryReadOnlyExecute(vector, func(item T) {
		if item < minValue {
			minValue = item
		}

		if item > maxValue {
			maxValue = item
		}
	}, core.BlazeDefaultStride)

	return minValue, maxValue
}

/*BlazeReduceVectorSumF32 computes the linear sum of the vector’s elements in float32 precision.*/
func BlazeReduceVectorSumF32[T foundation.Numeric](vector memcore.MarkRaw) float32 {
	sum := float32(0)

	memstruct.VectorUnaryReadOnlyExecute(vector, func(a T) {
		sum += float32(a)
	}, core.BlazeDefaultStride)

	return sum
}

/*BlazeReduceVectorSumF64 computes the linear sum of the vector’s elements in float64 precision.*/
func BlazeReduceVectorSumF64[T foundation.Numeric](vector memcore.MarkRaw) float64 {
	sum := float64(0)

	memstruct.VectorUnaryReadOnlyExecute(vector, func(a T) {
		sum += float64(a)
	}, core.BlazeDefaultStride)

	return sum
}

/*BlazeReduceVectorSumSquaredF32 computes the sum of squares (used in magnitude calculation) in float32 precision.*/
func BlazeReduceVectorSumSquaredF32[T foundation.Numeric](vector memcore.MarkRaw) float32 {
	sqrSum := float32(0)

	memstruct.VectorUnaryReadOnlyExecute(vector, func(a T) {
		sqrSum += float32(a) * float32(a)
	}, core.BlazeDefaultStride)

	return sqrSum
}

/*BlazeReduceVectorSumSquaredF64 computes the sum of squares (used in magnitude calculation) in float64 precision.*/
func BlazeReduceVectorSumSquaredF64[T foundation.Numeric](vector memcore.MarkRaw) float64 {
	sqrSum := float64(0)

	memstruct.VectorUnaryReadOnlyExecute(vector, func(a T) {
		sqrSum += float64(a) * float64(a)
	}, core.BlazeDefaultStride)

	return sqrSum
}

/*BlazeReduceDotProductF32 computes the dot product between two vectors (float32 precision).*/
func BlazeReduceDotProductF32[T, U foundation.Numeric](aAddr, bAddr memcore.MarkRaw) float32 {
	dot := float32(0)
	memstruct.VectorBinaryReadOnlyExecute(aAddr, bAddr, func(a T, b U) {
		dot += float32(a) * float32(b)
	}, core.BlazeDefaultStride)
	return dot
}

/* BlazeReduceDotProductF64 computes the dot product between two vectors (float64 precision). */
func BlazeReduceDotProductF64[T, U foundation.Numeric](aAddr, bAddr memcore.MarkRaw) float64 {
	dot := float64(0)
	memstruct.VectorBinaryReadOnlyExecute(aAddr, bAddr, func(a T, b U) {
		dot += float64(a) * float64(b)
	}, core.BlazeDefaultStride)
	return dot
}

/* BlazeReduceVectorMeanF32 computes the arithmetic mean of the vector's elements in float32 precision.*/
func BlazeReduceVectorMeanF32[T foundation.Numeric](vector memcore.MarkRaw) float32 {
	sum := BlazeReduceVectorSumF32[T](vector)
	count := float32(memstruct.VectorCapacityGet[T](vector))
	return sum / count
}

/* BlazeReduceVectorMeanF64 computes the arithmetic mean of the vector's elements in float64 precision. */
func BlazeReduceVectorMeanF64[T foundation.Numeric](vector memcore.MarkRaw) float64 {
	sum := BlazeReduceVectorSumF64[T](vector)
	count := float64(memstruct.VectorCapacityGet[T](vector))
	return sum / count
}

/*BlazeReduceCovarianceF32 computes the covariance between two vectors in float32 precision.
Covariance measures the joint variability of two variables: Σ((a_i - meanA) * (b_i - meanB)).
This is a basic numeric operation suitable for Blaze's numeric primitive focus.*/
func BlazeReduceCovarianceF32[T, U foundation.Numeric](
	vectorA memcore.MarkRaw,
	vectorB memcore.MarkRaw,
	meanA float32,
	meanB float32,
) float32 {
	var covariance float32 = 0
	memstruct.VectorBinaryReadOnlyExecute(vectorA, vectorB, func(a T, b U) {
		devA := float32(a) - meanA
		devB := float32(b) - meanB
		covariance += devA * devB
	}, core.BlazeDefaultStride)
	return covariance
}

/*BlazeReduceCovarianceF64 computes the covariance between two vectors in float64 precision.
Covariance measures the joint variability of two variables: Σ((a_i - meanA) * (b_i - meanB)).
This is a basic numeric operation suitable for Blaze's numeric primitive focus.*/
func BlazeReduceCovarianceF64[T, U foundation.Numeric](
	vectorA memcore.MarkRaw,
	vectorB memcore.MarkRaw,
	meanA float64,
	meanB float64,
) float64 {
	var covariance float64 = 0
	memstruct.VectorBinaryReadOnlyExecute(vectorA, vectorB, func(a T, b U) {
		devA := float64(a) - meanA
		devB := float64(b) - meanB
		covariance += devA * devB
	}, core.BlazeDefaultStride)
	return covariance
}
