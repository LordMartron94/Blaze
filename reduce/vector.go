package reduce

import (
	"blaze/core"
	"foundation"
	"memcore"
	"memstruct"
)

/*
BlazeReduceVectorMin finds the minimum (lowest) element in a vector.

Returns the smallest value of type T found in the vector.

Use cases:
- Finding minimum values in datasets
- Range calculations (combined with max)
- Data validation (checking bounds)
- Optimization problems

Time complexity: O(n) - single pass through vector
Space complexity: O(1) - only accumulator variable used

Prerequisites:
- Vector must be valid and initialized
- Vector must have at least one element (undefined behavior for empty vectors)

Edge cases:
- Returns MaxValue[T]() if vector is empty (initial value)
- Works with any numeric type (int, float32, float64, etc.)
- No type conversions (returns same type as input)
*/
func BlazeReduceVectorMin[T foundation.Numeric](vector memcore.MarkRaw) T {
	minValue := foundation.MaxValue[T]()

	memstruct.VectorUnaryReadOnlyExecute(vector, func(item T) {
		if item < minValue {
			minValue = item
		}
	}, core.BlazeDefaultStride)

	return minValue
}

/*
BlazeReduceVectorMax finds the maximum (highest) element in a vector.

Returns the largest value of type T found in the vector.

Use cases:
- Finding maximum values in datasets
- Range calculations (combined with min)
- Data validation (checking bounds)
- Optimization problems

Time complexity: O(n) - single pass through vector
Space complexity: O(1) - only accumulator variable used

Prerequisites:
- Vector must be valid and initialized
- Vector must have at least one element (undefined behavior for empty vectors)

Edge cases:
- Returns MinValue[T]() if vector is empty (initial value)
- Works with any numeric type (int, float32, float64, etc.)
- No type conversions (returns same type as input)
*/
func BlazeReduceVectorMax[T foundation.Numeric](vector memcore.MarkRaw) T {
	maxValue := foundation.MinValue[T]()

	memstruct.VectorUnaryReadOnlyExecute(vector, func(item T) {
		if item > maxValue {
			maxValue = item
		}
	}, core.BlazeDefaultStride)

	return maxValue
}

/*
BlazeReduceVectorMinMax finds both the minimum and maximum elements in a vector in a single pass.

Returns both the smallest and largest values of type T found in the vector.
This is more efficient than calling BlazeReduceVectorMin and BlazeReduceVectorMax separately.

Use cases:
- Range calculations (computing data spread)
- Data validation (checking bounds)
- Normalization preprocessing (determining scale)
- Optimization problems

Time complexity: O(n) - single pass through vector
Space complexity: O(1) - only accumulator variables used

Prerequisites:
- Vector must be valid and initialized
- Vector must have at least one element (undefined behavior for empty vectors)

Edge cases:
- Returns (MaxValue[T](), MinValue[T]()) if vector is empty (initial values)
- Works with any numeric type (int, float32, float64, etc.)
- No type conversions (returns same type as input)
*/
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

/*
BlazeReduceVectorSumF32 computes the sum of all vector elements in float32 precision.

Computes: sum = Σ(vector[i]) for all elements i.

Use cases:
- Aggregating data (total, cumulative sums)
- Statistical computations (mean calculation)
- Numerical integration (Riemann sums)
- Signal processing (accumulation)

Time complexity: O(n) - single pass through vector
Space complexity: O(1) - only accumulator variable used

Prerequisites:
- Vector must be valid and initialized

Edge cases:
- Returns 0.0 for empty vectors
- Works with any numeric type (int, float32, float64, etc.)
- Type conversions occur during computation (input types → float32)
- Potential precision loss for large integer sums
- No overflow checking (relies on Go's numeric behavior)
*/
func BlazeReduceVectorSumF32[T foundation.Numeric](vector memcore.MarkRaw) float32 {
	sum := float32(0)

	memstruct.VectorUnaryReadOnlyExecute(vector, func(a T) {
		sum += float32(a)
	}, core.BlazeDefaultStride)

	return sum
}

/*
BlazeReduceVectorSumF64 computes the sum of all vector elements in float64 precision.

Computes: sum = Σ(vector[i]) for all elements i.

Use cases:
- Aggregating data (total, cumulative sums)
- Statistical computations (mean calculation)
- Numerical integration (Riemann sums)
- Signal processing (accumulation)

Time complexity: O(n) - single pass through vector
Space complexity: O(1) - only accumulator variable used

Prerequisites:
- Vector must be valid and initialized

Edge cases:
- Returns 0.0 for empty vectors
- Works with any numeric type (int, float32, float64, etc.)
- Type conversions occur during computation (input types → float64)
- Better precision than F32 variant for large sums
- No overflow checking (relies on Go's numeric behavior)
*/
func BlazeReduceVectorSumF64[T foundation.Numeric](vector memcore.MarkRaw) float64 {
	sum := float64(0)

	memstruct.VectorUnaryReadOnlyExecute(vector, func(a T) {
		sum += float64(a)
	}, core.BlazeDefaultStride)

	return sum
}

/*
BlazeReduceVectorSumSquaredF32 computes the sum of squares of all vector elements in float32 precision.

Computes: sumSquared = Σ(vector[i]²) for all elements i.

Use cases:
- Magnitude calculations (L2 norm, Euclidean distance)
- Variance calculations (sum of squared deviations)
- Statistical computations (second moment)
- Signal processing (power calculations)

Time complexity: O(n) - single pass through vector
Space complexity: O(1) - only accumulator variable used

Prerequisites:
- Vector must be valid and initialized

Edge cases:
- Returns 0.0 for empty vectors
- Works with any numeric type (int, float32, float64, etc.)
- Type conversions occur during computation (input types → float32)
- Potential precision loss for large values
- No overflow checking (relies on Go's numeric behavior)
*/
func BlazeReduceVectorSumSquaredF32[T foundation.Numeric](vector memcore.MarkRaw) float32 {
	sqrSum := float32(0)

	memstruct.VectorUnaryReadOnlyExecute(vector, func(a T) {
		sqrSum += float32(a) * float32(a)
	}, core.BlazeDefaultStride)

	return sqrSum
}

/*
BlazeReduceVectorSumSquaredF64 computes the sum of squares of all vector elements in float64 precision.

Computes: sumSquared = Σ(vector[i]²) for all elements i.

Use cases:
- Magnitude calculations (L2 norm, Euclidean distance)
- Variance calculations (sum of squared deviations)
- Statistical computations (second moment)
- Signal processing (power calculations)

Time complexity: O(n) - single pass through vector
Space complexity: O(1) - only accumulator variable used

Prerequisites:
- Vector must be valid and initialized

Edge cases:
- Returns 0.0 for empty vectors
- Works with any numeric type (int, float32, float64, etc.)
- Type conversions occur during computation (input types → float64)
- Better precision than F32 variant for large values
- No overflow checking (relies on Go's numeric behavior)
*/
func BlazeReduceVectorSumSquaredF64[T foundation.Numeric](vector memcore.MarkRaw) float64 {
	sqrSum := float64(0)

	memstruct.VectorUnaryReadOnlyExecute(vector, func(a T) {
		sqrSum += float64(a) * float64(a)
	}, core.BlazeDefaultStride)

	return sqrSum
}

/*
BlazeReduceDotProductF32 computes the dot product (inner product) of two vectors in float32 precision.

Computes: dot = Σ(vectorA[i] * vectorB[i]) for all elements i.

The dot product measures the similarity and projection between two vectors.
It is fundamental to linear algebra, machine learning, and signal processing.

Use cases:
- Linear algebra operations (projections, angles)
- Machine learning (feature similarity, neural networks)
- Signal processing (correlation, filtering)
- Statistical computations (covariance, correlation)

Time complexity: O(n) - single pass through both vectors
Space complexity: O(1) - only accumulator variable used

Prerequisites:
- Both vectors must be valid and initialized
- Both vectors must have the same length

Edge cases:
- Returns 0.0 for empty vectors
- Works with any numeric type (int, float32, float64, etc.)
- Type conversions occur during computation (input types → float32)
- Potential precision loss for large values
- No overflow checking (relies on Go's numeric behavior)
*/
func BlazeReduceDotProductF32[T, U foundation.Numeric](aAddr, bAddr memcore.MarkRaw) float32 {
	dot := float32(0)
	memstruct.VectorBinaryReadOnlyExecute(aAddr, bAddr, func(a T, b U) {
		dot += float32(a) * float32(b)
	}, core.BlazeDefaultStride)
	return dot
}

/*
BlazeReduceDotProductF64 computes the dot product (inner product) of two vectors in float64 precision.

Computes: dot = Σ(vectorA[i] * vectorB[i]) for all elements i.

The dot product measures the similarity and projection between two vectors.
It is fundamental to linear algebra, machine learning, and signal processing.

Use cases:
- Linear algebra operations (projections, angles)
- Machine learning (feature similarity, neural networks)
- Signal processing (correlation, filtering)
- Statistical computations (covariance, correlation)

Time complexity: O(n) - single pass through both vectors
Space complexity: O(1) - only accumulator variable used

Prerequisites:
- Both vectors must be valid and initialized
- Both vectors must have the same length

Edge cases:
- Returns 0.0 for empty vectors
- Works with any numeric type (int, float32, float64, etc.)
- Type conversions occur during computation (input types → float64)
- Better precision than F32 variant for large values
- No overflow checking (relies on Go's numeric behavior)
*/
func BlazeReduceDotProductF64[T, U foundation.Numeric](aAddr, bAddr memcore.MarkRaw) float64 {
	dot := float64(0)
	memstruct.VectorBinaryReadOnlyExecute(aAddr, bAddr, func(a T, b U) {
		dot += float64(a) * float64(b)
	}, core.BlazeDefaultStride)
	return dot
}

/*
BlazeReduceVectorMeanF32 computes the arithmetic mean (average) of all vector elements in float32 precision.

Computes: mean = (Σ(vector[i])) / n where n is the vector length.

The arithmetic mean is the sum of all elements divided by the count.
This is a fundamental statistical measure of central tendency.

Use cases:
- Statistical analysis (central tendency)
- Data aggregation (averaging)
- Signal processing (DC component)
- Machine learning (feature normalization)

Time complexity: O(n) - single pass through vector (uses BlazeReduceVectorSumF32)
Space complexity: O(1) - only accumulator variables used

Prerequisites:
- Vector must be valid and initialized
- Vector must have at least one element

Edge cases:
- Panics if vector is empty (division by zero)
- Works with any numeric type (int, float32, float64, etc.)
- Type conversions occur during computation (input types → float32)
- Potential precision loss for large integer means
*/
func BlazeReduceVectorMeanF32[T foundation.Numeric](vector memcore.MarkRaw) float32 {
	sum := BlazeReduceVectorSumF32[T](vector)
	count := float32(memstruct.VectorCapacityGet[T](vector))
	return sum / count
}

/*
BlazeReduceVectorMeanF64 computes the arithmetic mean (average) of all vector elements in float64 precision.

Computes: mean = (Σ(vector[i])) / n where n is the vector length.

The arithmetic mean is the sum of all elements divided by the count.
This is a fundamental statistical measure of central tendency.

Use cases:
- Statistical analysis (central tendency)
- Data aggregation (averaging)
- Signal processing (DC component)
- Machine learning (feature normalization)

Time complexity: O(n) - single pass through vector (uses BlazeReduceVectorSumF64)
Space complexity: O(1) - only accumulator variables used

Prerequisites:
- Vector must be valid and initialized
- Vector must have at least one element

Edge cases:
- Panics if vector is empty (division by zero)
- Works with any numeric type (int, float32, float64, etc.)
- Type conversions occur during computation (input types → float64)
- Better precision than F32 variant for large values
*/
func BlazeReduceVectorMeanF64[T foundation.Numeric](vector memcore.MarkRaw) float64 {
	sum := BlazeReduceVectorSumF64[T](vector)
	count := float64(memstruct.VectorCapacityGet[T](vector))
	return sum / count
}

/*
BlazeReduceCovarianceF32 computes the covariance between two vectors in float32 precision.

Computes: covariance = Σ((vectorA[i] - meanA) * (vectorB[i] - meanB)) for all elements i.

Covariance measures the joint variability of two variables. It indicates the direction
of the linear relationship between variables. This function requires pre-computed means
for both vectors (use BlazeReduceVectorMeanF32).

This is a basic numeric operation suitable for Blaze's numeric primitive focus.
For complete statistical analysis, see statarch/correlation.

Use cases:
- Statistical analysis (relationship between variables)
- Machine learning (feature relationships)
- Signal processing (cross-correlation)
- Financial analysis (asset relationships)

Time complexity: O(n) - single pass through both vectors
Space complexity: O(1) - only accumulator variable used

Prerequisites:
- Both vectors must be valid and initialized
- Both vectors must have the same length
- meanA and meanB must be pre-computed (use BlazeReduceVectorMeanF32)

Edge cases:
- Returns 0.0 for empty vectors
- Works with any numeric type (int, float32, float64, etc.)
- Type conversions occur during computation (input types → float32)
- Potential precision loss for large values
- No overflow checking (relies on Go's numeric behavior)
*/
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

/*
BlazeReduceCovarianceF64 computes the covariance between two vectors in float64 precision.

Computes: covariance = Σ((vectorA[i] - meanA) * (vectorB[i] - meanB)) for all elements i.

Covariance measures the joint variability of two variables. It indicates the direction
of the linear relationship between variables. This function requires pre-computed means
for both vectors (use BlazeReduceVectorMeanF64).

This is a basic numeric operation suitable for Blaze's numeric primitive focus.
For complete statistical analysis, see statarch/correlation.

Use cases:
- Statistical analysis (relationship between variables)
- Machine learning (feature relationships)
- Signal processing (cross-correlation)
- Financial analysis (asset relationships)

Time complexity: O(n) - single pass through both vectors
Space complexity: O(1) - only accumulator variable used

Prerequisites:
- Both vectors must be valid and initialized
- Both vectors must have the same length
- meanA and meanB must be pre-computed (use BlazeReduceVectorMeanF64)

Edge cases:
- Returns 0.0 for empty vectors
- Works with any numeric type (int, float32, float64, etc.)
- Type conversions occur during computation (input types → float64)
- Better precision than F32 variant for large values
- No overflow checking (relies on Go's numeric behavior)
*/
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
