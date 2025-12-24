package compare

import (
	"blaze/core"
	"fmt"
	"foundation"
	"math"
	"memcore"
	"memstruct"
	"sort"
)

/*
BlazeCompareVectorGreaterThanOrEqualTo produces a boolean mask indicating whether each element
in Vector A is greater than or equal to the corresponding element in Vector B.

For each element i: mask[i] = (vectorA[i] >= vectorB[i])

The output mask must be of type Array[bool] with the same capacity as the input vectors.

Use cases:
- Filtering data (threshold comparisons)
- Conditional operations (masking)
- Data validation (range checks)
- Signal processing (threshold detection)

Time complexity: O(n) - single pass through both vectors
Space complexity: O(1) - only accumulator variables used

Prerequisites:
- Both input vectors must be valid and initialized
- Both input vectors must have the same length
- Mask array must be of type Array[bool] with same capacity as input vectors
- Capacity equality must be guaranteed by the caller

Edge cases:
- Works with any numeric type (int, float32, float64, etc.)
- Type conversions occur during comparison (input types → float64)
- Comparisons are performed using float64 precision for consistency
*/
func BlazeCompareVectorGreaterThanOrEqualTo[T, U foundation.Numeric](
	vectorAAddr, vectorBAddr, maskAddr memcore.MarkRaw,
) {
	i := uint64(0)
	memstruct.VectorBinaryReadOnlyExecute(vectorAAddr, vectorBAddr, func(a T, b U) {
		if float64(a) >= float64(b) {
			memstruct.ArraySetAtUnsafe(maskAddr, i, true)
		} else {
			memstruct.ArraySetAtUnsafe(maskAddr, i, false)
		}

		i++
	}, core.BlazeDefaultStride)
}

/*
BlazeCompareVectorGreaterThan produces a boolean mask indicating whether each element
in Vector A is greater than the corresponding element in Vector B.

For each element i: mask[i] = (vectorA[i] > vectorB[i])

The output mask must be of type Array[bool] with the same capacity as the input vectors.

Use cases:
- Filtering data (strict threshold comparisons)
- Conditional operations (masking)
- Data validation (strict range checks)
- Signal processing (strict threshold detection)

Time complexity: O(n) - single pass through both vectors
Space complexity: O(1) - only accumulator variables used

Prerequisites:
- Both input vectors must be valid and initialized
- Both input vectors must have the same length
- Mask array must be of type Array[bool] with same capacity as input vectors
- Capacity equality must be guaranteed by the caller

Edge cases:
- Works with any numeric type (int, float32, float64, etc.)
- Type conversions occur during comparison (input types → float64)
- Comparisons are performed using float64 precision for consistency
*/
func BlazeCompareVectorGreaterThan[T, U foundation.Numeric](
	vectorAAddr, vectorBAddr, maskAddr memcore.MarkRaw,
) {
	i := uint64(0)
	memstruct.VectorBinaryReadOnlyExecute(vectorAAddr, vectorBAddr, func(a T, b U) {
		if float64(a) > float64(b) {
			memstruct.ArraySetAtUnsafe(maskAddr, i, true)
		} else {
			memstruct.ArraySetAtUnsafe(maskAddr, i, false)
		}

		i++
	}, core.BlazeDefaultStride)
}

/*
BlazeCompareVectorSmallerThanOrEqualTo produces a boolean mask indicating whether each element
in Vector A is less than or equal to the corresponding element in Vector B.

For each element i: mask[i] = (vectorA[i] <= vectorB[i])

The output mask must be of type Array[bool] with the same capacity as the input vectors.

Use cases:
- Filtering data (threshold comparisons)
- Conditional operations (masking)
- Data validation (range checks)
- Signal processing (threshold detection)

Time complexity: O(n) - single pass through both vectors
Space complexity: O(1) - only accumulator variables used

Prerequisites:
- Both input vectors must be valid and initialized
- Both input vectors must have the same length
- Mask array must be of type Array[bool] with same capacity as input vectors
- Capacity equality must be guaranteed by the caller

Edge cases:
- Works with any numeric type (int, float32, float64, etc.)
- Type conversions occur during comparison (input types → float64)
- Comparisons are performed using float64 precision for consistency
*/
func BlazeCompareVectorSmallerThanOrEqualTo[T, U foundation.Numeric](
	vectorAAddr, vectorBAddr, maskAddr memcore.MarkRaw,
) {
	i := uint64(0)
	memstruct.VectorBinaryReadOnlyExecute(vectorAAddr, vectorBAddr, func(a T, b U) {
		if float64(a) <= float64(b) {
			memstruct.ArraySetAtUnsafe(maskAddr, i, true)
		} else {
			memstruct.ArraySetAtUnsafe(maskAddr, i, false)
		}

		i++
	}, core.BlazeDefaultStride)
}

/*
BlazeCompareVectorSmallerThan produces a boolean mask indicating whether each element
in Vector A is less than the corresponding element in Vector B.

For each element i: mask[i] = (vectorA[i] < vectorB[i])

The output mask must be of type Array[bool] with the same capacity as the input vectors.

Use cases:
- Filtering data (strict threshold comparisons)
- Conditional operations (masking)
- Data validation (strict range checks)
- Signal processing (strict threshold detection)

Time complexity: O(n) - single pass through both vectors
Space complexity: O(1) - only accumulator variables used

Prerequisites:
- Both input vectors must be valid and initialized
- Both input vectors must have the same length
- Mask array must be of type Array[bool] with same capacity as input vectors
- Capacity equality must be guaranteed by the caller

Edge cases:
- Works with any numeric type (int, float32, float64, etc.)
- Type conversions occur during comparison (input types → float64)
- Comparisons are performed using float64 precision for consistency
*/
func BlazeCompareVectorSmallerThan[T, U foundation.Numeric](
	vectorAAddr, vectorBAddr, maskAddr memcore.MarkRaw,
) {
	i := uint64(0)
	memstruct.VectorBinaryReadOnlyExecute(vectorAAddr, vectorBAddr, func(a T, b U) {
		if float64(a) < float64(b) {
			memstruct.ArraySetAtUnsafe(maskAddr, i, true)
		} else {
			memstruct.ArraySetAtUnsafe(maskAddr, i, false)
		}

		i++
	}, core.BlazeDefaultStride)
}

/*
BlazeCompareVectorEqualTo produces a boolean mask indicating whether each element
in Vector A is equal to the corresponding element in Vector B within a specified tolerance.

For each element i: mask[i] = (|vectorA[i] - vectorB[i]| <= tolerance)

This function uses a tolerance-based comparison, which is essential for floating-point
comparisons where exact equality is rarely achieved due to numerical precision.

The output mask must be of type Array[bool] with the same capacity as the input vectors.

Use cases:
- Floating-point equality checks (with tolerance)
- Data validation (approximate matching)
- Signal processing (matching signals within tolerance)
- Numerical stability checks

Time complexity: O(n) - single pass through both vectors
Space complexity: O(1) - only accumulator variables used

Prerequisites:
- Both input vectors must be valid and initialized
- Both input vectors must have the same length
- Mask array must be of type Array[bool] with same capacity as input vectors
- Capacity equality must be guaranteed by the caller
- tolerance should be chosen appropriately for the data scale (e.g., 1e-9 for float64)

Edge cases:
- Works with any numeric type (int, float32, float64, etc.)
- Type conversions occur during comparison (input types → float64)
- Tolerance-based comparison handles floating-point precision issues
- For integer types, tolerance of 0.0 provides exact equality
*/
func BlazeCompareVectorEqualTo[T, U foundation.Numeric](
	vectorAAddr, vectorBAddr, maskAddr memcore.MarkRaw,
	tolerance float64,
) {
	i := uint64(0)
	memstruct.VectorBinaryReadOnlyExecute(vectorAAddr, vectorBAddr, func(a T, b U) {
		equalEnough := math.Abs(float64(a)-float64(b)) <= tolerance
		if equalEnough {
			memstruct.ArraySetAtUnsafe(maskAddr, i, true)
		} else {
			memstruct.ArraySetAtUnsafe(maskAddr, i, false)
		}

		i++
	}, core.BlazeDefaultStride)
}

type weightedValue struct {
	id     float64
	weight float64
}

/*
BlazeCompareVectorSparseJaccardWeightedSimilarity computes the weighted Jaccard similarity
between two sparse vectors.

The weighted Jaccard similarity is defined as:
similarity = Σ(min(weightA[i], weightB[i])) / Σ(max(weightA[i], weightB[i]))
where the sums are over matching IDs (within tolerance).

Both vectors contain IDs, and both weight arrays contain a per-ID weight.
The function assumes that every index inside vector capacity is used.

Use cases:
- Sparse vector similarity (document similarity, feature matching)
- Recommendation systems (user/item similarity)
- Information retrieval (query matching)
- Set similarity with weights

Time complexity: O(n log n) - dominated by sorting both vectors by ID
Space complexity: O(n) - temporary slices for sorting

Prerequisites:
- Both input vectors must be valid and initialized
- vectorAWeights must have length equal to vectorA capacity
- vectorBWeights must have length equal to vectorB capacity
- tolerance determines ID matching threshold

Edge cases:
- Returns 0.0 if denominator is zero (no weights)
- Works with any numeric type for IDs (int, float32, float64, etc.)
- Weights are taken as absolute values (negative weights become positive)
- IDs are matched within tolerance (useful for floating-point IDs)
- Returns 0.0 if no IDs match

Algorithm:
1. Build temporary slices of (ID, weight) pairs for both vectors
2. Sort both lists by ID (ascending)
3. Merge walk to compute weighted Jaccard: sum min weights for matches, sum max weights for all
4. Return numerator / denominator
*/
func BlazeCompareVectorSparseJaccardWeightedSimilarity[T, U foundation.Numeric](
	vectorAAddr, vectorBAddr memcore.MarkRaw,
	vectorAWeights, vectorBWeights []float64,
	tolerance float64,
) float64 {

	vectorACapacity := memstruct.VectorCapacityGet[T](vectorAAddr)
	vectorBCapacity := memstruct.VectorCapacityGet[U](vectorBAddr)

	if uint64(len(vectorAWeights)) != vectorACapacity {
		panic(fmt.Errorf("cannot compute jaccard weighted similarity: reason=[a_lengthmismatch:vector=%d,capacity=%d]",
			vectorACapacity, len(vectorAWeights)))
	}

	if uint64(len(vectorBWeights)) != vectorBCapacity {
		panic(fmt.Errorf("cannot compute jaccard weighted similarity: reason=[b_lengthmismatch:vector=%d,capacity=%d]",
			vectorBCapacity, len(vectorBWeights)))
	}

	// ------------------------------------------------------------
	// 1. Build temporary slices of (sortedID, weight) pairs
	// ------------------------------------------------------------
	tmpA := make([]weightedValue, vectorACapacity)
	tmpB := make([]weightedValue, vectorBCapacity)

	for i := uint64(0); i < vectorACapacity; i++ {
		id := float64(memstruct.VectorItemGetAtUnsafe[T](vectorAAddr, i))
		tmpA[i] = weightedValue{
			id:     id,
			weight: math.Abs(vectorAWeights[i]),
		}
	}

	for j := uint64(0); j < vectorBCapacity; j++ {
		id := float64(memstruct.VectorItemGetAtUnsafe[U](vectorBAddr, j))
		tmpB[j] = weightedValue{
			id:     id,
			weight: math.Abs(vectorBWeights[j]),
		}
	}

	// ------------------------------------------------------------
	// 2. Sort both weighted lists by increasing ID
	// ------------------------------------------------------------
	sort.Slice(tmpA, func(i, j int) bool { return tmpA[i].id < tmpA[j].id })
	sort.Slice(tmpB, func(i, j int) bool { return tmpB[i].id < tmpB[j].id })

	// ------------------------------------------------------------
	// 3. Merge walk to compute weighted Jaccard
	// ------------------------------------------------------------
	i, j := 0, 0
	numerator := 0.0
	denominator := 0.0

	for i < len(tmpA) && j < len(tmpB) {
		a := tmpA[i]
		b := tmpB[j]

		equalEnough := math.Abs(a.id-b.id) <= tolerance

		if equalEnough {
			numerator += math.Min(a.weight, b.weight)
			denominator += math.Max(a.weight, b.weight)
			i++
			j++
		} else if a.id < b.id {
			denominator += a.weight
			i++
		} else {
			denominator += b.weight
			j++
		}
	}

	// ------------------------------------------------------------
	// 4. Add remaining tail of A or B to the denominator
	// ------------------------------------------------------------
	for i < len(tmpA) {
		denominator += tmpA[i].weight
		i++
	}
	for j < len(tmpB) {
		denominator += tmpB[j].weight
		j++
	}

	if denominator == 0 {
		return 0
	}

	return numerator / denominator
}
