package compare

import (
	"fmt"
	"math"
	"sort"
	"unsafe"

	"foundation"
	"memcore"
	"memstruct"
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
	aCapacity := memstruct.VectorCapacityGet[T](vectorAAddr)
	bCapacity := memstruct.VectorCapacityGet[U](vectorBAddr)

	if aCapacity != bCapacity {
		panic(fmt.Errorf("cannot perform compare: capacity mismatch (a=%d, b=%d)",
			aCapacity, bCapacity))
	}

	aData := memstruct.VectorDataPtrGet[T](vectorAAddr)
	bData := memstruct.VectorDataPtrGet[U](vectorBAddr)
	maskData := memstruct.ArrayDataPtrGet[bool](maskAddr)
	aSize := uintptr(memcore.SizeOf[T]())
	bSize := uintptr(memcore.SizeOf[U]())
	maskSize := uintptr(memcore.SizeOf[bool]())
	capacity := aCapacity

	var i uint64

	// Manually unrolled loop for stride 8
	for ; i+7 < capacity; i += 8 {
		a0 := float64(*(*T)(unsafe.Add(aData, uintptr(i+0)*aSize)))
		a1 := float64(*(*T)(unsafe.Add(aData, uintptr(i+1)*aSize)))
		a2 := float64(*(*T)(unsafe.Add(aData, uintptr(i+2)*aSize)))
		a3 := float64(*(*T)(unsafe.Add(aData, uintptr(i+3)*aSize)))
		a4 := float64(*(*T)(unsafe.Add(aData, uintptr(i+4)*aSize)))
		a5 := float64(*(*T)(unsafe.Add(aData, uintptr(i+5)*aSize)))
		a6 := float64(*(*T)(unsafe.Add(aData, uintptr(i+6)*aSize)))
		a7 := float64(*(*T)(unsafe.Add(aData, uintptr(i+7)*aSize)))

		b0 := float64(*(*U)(unsafe.Add(bData, uintptr(i+0)*bSize)))
		b1 := float64(*(*U)(unsafe.Add(bData, uintptr(i+1)*bSize)))
		b2 := float64(*(*U)(unsafe.Add(bData, uintptr(i+2)*bSize)))
		b3 := float64(*(*U)(unsafe.Add(bData, uintptr(i+3)*bSize)))
		b4 := float64(*(*U)(unsafe.Add(bData, uintptr(i+4)*bSize)))
		b5 := float64(*(*U)(unsafe.Add(bData, uintptr(i+5)*bSize)))
		b6 := float64(*(*U)(unsafe.Add(bData, uintptr(i+6)*bSize)))
		b7 := float64(*(*U)(unsafe.Add(bData, uintptr(i+7)*bSize)))

		*(*bool)(unsafe.Add(maskData, uintptr(i+0)*maskSize)) = a0 >= b0
		*(*bool)(unsafe.Add(maskData, uintptr(i+1)*maskSize)) = a1 >= b1
		*(*bool)(unsafe.Add(maskData, uintptr(i+2)*maskSize)) = a2 >= b2
		*(*bool)(unsafe.Add(maskData, uintptr(i+3)*maskSize)) = a3 >= b3
		*(*bool)(unsafe.Add(maskData, uintptr(i+4)*maskSize)) = a4 >= b4
		*(*bool)(unsafe.Add(maskData, uintptr(i+5)*maskSize)) = a5 >= b5
		*(*bool)(unsafe.Add(maskData, uintptr(i+6)*maskSize)) = a6 >= b6
		*(*bool)(unsafe.Add(maskData, uintptr(i+7)*maskSize)) = a7 >= b7
	}

	// Handle remaining elements
	for ; i < capacity; i++ {
		a := float64(*(*T)(unsafe.Add(aData, uintptr(i)*aSize)))
		b := float64(*(*U)(unsafe.Add(bData, uintptr(i)*bSize)))
		*(*bool)(unsafe.Add(maskData, uintptr(i)*maskSize)) = a >= b
	}
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
	aCapacity := memstruct.VectorCapacityGet[T](vectorAAddr)
	bCapacity := memstruct.VectorCapacityGet[U](vectorBAddr)

	if aCapacity != bCapacity {
		panic(fmt.Errorf("cannot perform compare: capacity mismatch (a=%d, b=%d)",
			aCapacity, bCapacity))
	}

	aData := memstruct.VectorDataPtrGet[T](vectorAAddr)
	bData := memstruct.VectorDataPtrGet[U](vectorBAddr)
	maskData := memstruct.ArrayDataPtrGet[bool](maskAddr)
	aSize := uintptr(memcore.SizeOf[T]())
	bSize := uintptr(memcore.SizeOf[U]())
	maskSize := uintptr(memcore.SizeOf[bool]())
	capacity := aCapacity

	var i uint64

	// Manually unrolled loop for stride 8
	for ; i+7 < capacity; i += 8 {
		a0 := float64(*(*T)(unsafe.Add(aData, uintptr(i+0)*aSize)))
		a1 := float64(*(*T)(unsafe.Add(aData, uintptr(i+1)*aSize)))
		a2 := float64(*(*T)(unsafe.Add(aData, uintptr(i+2)*aSize)))
		a3 := float64(*(*T)(unsafe.Add(aData, uintptr(i+3)*aSize)))
		a4 := float64(*(*T)(unsafe.Add(aData, uintptr(i+4)*aSize)))
		a5 := float64(*(*T)(unsafe.Add(aData, uintptr(i+5)*aSize)))
		a6 := float64(*(*T)(unsafe.Add(aData, uintptr(i+6)*aSize)))
		a7 := float64(*(*T)(unsafe.Add(aData, uintptr(i+7)*aSize)))

		b0 := float64(*(*U)(unsafe.Add(bData, uintptr(i+0)*bSize)))
		b1 := float64(*(*U)(unsafe.Add(bData, uintptr(i+1)*bSize)))
		b2 := float64(*(*U)(unsafe.Add(bData, uintptr(i+2)*bSize)))
		b3 := float64(*(*U)(unsafe.Add(bData, uintptr(i+3)*bSize)))
		b4 := float64(*(*U)(unsafe.Add(bData, uintptr(i+4)*bSize)))
		b5 := float64(*(*U)(unsafe.Add(bData, uintptr(i+5)*bSize)))
		b6 := float64(*(*U)(unsafe.Add(bData, uintptr(i+6)*bSize)))
		b7 := float64(*(*U)(unsafe.Add(bData, uintptr(i+7)*bSize)))

		*(*bool)(unsafe.Add(maskData, uintptr(i+0)*maskSize)) = a0 > b0
		*(*bool)(unsafe.Add(maskData, uintptr(i+1)*maskSize)) = a1 > b1
		*(*bool)(unsafe.Add(maskData, uintptr(i+2)*maskSize)) = a2 > b2
		*(*bool)(unsafe.Add(maskData, uintptr(i+3)*maskSize)) = a3 > b3
		*(*bool)(unsafe.Add(maskData, uintptr(i+4)*maskSize)) = a4 > b4
		*(*bool)(unsafe.Add(maskData, uintptr(i+5)*maskSize)) = a5 > b5
		*(*bool)(unsafe.Add(maskData, uintptr(i+6)*maskSize)) = a6 > b6
		*(*bool)(unsafe.Add(maskData, uintptr(i+7)*maskSize)) = a7 > b7
	}

	// Handle remaining elements
	for ; i < capacity; i++ {
		a := float64(*(*T)(unsafe.Add(aData, uintptr(i)*aSize)))
		b := float64(*(*U)(unsafe.Add(bData, uintptr(i)*bSize)))
		*(*bool)(unsafe.Add(maskData, uintptr(i)*maskSize)) = a > b
	}
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
	aCapacity := memstruct.VectorCapacityGet[T](vectorAAddr)
	bCapacity := memstruct.VectorCapacityGet[U](vectorBAddr)

	if aCapacity != bCapacity {
		panic(fmt.Errorf("cannot perform compare: capacity mismatch (a=%d, b=%d)",
			aCapacity, bCapacity))
	}

	aData := memstruct.VectorDataPtrGet[T](vectorAAddr)
	bData := memstruct.VectorDataPtrGet[U](vectorBAddr)
	maskData := memstruct.ArrayDataPtrGet[bool](maskAddr)
	aSize := uintptr(memcore.SizeOf[T]())
	bSize := uintptr(memcore.SizeOf[U]())
	maskSize := uintptr(memcore.SizeOf[bool]())
	capacity := aCapacity

	var i uint64

	// Manually unrolled loop for stride 8
	for ; i+7 < capacity; i += 8 {
		a0 := float64(*(*T)(unsafe.Add(aData, uintptr(i+0)*aSize)))
		a1 := float64(*(*T)(unsafe.Add(aData, uintptr(i+1)*aSize)))
		a2 := float64(*(*T)(unsafe.Add(aData, uintptr(i+2)*aSize)))
		a3 := float64(*(*T)(unsafe.Add(aData, uintptr(i+3)*aSize)))
		a4 := float64(*(*T)(unsafe.Add(aData, uintptr(i+4)*aSize)))
		a5 := float64(*(*T)(unsafe.Add(aData, uintptr(i+5)*aSize)))
		a6 := float64(*(*T)(unsafe.Add(aData, uintptr(i+6)*aSize)))
		a7 := float64(*(*T)(unsafe.Add(aData, uintptr(i+7)*aSize)))

		b0 := float64(*(*U)(unsafe.Add(bData, uintptr(i+0)*bSize)))
		b1 := float64(*(*U)(unsafe.Add(bData, uintptr(i+1)*bSize)))
		b2 := float64(*(*U)(unsafe.Add(bData, uintptr(i+2)*bSize)))
		b3 := float64(*(*U)(unsafe.Add(bData, uintptr(i+3)*bSize)))
		b4 := float64(*(*U)(unsafe.Add(bData, uintptr(i+4)*bSize)))
		b5 := float64(*(*U)(unsafe.Add(bData, uintptr(i+5)*bSize)))
		b6 := float64(*(*U)(unsafe.Add(bData, uintptr(i+6)*bSize)))
		b7 := float64(*(*U)(unsafe.Add(bData, uintptr(i+7)*bSize)))

		*(*bool)(unsafe.Add(maskData, uintptr(i+0)*maskSize)) = a0 <= b0
		*(*bool)(unsafe.Add(maskData, uintptr(i+1)*maskSize)) = a1 <= b1
		*(*bool)(unsafe.Add(maskData, uintptr(i+2)*maskSize)) = a2 <= b2
		*(*bool)(unsafe.Add(maskData, uintptr(i+3)*maskSize)) = a3 <= b3
		*(*bool)(unsafe.Add(maskData, uintptr(i+4)*maskSize)) = a4 <= b4
		*(*bool)(unsafe.Add(maskData, uintptr(i+5)*maskSize)) = a5 <= b5
		*(*bool)(unsafe.Add(maskData, uintptr(i+6)*maskSize)) = a6 <= b6
		*(*bool)(unsafe.Add(maskData, uintptr(i+7)*maskSize)) = a7 <= b7
	}

	// Handle remaining elements
	for ; i < capacity; i++ {
		a := float64(*(*T)(unsafe.Add(aData, uintptr(i)*aSize)))
		b := float64(*(*U)(unsafe.Add(bData, uintptr(i)*bSize)))
		*(*bool)(unsafe.Add(maskData, uintptr(i)*maskSize)) = a <= b
	}
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
	aCapacity := memstruct.VectorCapacityGet[T](vectorAAddr)
	bCapacity := memstruct.VectorCapacityGet[U](vectorBAddr)

	if aCapacity != bCapacity {
		panic(fmt.Errorf("cannot perform compare: capacity mismatch (a=%d, b=%d)",
			aCapacity, bCapacity))
	}

	aData := memstruct.VectorDataPtrGet[T](vectorAAddr)
	bData := memstruct.VectorDataPtrGet[U](vectorBAddr)
	maskData := memstruct.ArrayDataPtrGet[bool](maskAddr)
	aSize := uintptr(memcore.SizeOf[T]())
	bSize := uintptr(memcore.SizeOf[U]())
	maskSize := uintptr(memcore.SizeOf[bool]())
	capacity := aCapacity

	var i uint64

	// Manually unrolled loop for stride 8
	for ; i+7 < capacity; i += 8 {
		a0 := float64(*(*T)(unsafe.Add(aData, uintptr(i+0)*aSize)))
		a1 := float64(*(*T)(unsafe.Add(aData, uintptr(i+1)*aSize)))
		a2 := float64(*(*T)(unsafe.Add(aData, uintptr(i+2)*aSize)))
		a3 := float64(*(*T)(unsafe.Add(aData, uintptr(i+3)*aSize)))
		a4 := float64(*(*T)(unsafe.Add(aData, uintptr(i+4)*aSize)))
		a5 := float64(*(*T)(unsafe.Add(aData, uintptr(i+5)*aSize)))
		a6 := float64(*(*T)(unsafe.Add(aData, uintptr(i+6)*aSize)))
		a7 := float64(*(*T)(unsafe.Add(aData, uintptr(i+7)*aSize)))

		b0 := float64(*(*U)(unsafe.Add(bData, uintptr(i+0)*bSize)))
		b1 := float64(*(*U)(unsafe.Add(bData, uintptr(i+1)*bSize)))
		b2 := float64(*(*U)(unsafe.Add(bData, uintptr(i+2)*bSize)))
		b3 := float64(*(*U)(unsafe.Add(bData, uintptr(i+3)*bSize)))
		b4 := float64(*(*U)(unsafe.Add(bData, uintptr(i+4)*bSize)))
		b5 := float64(*(*U)(unsafe.Add(bData, uintptr(i+5)*bSize)))
		b6 := float64(*(*U)(unsafe.Add(bData, uintptr(i+6)*bSize)))
		b7 := float64(*(*U)(unsafe.Add(bData, uintptr(i+7)*bSize)))

		*(*bool)(unsafe.Add(maskData, uintptr(i+0)*maskSize)) = a0 < b0
		*(*bool)(unsafe.Add(maskData, uintptr(i+1)*maskSize)) = a1 < b1
		*(*bool)(unsafe.Add(maskData, uintptr(i+2)*maskSize)) = a2 < b2
		*(*bool)(unsafe.Add(maskData, uintptr(i+3)*maskSize)) = a3 < b3
		*(*bool)(unsafe.Add(maskData, uintptr(i+4)*maskSize)) = a4 < b4
		*(*bool)(unsafe.Add(maskData, uintptr(i+5)*maskSize)) = a5 < b5
		*(*bool)(unsafe.Add(maskData, uintptr(i+6)*maskSize)) = a6 < b6
		*(*bool)(unsafe.Add(maskData, uintptr(i+7)*maskSize)) = a7 < b7
	}

	// Handle remaining elements
	for ; i < capacity; i++ {
		a := float64(*(*T)(unsafe.Add(aData, uintptr(i)*aSize)))
		b := float64(*(*U)(unsafe.Add(bData, uintptr(i)*bSize)))
		*(*bool)(unsafe.Add(maskData, uintptr(i)*maskSize)) = a < b
	}
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
	aCapacity := memstruct.VectorCapacityGet[T](vectorAAddr)
	bCapacity := memstruct.VectorCapacityGet[U](vectorBAddr)

	if aCapacity != bCapacity {
		panic(fmt.Errorf("cannot perform compare: capacity mismatch (a=%d, b=%d)",
			aCapacity, bCapacity))
	}

	aData := memstruct.VectorDataPtrGet[T](vectorAAddr)
	bData := memstruct.VectorDataPtrGet[U](vectorBAddr)
	maskData := memstruct.ArrayDataPtrGet[bool](maskAddr)
	aSize := uintptr(memcore.SizeOf[T]())
	bSize := uintptr(memcore.SizeOf[U]())
	maskSize := uintptr(memcore.SizeOf[bool]())
	capacity := aCapacity

	var i uint64

	// Manually unrolled loop for stride 8
	for ; i+7 < capacity; i += 8 {
		a0 := float64(*(*T)(unsafe.Add(aData, uintptr(i+0)*aSize)))
		a1 := float64(*(*T)(unsafe.Add(aData, uintptr(i+1)*aSize)))
		a2 := float64(*(*T)(unsafe.Add(aData, uintptr(i+2)*aSize)))
		a3 := float64(*(*T)(unsafe.Add(aData, uintptr(i+3)*aSize)))
		a4 := float64(*(*T)(unsafe.Add(aData, uintptr(i+4)*aSize)))
		a5 := float64(*(*T)(unsafe.Add(aData, uintptr(i+5)*aSize)))
		a6 := float64(*(*T)(unsafe.Add(aData, uintptr(i+6)*aSize)))
		a7 := float64(*(*T)(unsafe.Add(aData, uintptr(i+7)*aSize)))

		b0 := float64(*(*U)(unsafe.Add(bData, uintptr(i+0)*bSize)))
		b1 := float64(*(*U)(unsafe.Add(bData, uintptr(i+1)*bSize)))
		b2 := float64(*(*U)(unsafe.Add(bData, uintptr(i+2)*bSize)))
		b3 := float64(*(*U)(unsafe.Add(bData, uintptr(i+3)*bSize)))
		b4 := float64(*(*U)(unsafe.Add(bData, uintptr(i+4)*bSize)))
		b5 := float64(*(*U)(unsafe.Add(bData, uintptr(i+5)*bSize)))
		b6 := float64(*(*U)(unsafe.Add(bData, uintptr(i+6)*bSize)))
		b7 := float64(*(*U)(unsafe.Add(bData, uintptr(i+7)*bSize)))

		*(*bool)(unsafe.Add(maskData, uintptr(i+0)*maskSize)) = math.Abs(a0-b0) <= tolerance
		*(*bool)(unsafe.Add(maskData, uintptr(i+1)*maskSize)) = math.Abs(a1-b1) <= tolerance
		*(*bool)(unsafe.Add(maskData, uintptr(i+2)*maskSize)) = math.Abs(a2-b2) <= tolerance
		*(*bool)(unsafe.Add(maskData, uintptr(i+3)*maskSize)) = math.Abs(a3-b3) <= tolerance
		*(*bool)(unsafe.Add(maskData, uintptr(i+4)*maskSize)) = math.Abs(a4-b4) <= tolerance
		*(*bool)(unsafe.Add(maskData, uintptr(i+5)*maskSize)) = math.Abs(a5-b5) <= tolerance
		*(*bool)(unsafe.Add(maskData, uintptr(i+6)*maskSize)) = math.Abs(a6-b6) <= tolerance
		*(*bool)(unsafe.Add(maskData, uintptr(i+7)*maskSize)) = math.Abs(a7-b7) <= tolerance
	}

	// Handle remaining elements
	for ; i < capacity; i++ {
		a := float64(*(*T)(unsafe.Add(aData, uintptr(i)*aSize)))
		b := float64(*(*U)(unsafe.Add(bData, uintptr(i)*bSize)))
		*(*bool)(unsafe.Add(maskData, uintptr(i)*maskSize)) = math.Abs(a-b) <= tolerance
	}
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
