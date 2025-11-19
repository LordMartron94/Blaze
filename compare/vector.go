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

// BlazeCompareVectorGreaterThanOrEqualTo produces a boolean mask determining whether each element
// in Vector A is bigger than or equal to the same element in Vector B.
// The mask must be of type Array[bool]
// Capacity equality must be guaranteed by the caller.
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

// BlazeCompareVectorGreaterThan produces a boolean mask determining whether each element
// in Vector A is bigger than the same element in Vector B.
// The mask must be of type Array[bool]
// Capacity equality must be guaranteed by the caller.
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

// BlazeCompareVectorSmallerThanOrEqualTo produces a boolean mask determining whether each element
// in Vector A is smaller than or equal to the same element in Vector B.
// The mask must be of type Array[bool]
// Capacity equality must be guaranteed by the caller.
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

// BlazeCompareVectorSmallerThan produces a boolean mask determining whether each element
// in Vector A is smaller than the same element in Vector B.
// The mask must be of type Array[bool]
// Capacity equality must be guaranteed by the caller.
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

// BlazeCompareVectorEqualTo produces a boolean mask determining whether each element
// in Vector A is equal to the same element in Vector B within tolerance.
// The mask must be of type Array[bool]
// Capacity equality must be guaranteed by the caller.
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

// BlazeCompareVectorSparseJaccardWeightedSimilarity computes similarity over two
// sparse vectors using weighted Jaccard similarity.
// Both vectors contain IDs, and both weight arrays contain a per-ID weight.
// The function assumes that every index inside vector capacity is used.
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
