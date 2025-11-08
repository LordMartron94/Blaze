package compare

import (
	"blaze/core"
	"foundation"
	"math"
	"memcore"
	"memstruct"
)

// BlazeElementWiseVectorGreaterThanOrEqualTo produces a boolean mask determining whether each element
// in Vector A is bigger than or equal to the same element in Vector B.
// The mask must be of type Array[bool]
// Capacity equality must be guaranteed by the caller.
func BlazeElementWiseVectorGreaterThanOrEqualTo[T, U foundation.Numeric](
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

// BlazeElementWiseVectorGreaterThan produces a boolean mask determining whether each element
// in Vector A is bigger than the same element in Vector B.
// The mask must be of type Array[bool]
// Capacity equality must be guaranteed by the caller.
func BlazeElementWiseVectorGreaterThan[T, U foundation.Numeric](
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

// BlazeElementWiseVectorSmallerThanOrEqualTo produces a boolean mask determining whether each element
// in Vector A is smaller than or equal to the same element in Vector B.
// The mask must be of type Array[bool]
// Capacity equality must be guaranteed by the caller.
func BlazeElementWiseVectorSmallerThanOrEqualTo[T, U foundation.Numeric](
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

// BlazeElementWiseVectorSmallerThan produces a boolean mask determining whether each element
// in Vector A is smaller than the same element in Vector B.
// The mask must be of type Array[bool]
// Capacity equality must be guaranteed by the caller.
func BlazeElementWiseVectorSmallerThan[T, U foundation.Numeric](
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

// BlazeElementWiseVectorEqualTo produces a boolean mask determining whether each element
// in Vector A is equal to the same element in Vector B within tolerance.
// The mask must be of type Array[bool]
// Capacity equality must be guaranteed by the caller.
func BlazeElementWiseVectorEqualTo[T, U foundation.Numeric](
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
