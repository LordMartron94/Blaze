package elementwise

import (
	"blaze/core"
	"foundation"
	"memcore"
	"memstruct"
)

// BlazeElementWiseVectorAddF32 adds the values of Vector B to Vector A, resulting in Vector C at newVectorAddr.
// It does so in float32 precision.
func BlazeElementWiseVectorAddF32[T, U foundation.Numeric](
	vectorAAddr, vectorBAddr, newVectorAddr memcore.MarkRaw,
) {
	memstruct.VectorBinaryExecute(vectorAAddr, vectorBAddr, newVectorAddr, func(a T, b U) float32 {
		return float32(a) + float32(b)
	}, core.BlazeDefaultStride)
}

// BlazeElementWiseVectorAddF64 adds the values of Vector B to Vector A, resulting in Vector C at newVectorAddr.
// It does so in float64 precision.
func BlazeElementWiseVectorAddF64[T, U foundation.Numeric](
	vectorAAddr, vectorBAddr, newVectorAddr memcore.MarkRaw,
) {
	memstruct.VectorBinaryExecute(vectorAAddr, vectorBAddr, newVectorAddr, func(a T, b U) float64 {
		return float64(a) + float64(b)
	}, core.BlazeDefaultStride)
}

// BlazeElementWiseVectorSubtractF32 subtracts the values of Vector B from Vector A, resulting in Vector C at newVectorAddr.
// It does so in float32 precision.
func BlazeElementWiseVectorSubtractF32[T, U foundation.Numeric](
	vectorAAddr, vectorBAddr, newVectorAddr memcore.MarkRaw,
) {
	memstruct.VectorBinaryExecute(vectorAAddr, vectorBAddr, newVectorAddr, func(a T, b U) float32 {
		return float32(a) - float32(b)
	}, core.BlazeDefaultStride)
}

// BlazeElementWiseVectorSubtractF64 subtracts the values of Vector B from Vector A, resulting in Vector C at newVectorAddr.
// It does so in float364precision.
func BlazeElementWiseVectorSubtractF64[T, U foundation.Numeric](
	vectorAAddr, vectorBAddr, newVectorAddr memcore.MarkRaw,
) {
	memstruct.VectorBinaryExecute(vectorAAddr, vectorBAddr, newVectorAddr, func(a T, b U) float64 {
		return float64(a) - float64(b)
	}, core.BlazeDefaultStride)
}

// BlazeElementWiseVectorMultiplyF32 multiplies the values of Vector A by Vector B, resulting in Vector C at newVectorAddr.
// It does so in float32 precision.
func BlazeElementWiseVectorMultiplyF32[T, U foundation.Numeric](
	vectorAAddr, vectorBAddr, newVectorAddr memcore.MarkRaw,
) {
	memstruct.VectorBinaryExecute(vectorAAddr, vectorBAddr, newVectorAddr, func(a T, b U) float32 {
		return float32(a) * float32(b)
	}, core.BlazeDefaultStride)
}

// BlazeElementWiseVectorMultiplyF64 multiplies the values of Vector A by Vector B, resulting in Vector C at newVectorAddr.
// It does so in float64 precision.
func BlazeElementWiseVectorMultiplyF64[T, U foundation.Numeric](
	vectorAAddr, vectorBAddr, newVectorAddr memcore.MarkRaw,
) {
	memstruct.VectorBinaryExecute(vectorAAddr, vectorBAddr, newVectorAddr, func(a T, b U) float64 {
		return float64(a) * float64(b)
	}, core.BlazeDefaultStride)
}

// BlazeElementWiseVectorDivideF32 divides the values of Vector A by Vector B, resulting in Vector C at newVectorAddr.
// It does so in float32 precision.
func BlazeElementWiseVectorDivideF32[T, U foundation.Numeric](
	vectorAAddr, vectorBAddr, newVectorAddr memcore.MarkRaw,
) {
	memstruct.VectorBinaryExecute(vectorAAddr, vectorBAddr, newVectorAddr, func(a T, b U) float32 {
		return float32(a) / float32(b)
	}, core.BlazeDefaultStride)
}

// BlazeElementWiseVectorDivideF64 divides the values of Vector A by Vector B, resulting in Vector C at newVectorAddr.
// It does so in float64 precision.
func BlazeElementWiseVectorDivideF64[T, U foundation.Numeric](
	vectorAAddr, vectorBAddr, newVectorAddr memcore.MarkRaw,
) {
	memstruct.VectorBinaryExecute(vectorAAddr, vectorBAddr, newVectorAddr, func(a T, b U) float64 {
		return float64(a) / float64(b)
	}, core.BlazeDefaultStride)
}
