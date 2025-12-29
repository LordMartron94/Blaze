package elementwise

import (
	"fmt"
	"unsafe"

	"foundation"
	"memcore"
	"memstruct"
)

/*
BlazeElementWiseVectorAddF32 performs element-wise addition of two vectors in float32 precision.

Computes: result[i] = vectorA[i] + vectorB[i] for all elements i.

Use cases:
- Signal processing (combining signals)
- Image processing (blending images)
- Numerical simulations (combining vectors)
- Data transformations

Time complexity: O(n) - single pass through vectors
Space complexity: O(1) - only accumulator variables used

Prerequisites:
- Both input vectors must have the same length
- Output vector must have the same length as inputs
- All vectors must be valid and initialized

Edge cases:
- Works with any numeric type (int, float32, float64, etc.)
- Type conversions occur during computation (input types → float32)
- No overflow checking (relies on Go's numeric behavior)
*/
func BlazeElementWiseVectorAddF32[T, U foundation.Numeric](
	vectorAAddr, vectorBAddr, newVectorAddr memcore.MarkRaw,
) {
	aCapacity := memstruct.VectorCapacityGet[T](vectorAAddr)
	bCapacity := memstruct.VectorCapacityGet[U](vectorBAddr)

	if aCapacity != bCapacity {
		panic(fmt.Errorf("cannot perform elementwise add: capacity mismatch (a=%d, b=%d)",
			aCapacity, bCapacity))
	}

	aData := memstruct.VectorDataPtrGet[T](vectorAAddr)
	bData := memstruct.VectorDataPtrGet[U](vectorBAddr)
	dstData := memstruct.VectorDataPtrGet[float32](newVectorAddr)
	aSize := uintptr(memcore.SizeOf[T]())
	bSize := uintptr(memcore.SizeOf[U]())
	dstSize := uintptr(memcore.SizeOf[float32]())
	capacity := aCapacity

	var i uint64

	// Manually unrolled loop for stride 8
	for ; i+7 < capacity; i += 8 {
		a0 := float32(*(*T)(unsafe.Add(aData, uintptr(i+0)*aSize)))
		a1 := float32(*(*T)(unsafe.Add(aData, uintptr(i+1)*aSize)))
		a2 := float32(*(*T)(unsafe.Add(aData, uintptr(i+2)*aSize)))
		a3 := float32(*(*T)(unsafe.Add(aData, uintptr(i+3)*aSize)))
		a4 := float32(*(*T)(unsafe.Add(aData, uintptr(i+4)*aSize)))
		a5 := float32(*(*T)(unsafe.Add(aData, uintptr(i+5)*aSize)))
		a6 := float32(*(*T)(unsafe.Add(aData, uintptr(i+6)*aSize)))
		a7 := float32(*(*T)(unsafe.Add(aData, uintptr(i+7)*aSize)))

		b0 := float32(*(*U)(unsafe.Add(bData, uintptr(i+0)*bSize)))
		b1 := float32(*(*U)(unsafe.Add(bData, uintptr(i+1)*bSize)))
		b2 := float32(*(*U)(unsafe.Add(bData, uintptr(i+2)*bSize)))
		b3 := float32(*(*U)(unsafe.Add(bData, uintptr(i+3)*bSize)))
		b4 := float32(*(*U)(unsafe.Add(bData, uintptr(i+4)*bSize)))
		b5 := float32(*(*U)(unsafe.Add(bData, uintptr(i+5)*bSize)))
		b6 := float32(*(*U)(unsafe.Add(bData, uintptr(i+6)*bSize)))
		b7 := float32(*(*U)(unsafe.Add(bData, uintptr(i+7)*bSize)))

		*(*float32)(unsafe.Add(dstData, uintptr(i+0)*dstSize)) = a0 + b0
		*(*float32)(unsafe.Add(dstData, uintptr(i+1)*dstSize)) = a1 + b1
		*(*float32)(unsafe.Add(dstData, uintptr(i+2)*dstSize)) = a2 + b2
		*(*float32)(unsafe.Add(dstData, uintptr(i+3)*dstSize)) = a3 + b3
		*(*float32)(unsafe.Add(dstData, uintptr(i+4)*dstSize)) = a4 + b4
		*(*float32)(unsafe.Add(dstData, uintptr(i+5)*dstSize)) = a5 + b5
		*(*float32)(unsafe.Add(dstData, uintptr(i+6)*dstSize)) = a6 + b6
		*(*float32)(unsafe.Add(dstData, uintptr(i+7)*dstSize)) = a7 + b7
	}

	// Handle remaining elements
	for ; i < capacity; i++ {
		a := float32(*(*T)(unsafe.Add(aData, uintptr(i)*aSize)))
		b := float32(*(*U)(unsafe.Add(bData, uintptr(i)*bSize)))
		*(*float32)(unsafe.Add(dstData, uintptr(i)*dstSize)) = a + b
	}
}

/*
BlazeElementWiseVectorAddF64 performs element-wise addition of two vectors in float64 precision.

Computes: result[i] = vectorA[i] + vectorB[i] for all elements i.

Use cases:
- Signal processing (combining signals)
- Image processing (blending images)
- Numerical simulations (combining vectors)
- Data transformations

Time complexity: O(n) - single pass through vectors
Space complexity: O(1) - only accumulator variables used

Prerequisites:
- Both input vectors must have the same length
- Output vector must have the same length as inputs
- All vectors must be valid and initialized

Edge cases:
- Works with any numeric type (int, float32, float64, etc.)
- Type conversions occur during computation (input types → float64)
- No overflow checking (relies on Go's numeric behavior)
*/
func BlazeElementWiseVectorAddF64[T, U foundation.Numeric](
	vectorAAddr, vectorBAddr, newVectorAddr memcore.MarkRaw,
) {
	aCapacity := memstruct.VectorCapacityGet[T](vectorAAddr)
	bCapacity := memstruct.VectorCapacityGet[U](vectorBAddr)

	if aCapacity != bCapacity {
		panic(fmt.Errorf("cannot perform elementwise add: capacity mismatch (a=%d, b=%d)",
			aCapacity, bCapacity))
	}

	aData := memstruct.VectorDataPtrGet[T](vectorAAddr)
	bData := memstruct.VectorDataPtrGet[U](vectorBAddr)
	dstData := memstruct.VectorDataPtrGet[float64](newVectorAddr)
	aSize := uintptr(memcore.SizeOf[T]())
	bSize := uintptr(memcore.SizeOf[U]())
	dstSize := uintptr(memcore.SizeOf[float64]())
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

		*(*float64)(unsafe.Add(dstData, uintptr(i+0)*dstSize)) = a0 + b0
		*(*float64)(unsafe.Add(dstData, uintptr(i+1)*dstSize)) = a1 + b1
		*(*float64)(unsafe.Add(dstData, uintptr(i+2)*dstSize)) = a2 + b2
		*(*float64)(unsafe.Add(dstData, uintptr(i+3)*dstSize)) = a3 + b3
		*(*float64)(unsafe.Add(dstData, uintptr(i+4)*dstSize)) = a4 + b4
		*(*float64)(unsafe.Add(dstData, uintptr(i+5)*dstSize)) = a5 + b5
		*(*float64)(unsafe.Add(dstData, uintptr(i+6)*dstSize)) = a6 + b6
		*(*float64)(unsafe.Add(dstData, uintptr(i+7)*dstSize)) = a7 + b7
	}

	// Handle remaining elements
	for ; i < capacity; i++ {
		a := float64(*(*T)(unsafe.Add(aData, uintptr(i)*aSize)))
		b := float64(*(*U)(unsafe.Add(bData, uintptr(i)*bSize)))
		*(*float64)(unsafe.Add(dstData, uintptr(i)*dstSize)) = a + b
	}
}

/*
BlazeElementWiseVectorSubtractF32 performs element-wise subtraction of two vectors in float32 precision.

Computes: result[i] = vectorA[i] - vectorB[i] for all elements i.

Use cases:
- Signal processing (difference between signals)
- Image processing (subtracting backgrounds)
- Numerical simulations (computing differences)
- Error calculations

Time complexity: O(n) - single pass through vectors
Space complexity: O(1) - only accumulator variables used

Prerequisites:
- Both input vectors must have the same length
- Output vector must have the same length as inputs
- All vectors must be valid and initialized

Edge cases:
- Works with any numeric type (int, float32, float64, etc.)
- Type conversions occur during computation (input types → float32)
- No overflow checking (relies on Go's numeric behavior)
*/
func BlazeElementWiseVectorSubtractF32[T, U foundation.Numeric](
	vectorAAddr, vectorBAddr, newVectorAddr memcore.MarkRaw,
) {
	aCapacity := memstruct.VectorCapacityGet[T](vectorAAddr)
	bCapacity := memstruct.VectorCapacityGet[U](vectorBAddr)

	if aCapacity != bCapacity {
		panic(fmt.Errorf("cannot perform elementwise subtract: capacity mismatch (a=%d, b=%d)",
			aCapacity, bCapacity))
	}

	aData := memstruct.VectorDataPtrGet[T](vectorAAddr)
	bData := memstruct.VectorDataPtrGet[U](vectorBAddr)
	dstData := memstruct.VectorDataPtrGet[float32](newVectorAddr)
	aSize := uintptr(memcore.SizeOf[T]())
	bSize := uintptr(memcore.SizeOf[U]())
	dstSize := uintptr(memcore.SizeOf[float32]())
	capacity := aCapacity

	var i uint64

	// Manually unrolled loop for stride 8
	for ; i+7 < capacity; i += 8 {
		a0 := float32(*(*T)(unsafe.Add(aData, uintptr(i+0)*aSize)))
		a1 := float32(*(*T)(unsafe.Add(aData, uintptr(i+1)*aSize)))
		a2 := float32(*(*T)(unsafe.Add(aData, uintptr(i+2)*aSize)))
		a3 := float32(*(*T)(unsafe.Add(aData, uintptr(i+3)*aSize)))
		a4 := float32(*(*T)(unsafe.Add(aData, uintptr(i+4)*aSize)))
		a5 := float32(*(*T)(unsafe.Add(aData, uintptr(i+5)*aSize)))
		a6 := float32(*(*T)(unsafe.Add(aData, uintptr(i+6)*aSize)))
		a7 := float32(*(*T)(unsafe.Add(aData, uintptr(i+7)*aSize)))

		b0 := float32(*(*U)(unsafe.Add(bData, uintptr(i+0)*bSize)))
		b1 := float32(*(*U)(unsafe.Add(bData, uintptr(i+1)*bSize)))
		b2 := float32(*(*U)(unsafe.Add(bData, uintptr(i+2)*bSize)))
		b3 := float32(*(*U)(unsafe.Add(bData, uintptr(i+3)*bSize)))
		b4 := float32(*(*U)(unsafe.Add(bData, uintptr(i+4)*bSize)))
		b5 := float32(*(*U)(unsafe.Add(bData, uintptr(i+5)*bSize)))
		b6 := float32(*(*U)(unsafe.Add(bData, uintptr(i+6)*bSize)))
		b7 := float32(*(*U)(unsafe.Add(bData, uintptr(i+7)*bSize)))

		*(*float32)(unsafe.Add(dstData, uintptr(i+0)*dstSize)) = a0 - b0
		*(*float32)(unsafe.Add(dstData, uintptr(i+1)*dstSize)) = a1 - b1
		*(*float32)(unsafe.Add(dstData, uintptr(i+2)*dstSize)) = a2 - b2
		*(*float32)(unsafe.Add(dstData, uintptr(i+3)*dstSize)) = a3 - b3
		*(*float32)(unsafe.Add(dstData, uintptr(i+4)*dstSize)) = a4 - b4
		*(*float32)(unsafe.Add(dstData, uintptr(i+5)*dstSize)) = a5 - b5
		*(*float32)(unsafe.Add(dstData, uintptr(i+6)*dstSize)) = a6 - b6
		*(*float32)(unsafe.Add(dstData, uintptr(i+7)*dstSize)) = a7 - b7
	}

	// Handle remaining elements
	for ; i < capacity; i++ {
		a := float32(*(*T)(unsafe.Add(aData, uintptr(i)*aSize)))
		b := float32(*(*U)(unsafe.Add(bData, uintptr(i)*bSize)))
		*(*float32)(unsafe.Add(dstData, uintptr(i)*dstSize)) = a - b
	}
}

/*
BlazeElementWiseVectorSubtractF64 performs element-wise subtraction of two vectors in float64 precision.

Computes: result[i] = vectorA[i] - vectorB[i] for all elements i.

Use cases:
- Signal processing (difference between signals)
- Image processing (subtracting backgrounds)
- Numerical simulations (computing differences)
- Error calculations

Time complexity: O(n) - single pass through vectors
Space complexity: O(1) - only accumulator variables used

Prerequisites:
- Both input vectors must have the same length
- Output vector must have the same length as inputs
- All vectors must be valid and initialized

Edge cases:
- Works with any numeric type (int, float32, float64, etc.)
- Type conversions occur during computation (input types → float64)
- No overflow checking (relies on Go's numeric behavior)
*/
func BlazeElementWiseVectorSubtractF64[T, U foundation.Numeric](
	vectorAAddr, vectorBAddr, newVectorAddr memcore.MarkRaw,
) {
	aCapacity := memstruct.VectorCapacityGet[T](vectorAAddr)
	bCapacity := memstruct.VectorCapacityGet[U](vectorBAddr)

	if aCapacity != bCapacity {
		panic(fmt.Errorf("cannot perform elementwise subtract: capacity mismatch (a=%d, b=%d)",
			aCapacity, bCapacity))
	}

	aData := memstruct.VectorDataPtrGet[T](vectorAAddr)
	bData := memstruct.VectorDataPtrGet[U](vectorBAddr)
	dstData := memstruct.VectorDataPtrGet[float64](newVectorAddr)
	aSize := uintptr(memcore.SizeOf[T]())
	bSize := uintptr(memcore.SizeOf[U]())
	dstSize := uintptr(memcore.SizeOf[float64]())
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

		*(*float64)(unsafe.Add(dstData, uintptr(i+0)*dstSize)) = a0 - b0
		*(*float64)(unsafe.Add(dstData, uintptr(i+1)*dstSize)) = a1 - b1
		*(*float64)(unsafe.Add(dstData, uintptr(i+2)*dstSize)) = a2 - b2
		*(*float64)(unsafe.Add(dstData, uintptr(i+3)*dstSize)) = a3 - b3
		*(*float64)(unsafe.Add(dstData, uintptr(i+4)*dstSize)) = a4 - b4
		*(*float64)(unsafe.Add(dstData, uintptr(i+5)*dstSize)) = a5 - b5
		*(*float64)(unsafe.Add(dstData, uintptr(i+6)*dstSize)) = a6 - b6
		*(*float64)(unsafe.Add(dstData, uintptr(i+7)*dstSize)) = a7 - b7
	}

	// Handle remaining elements
	for ; i < capacity; i++ {
		a := float64(*(*T)(unsafe.Add(aData, uintptr(i)*aSize)))
		b := float64(*(*U)(unsafe.Add(bData, uintptr(i)*bSize)))
		*(*float64)(unsafe.Add(dstData, uintptr(i)*dstSize)) = a - b
	}
}

/*
BlazeElementWiseVectorMultiplyF32 performs element-wise multiplication of two vectors in float32 precision.

Computes: result[i] = vectorA[i] * vectorB[i] for all elements i.

Use cases:
- Signal processing (modulation, filtering)
- Image processing (masking, blending)
- Numerical simulations (element-wise scaling)
- Weighted operations

Time complexity: O(n) - single pass through vectors
Space complexity: O(1) - only accumulator variables used

Prerequisites:
- Both input vectors must have the same length
- Output vector must have the same length as inputs
- All vectors must be valid and initialized

Edge cases:
- Works with any numeric type (int, float32, float64, etc.)
- Type conversions occur during computation (input types → float32)
- No overflow checking (relies on Go's numeric behavior)
*/
func BlazeElementWiseVectorMultiplyF32[T, U foundation.Numeric](
	vectorAAddr, vectorBAddr, newVectorAddr memcore.MarkRaw,
) {
	aCapacity := memstruct.VectorCapacityGet[T](vectorAAddr)
	bCapacity := memstruct.VectorCapacityGet[U](vectorBAddr)

	if aCapacity != bCapacity {
		panic(fmt.Errorf("cannot perform elementwise multiply: capacity mismatch (a=%d, b=%d)",
			aCapacity, bCapacity))
	}

	aData := memstruct.VectorDataPtrGet[T](vectorAAddr)
	bData := memstruct.VectorDataPtrGet[U](vectorBAddr)
	dstData := memstruct.VectorDataPtrGet[float32](newVectorAddr)
	aSize := uintptr(memcore.SizeOf[T]())
	bSize := uintptr(memcore.SizeOf[U]())
	dstSize := uintptr(memcore.SizeOf[float32]())
	capacity := aCapacity

	var i uint64

	// Manually unrolled loop for stride 8
	for ; i+7 < capacity; i += 8 {
		a0 := float32(*(*T)(unsafe.Add(aData, uintptr(i+0)*aSize)))
		a1 := float32(*(*T)(unsafe.Add(aData, uintptr(i+1)*aSize)))
		a2 := float32(*(*T)(unsafe.Add(aData, uintptr(i+2)*aSize)))
		a3 := float32(*(*T)(unsafe.Add(aData, uintptr(i+3)*aSize)))
		a4 := float32(*(*T)(unsafe.Add(aData, uintptr(i+4)*aSize)))
		a5 := float32(*(*T)(unsafe.Add(aData, uintptr(i+5)*aSize)))
		a6 := float32(*(*T)(unsafe.Add(aData, uintptr(i+6)*aSize)))
		a7 := float32(*(*T)(unsafe.Add(aData, uintptr(i+7)*aSize)))

		b0 := float32(*(*U)(unsafe.Add(bData, uintptr(i+0)*bSize)))
		b1 := float32(*(*U)(unsafe.Add(bData, uintptr(i+1)*bSize)))
		b2 := float32(*(*U)(unsafe.Add(bData, uintptr(i+2)*bSize)))
		b3 := float32(*(*U)(unsafe.Add(bData, uintptr(i+3)*bSize)))
		b4 := float32(*(*U)(unsafe.Add(bData, uintptr(i+4)*bSize)))
		b5 := float32(*(*U)(unsafe.Add(bData, uintptr(i+5)*bSize)))
		b6 := float32(*(*U)(unsafe.Add(bData, uintptr(i+6)*bSize)))
		b7 := float32(*(*U)(unsafe.Add(bData, uintptr(i+7)*bSize)))

		*(*float32)(unsafe.Add(dstData, uintptr(i+0)*dstSize)) = a0 * b0
		*(*float32)(unsafe.Add(dstData, uintptr(i+1)*dstSize)) = a1 * b1
		*(*float32)(unsafe.Add(dstData, uintptr(i+2)*dstSize)) = a2 * b2
		*(*float32)(unsafe.Add(dstData, uintptr(i+3)*dstSize)) = a3 * b3
		*(*float32)(unsafe.Add(dstData, uintptr(i+4)*dstSize)) = a4 * b4
		*(*float32)(unsafe.Add(dstData, uintptr(i+5)*dstSize)) = a5 * b5
		*(*float32)(unsafe.Add(dstData, uintptr(i+6)*dstSize)) = a6 * b6
		*(*float32)(unsafe.Add(dstData, uintptr(i+7)*dstSize)) = a7 * b7
	}

	// Handle remaining elements
	for ; i < capacity; i++ {
		a := float32(*(*T)(unsafe.Add(aData, uintptr(i)*aSize)))
		b := float32(*(*U)(unsafe.Add(bData, uintptr(i)*bSize)))
		*(*float32)(unsafe.Add(dstData, uintptr(i)*dstSize)) = a * b
	}
}

/*
BlazeElementWiseVectorMultiplyF64 performs element-wise multiplication of two vectors in float64 precision.

Computes: result[i] = vectorA[i] * vectorB[i] for all elements i.

Use cases:
- Signal processing (modulation, filtering)
- Image processing (masking, blending)
- Numerical simulations (element-wise scaling)
- Weighted operations

Time complexity: O(n) - single pass through vectors
Space complexity: O(1) - only accumulator variables used

Prerequisites:
- Both input vectors must have the same length
- Output vector must have the same length as inputs
- All vectors must be valid and initialized

Edge cases:
- Works with any numeric type (int, float32, float64, etc.)
- Type conversions occur during computation (input types → float64)
- No overflow checking (relies on Go's numeric behavior)
*/
func BlazeElementWiseVectorMultiplyF64[T, U foundation.Numeric](
	vectorAAddr, vectorBAddr, newVectorAddr memcore.MarkRaw,
) {
	aCapacity := memstruct.VectorCapacityGet[T](vectorAAddr)
	bCapacity := memstruct.VectorCapacityGet[U](vectorBAddr)

	if aCapacity != bCapacity {
		panic(fmt.Errorf("cannot perform elementwise multiply: capacity mismatch (a=%d, b=%d)",
			aCapacity, bCapacity))
	}

	aData := memstruct.VectorDataPtrGet[T](vectorAAddr)
	bData := memstruct.VectorDataPtrGet[U](vectorBAddr)
	dstData := memstruct.VectorDataPtrGet[float64](newVectorAddr)
	aSize := uintptr(memcore.SizeOf[T]())
	bSize := uintptr(memcore.SizeOf[U]())
	dstSize := uintptr(memcore.SizeOf[float64]())
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

		*(*float64)(unsafe.Add(dstData, uintptr(i+0)*dstSize)) = a0 * b0
		*(*float64)(unsafe.Add(dstData, uintptr(i+1)*dstSize)) = a1 * b1
		*(*float64)(unsafe.Add(dstData, uintptr(i+2)*dstSize)) = a2 * b2
		*(*float64)(unsafe.Add(dstData, uintptr(i+3)*dstSize)) = a3 * b3
		*(*float64)(unsafe.Add(dstData, uintptr(i+4)*dstSize)) = a4 * b4
		*(*float64)(unsafe.Add(dstData, uintptr(i+5)*dstSize)) = a5 * b5
		*(*float64)(unsafe.Add(dstData, uintptr(i+6)*dstSize)) = a6 * b6
		*(*float64)(unsafe.Add(dstData, uintptr(i+7)*dstSize)) = a7 * b7
	}

	// Handle remaining elements
	for ; i < capacity; i++ {
		a := float64(*(*T)(unsafe.Add(aData, uintptr(i)*aSize)))
		b := float64(*(*U)(unsafe.Add(bData, uintptr(i)*bSize)))
		*(*float64)(unsafe.Add(dstData, uintptr(i)*dstSize)) = a * b
	}
}

/*
BlazeElementWiseVectorDivideF32 performs element-wise division of two vectors in float32 precision.

Computes: result[i] = vectorA[i] / vectorB[i] for all elements i.

Use cases:
- Signal processing (normalization, deconvolution)
- Image processing (ratio images)
- Numerical simulations (element-wise scaling)
- Normalization operations

Time complexity: O(n) - single pass through vectors
Space complexity: O(1) - only accumulator variables used

Prerequisites:
- Both input vectors must have the same length
- Output vector must have the same length as inputs
- All vectors must be valid and initialized

Edge cases:
- Works with any numeric type (int, float32, float64, etc.)
- Type conversions occur during computation (input types → float32)
- Division by zero results in ±Inf or NaN (Go's standard behavior)
- No overflow checking (relies on Go's numeric behavior)
*/
func BlazeElementWiseVectorDivideF32[T, U foundation.Numeric](
	vectorAAddr, vectorBAddr, newVectorAddr memcore.MarkRaw,
) {
	aCapacity := memstruct.VectorCapacityGet[T](vectorAAddr)
	bCapacity := memstruct.VectorCapacityGet[U](vectorBAddr)

	if aCapacity != bCapacity {
		panic(fmt.Errorf("cannot perform elementwise divide: capacity mismatch (a=%d, b=%d)",
			aCapacity, bCapacity))
	}

	aData := memstruct.VectorDataPtrGet[T](vectorAAddr)
	bData := memstruct.VectorDataPtrGet[U](vectorBAddr)
	dstData := memstruct.VectorDataPtrGet[float32](newVectorAddr)
	aSize := uintptr(memcore.SizeOf[T]())
	bSize := uintptr(memcore.SizeOf[U]())
	dstSize := uintptr(memcore.SizeOf[float32]())
	capacity := aCapacity

	var i uint64

	// Manually unrolled loop for stride 8
	for ; i+7 < capacity; i += 8 {
		a0 := float32(*(*T)(unsafe.Add(aData, uintptr(i+0)*aSize)))
		a1 := float32(*(*T)(unsafe.Add(aData, uintptr(i+1)*aSize)))
		a2 := float32(*(*T)(unsafe.Add(aData, uintptr(i+2)*aSize)))
		a3 := float32(*(*T)(unsafe.Add(aData, uintptr(i+3)*aSize)))
		a4 := float32(*(*T)(unsafe.Add(aData, uintptr(i+4)*aSize)))
		a5 := float32(*(*T)(unsafe.Add(aData, uintptr(i+5)*aSize)))
		a6 := float32(*(*T)(unsafe.Add(aData, uintptr(i+6)*aSize)))
		a7 := float32(*(*T)(unsafe.Add(aData, uintptr(i+7)*aSize)))

		b0 := float32(*(*U)(unsafe.Add(bData, uintptr(i+0)*bSize)))
		b1 := float32(*(*U)(unsafe.Add(bData, uintptr(i+1)*bSize)))
		b2 := float32(*(*U)(unsafe.Add(bData, uintptr(i+2)*bSize)))
		b3 := float32(*(*U)(unsafe.Add(bData, uintptr(i+3)*bSize)))
		b4 := float32(*(*U)(unsafe.Add(bData, uintptr(i+4)*bSize)))
		b5 := float32(*(*U)(unsafe.Add(bData, uintptr(i+5)*bSize)))
		b6 := float32(*(*U)(unsafe.Add(bData, uintptr(i+6)*bSize)))
		b7 := float32(*(*U)(unsafe.Add(bData, uintptr(i+7)*bSize)))

		*(*float32)(unsafe.Add(dstData, uintptr(i+0)*dstSize)) = a0 / b0
		*(*float32)(unsafe.Add(dstData, uintptr(i+1)*dstSize)) = a1 / b1
		*(*float32)(unsafe.Add(dstData, uintptr(i+2)*dstSize)) = a2 / b2
		*(*float32)(unsafe.Add(dstData, uintptr(i+3)*dstSize)) = a3 / b3
		*(*float32)(unsafe.Add(dstData, uintptr(i+4)*dstSize)) = a4 / b4
		*(*float32)(unsafe.Add(dstData, uintptr(i+5)*dstSize)) = a5 / b5
		*(*float32)(unsafe.Add(dstData, uintptr(i+6)*dstSize)) = a6 / b6
		*(*float32)(unsafe.Add(dstData, uintptr(i+7)*dstSize)) = a7 / b7
	}

	// Handle remaining elements
	for ; i < capacity; i++ {
		a := float32(*(*T)(unsafe.Add(aData, uintptr(i)*aSize)))
		b := float32(*(*U)(unsafe.Add(bData, uintptr(i)*bSize)))
		*(*float32)(unsafe.Add(dstData, uintptr(i)*dstSize)) = a / b
	}
}

/*
BlazeElementWiseVectorDivideF64 performs element-wise division of two vectors in float64 precision.

Computes: result[i] = vectorA[i] / vectorB[i] for all elements i.

Use cases:
- Signal processing (normalization, deconvolution)
- Image processing (ratio images)
- Numerical simulations (element-wise scaling)
- Normalization operations

Time complexity: O(n) - single pass through vectors
Space complexity: O(1) - only accumulator variables used

Prerequisites:
- Both input vectors must have the same length
- Output vector must have the same length as inputs
- All vectors must be valid and initialized

Edge cases:
- Works with any numeric type (int, float32, float64, etc.)
- Type conversions occur during computation (input types → float64)
- Division by zero results in ±Inf or NaN (Go's standard behavior)
- No overflow checking (relies on Go's numeric behavior)
*/
func BlazeElementWiseVectorDivideF64[T, U foundation.Numeric](
	vectorAAddr, vectorBAddr, newVectorAddr memcore.MarkRaw,
) {
	aCapacity := memstruct.VectorCapacityGet[T](vectorAAddr)
	bCapacity := memstruct.VectorCapacityGet[U](vectorBAddr)

	if aCapacity != bCapacity {
		panic(fmt.Errorf("cannot perform elementwise divide: capacity mismatch (a=%d, b=%d)",
			aCapacity, bCapacity))
	}

	aData := memstruct.VectorDataPtrGet[T](vectorAAddr)
	bData := memstruct.VectorDataPtrGet[U](vectorBAddr)
	dstData := memstruct.VectorDataPtrGet[float64](newVectorAddr)
	aSize := uintptr(memcore.SizeOf[T]())
	bSize := uintptr(memcore.SizeOf[U]())
	dstSize := uintptr(memcore.SizeOf[float64]())
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

		*(*float64)(unsafe.Add(dstData, uintptr(i+0)*dstSize)) = a0 / b0
		*(*float64)(unsafe.Add(dstData, uintptr(i+1)*dstSize)) = a1 / b1
		*(*float64)(unsafe.Add(dstData, uintptr(i+2)*dstSize)) = a2 / b2
		*(*float64)(unsafe.Add(dstData, uintptr(i+3)*dstSize)) = a3 / b3
		*(*float64)(unsafe.Add(dstData, uintptr(i+4)*dstSize)) = a4 / b4
		*(*float64)(unsafe.Add(dstData, uintptr(i+5)*dstSize)) = a5 / b5
		*(*float64)(unsafe.Add(dstData, uintptr(i+6)*dstSize)) = a6 / b6
		*(*float64)(unsafe.Add(dstData, uintptr(i+7)*dstSize)) = a7 / b7
	}

	// Handle remaining elements
	for ; i < capacity; i++ {
		a := float64(*(*T)(unsafe.Add(aData, uintptr(i)*aSize)))
		b := float64(*(*U)(unsafe.Add(bData, uintptr(i)*bSize)))
		*(*float64)(unsafe.Add(dstData, uintptr(i)*dstSize)) = a / b
	}
}
