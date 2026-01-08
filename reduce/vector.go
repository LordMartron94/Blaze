package reduce

import (
	"fmt"
	"unsafe"

	"blaze/core"
	"blaze/internal"
	"blaze/simd"
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
	data := memstruct.VectorDataPtrGet[T](vector)
	size := uintptr(memcore.SizeOf[T]())
	capacity := memstruct.VectorCapacityGet[T](vector)

	minValue := foundation.MaxValue[T]()
	var i uint64

	// Manually unrolled loop for stride 8
	for ; i+7 < capacity; i += 8 {
		v0 := *(*T)(unsafe.Add(data, uintptr(i+0)*size))
		v1 := *(*T)(unsafe.Add(data, uintptr(i+1)*size))
		v2 := *(*T)(unsafe.Add(data, uintptr(i+2)*size))
		v3 := *(*T)(unsafe.Add(data, uintptr(i+3)*size))
		v4 := *(*T)(unsafe.Add(data, uintptr(i+4)*size))
		v5 := *(*T)(unsafe.Add(data, uintptr(i+5)*size))
		v6 := *(*T)(unsafe.Add(data, uintptr(i+6)*size))
		v7 := *(*T)(unsafe.Add(data, uintptr(i+7)*size))

		if v0 < minValue {
			minValue = v0
		}
		if v1 < minValue {
			minValue = v1
		}
		if v2 < minValue {
			minValue = v2
		}
		if v3 < minValue {
			minValue = v3
		}
		if v4 < minValue {
			minValue = v4
		}
		if v5 < minValue {
			minValue = v5
		}
		if v6 < minValue {
			minValue = v6
		}
		if v7 < minValue {
			minValue = v7
		}
	}

	// Handle remaining elements
	for ; i < capacity; i++ {
		v := *(*T)(unsafe.Add(data, uintptr(i)*size))
		if v < minValue {
			minValue = v
		}
	}

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
	data := memstruct.VectorDataPtrGet[T](vector)
	size := uintptr(memcore.SizeOf[T]())
	capacity := memstruct.VectorCapacityGet[T](vector)

	maxValue := foundation.MinValue[T]()
	var i uint64

	// Manually unrolled loop for stride 8
	for ; i+7 < capacity; i += 8 {
		v0 := *(*T)(unsafe.Add(data, uintptr(i+0)*size))
		v1 := *(*T)(unsafe.Add(data, uintptr(i+1)*size))
		v2 := *(*T)(unsafe.Add(data, uintptr(i+2)*size))
		v3 := *(*T)(unsafe.Add(data, uintptr(i+3)*size))
		v4 := *(*T)(unsafe.Add(data, uintptr(i+4)*size))
		v5 := *(*T)(unsafe.Add(data, uintptr(i+5)*size))
		v6 := *(*T)(unsafe.Add(data, uintptr(i+6)*size))
		v7 := *(*T)(unsafe.Add(data, uintptr(i+7)*size))

		if v0 > maxValue {
			maxValue = v0
		}
		if v1 > maxValue {
			maxValue = v1
		}
		if v2 > maxValue {
			maxValue = v2
		}
		if v3 > maxValue {
			maxValue = v3
		}
		if v4 > maxValue {
			maxValue = v4
		}
		if v5 > maxValue {
			maxValue = v5
		}
		if v6 > maxValue {
			maxValue = v6
		}
		if v7 > maxValue {
			maxValue = v7
		}
	}

	// Handle remaining elements
	for ; i < capacity; i++ {
		v := *(*T)(unsafe.Add(data, uintptr(i)*size))
		if v > maxValue {
			maxValue = v
		}
	}

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
	data := memstruct.VectorDataPtrGet[T](vector)
	size := uintptr(memcore.SizeOf[T]())
	capacity := memstruct.VectorCapacityGet[T](vector)

	minValue := foundation.MaxValue[T]()
	maxValue := foundation.MinValue[T]()
	var i uint64

	// Manually unrolled loop for stride 8
	for ; i+7 < capacity; i += 8 {
		v0 := *(*T)(unsafe.Add(data, uintptr(i+0)*size))
		v1 := *(*T)(unsafe.Add(data, uintptr(i+1)*size))
		v2 := *(*T)(unsafe.Add(data, uintptr(i+2)*size))
		v3 := *(*T)(unsafe.Add(data, uintptr(i+3)*size))
		v4 := *(*T)(unsafe.Add(data, uintptr(i+4)*size))
		v5 := *(*T)(unsafe.Add(data, uintptr(i+5)*size))
		v6 := *(*T)(unsafe.Add(data, uintptr(i+6)*size))
		v7 := *(*T)(unsafe.Add(data, uintptr(i+7)*size))

		if v0 < minValue {
			minValue = v0
		}
		if v0 > maxValue {
			maxValue = v0
		}
		if v1 < minValue {
			minValue = v1
		}
		if v1 > maxValue {
			maxValue = v1
		}
		if v2 < minValue {
			minValue = v2
		}
		if v2 > maxValue {
			maxValue = v2
		}
		if v3 < minValue {
			minValue = v3
		}
		if v3 > maxValue {
			maxValue = v3
		}
		if v4 < minValue {
			minValue = v4
		}
		if v4 > maxValue {
			maxValue = v4
		}
		if v5 < minValue {
			minValue = v5
		}
		if v5 > maxValue {
			maxValue = v5
		}
		if v6 < minValue {
			minValue = v6
		}
		if v6 > maxValue {
			maxValue = v6
		}
		if v7 < minValue {
			minValue = v7
		}
		if v7 > maxValue {
			maxValue = v7
		}
	}

	// Handle remaining elements
	for ; i < capacity; i++ {
		v := *(*T)(unsafe.Add(data, uintptr(i)*size))
		if v < minValue {
			minValue = v
		}
		if v > maxValue {
			maxValue = v
		}
	}

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
func BlazeReduceVectorSumF32[T foundation.Numeric](vector memcore.MarkRaw, output *float32) {
	data := memstruct.VectorDataPtrGet[T](vector)
	n := memstruct.VectorCapacityGet[T](vector)

	frame := internal.CurrentKernelFrame
	frame.Reset().
		WithDim(n, 0, 0).
		WithBuffer(0, data, int64(memcore.SizeOf[T]())).
		WithReturn(0, unsafe.Pointer(output)).
		WithFlags(internal.Flag_Contiguous)

	if memcore.IsAligned(data, 32) {
		frame.Flags |= internal.Flag_Aligned32
	}

	if !simd.BlazeTryExecuteOperation(core.Blaze_Operation_Vector_Sum, frame, core.DTypeF32, core.BlazeDTypeGet[T]()) {
		blazeReduceVectorSumF32Go[T](frame)
	}
}

//go:inline
func blazeReduceVectorSumF32Go[T foundation.Numeric](
	frame *internal.BlazeKernelFrame,
) {
	n := frame.Dim[0]
	if n == 0 {
		return
	}

	ptr := frame.Buffers[0].Ptr
	baseP := unsafe.Pointer(ptr)

	var sum0, sum1 float32
	var i uint64

	elementSize := uintptr(memcore.SizeOf[T]())

	for ; i+7 < n; i += 8 {
		sum0 += float32(*(*T)(unsafe.Add(baseP, uintptr(i+0)*elementSize))) + float32(*(*T)(unsafe.Add(baseP, uintptr(i+1)*elementSize)))
		sum1 += float32(*(*T)(unsafe.Add(baseP, uintptr(i+2)*elementSize))) + float32(*(*T)(unsafe.Add(baseP, uintptr(i+3)*elementSize)))
		sum0 += float32(*(*T)(unsafe.Add(baseP, uintptr(i+4)*elementSize))) + float32(*(*T)(unsafe.Add(baseP, uintptr(i+5)*elementSize)))
		sum1 += float32(*(*T)(unsafe.Add(baseP, uintptr(i+6)*elementSize))) + float32(*(*T)(unsafe.Add(baseP, uintptr(i+7)*elementSize)))
	}

	finalSum := sum0 + sum1

	for ; i < n; i++ {
		finalSum += float32(*(*T)(unsafe.Add(baseP, uintptr(i)*elementSize)))
	}

	if frame.Returns[0] != nil {
		*(*float32)(frame.Returns[0]) = finalSum
	}
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
func BlazeReduceVectorSumF64[T foundation.Numeric](vector memcore.MarkRaw, output *float64) {
	data := memstruct.VectorDataPtrGet[T](vector)
	n := memstruct.VectorCapacityGet[T](vector)

	frame := internal.CurrentKernelFrame
	frame.Reset().
		WithDim(n, 0, 0).
		WithBuffer(0, data, int64(memcore.SizeOf[T]())).
		WithReturn(0, unsafe.Pointer(output)).
		WithFlags(internal.Flag_Contiguous)

	if memcore.IsAligned(data, 32) {
		frame.Flags |= internal.Flag_Aligned32
	}

	if !simd.BlazeTryExecuteOperation(core.Blaze_Operation_Vector_Sum, frame, core.DTypeF64, core.BlazeDTypeGet[T]()) {
		blazeReduceVectorSumF64Go[T](frame)
	}
}

//go:inline
func blazeReduceVectorSumF64Go[T foundation.Numeric](
	frame *internal.BlazeKernelFrame,
) {
	n := frame.Dim[0]
	if n == 0 {
		return
	}

	ptr := frame.Buffers[0].Ptr
	baseP := unsafe.Pointer(ptr)

	var sum0, sum1 float64
	var i uint64

	elementSize := uintptr(memcore.SizeOf[T]())

	for ; i+7 < n; i += 8 {
		sum0 += float64(*(*T)(unsafe.Add(baseP, uintptr(i+0)*elementSize))) + float64(*(*T)(unsafe.Add(baseP, uintptr(i+1)*elementSize)))
		sum1 += float64(*(*T)(unsafe.Add(baseP, uintptr(i+2)*elementSize))) + float64(*(*T)(unsafe.Add(baseP, uintptr(i+3)*elementSize)))
		sum0 += float64(*(*T)(unsafe.Add(baseP, uintptr(i+4)*elementSize))) + float64(*(*T)(unsafe.Add(baseP, uintptr(i+5)*elementSize)))
		sum1 += float64(*(*T)(unsafe.Add(baseP, uintptr(i+6)*elementSize))) + float64(*(*T)(unsafe.Add(baseP, uintptr(i+7)*elementSize)))
	}

	finalSum := sum0 + sum1

	for ; i < n; i++ {
		finalSum += float64(*(*T)(unsafe.Add(baseP, uintptr(i)*elementSize)))
	}

	if frame.Returns[0] != nil {
		*(*float64)(frame.Returns[0]) = finalSum
	}
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
	data := memstruct.VectorDataPtrGet[T](vector)
	size := uintptr(memcore.SizeOf[T]())
	capacity := memstruct.VectorCapacityGet[T](vector)

	return blazeReduceVectorSumSquaredF32Go[T](data, size, capacity)
}

//go:inline
func blazeReduceVectorSumSquaredF32Go[T foundation.Numeric](
	data unsafe.Pointer, size uintptr, capacity uint64,
) float32 {
	var sqrSum float32
	var i uint64

	// Manually unrolled loop for stride 8
	for ; i+7 < capacity; i += 8 {
		v0 := float32(*(*T)(unsafe.Add(data, uintptr(i+0)*size)))
		v1 := float32(*(*T)(unsafe.Add(data, uintptr(i+1)*size)))
		v2 := float32(*(*T)(unsafe.Add(data, uintptr(i+2)*size)))
		v3 := float32(*(*T)(unsafe.Add(data, uintptr(i+3)*size)))
		v4 := float32(*(*T)(unsafe.Add(data, uintptr(i+4)*size)))
		v5 := float32(*(*T)(unsafe.Add(data, uintptr(i+5)*size)))
		v6 := float32(*(*T)(unsafe.Add(data, uintptr(i+6)*size)))
		v7 := float32(*(*T)(unsafe.Add(data, uintptr(i+7)*size)))

		sqrSum += v0*v0 + v1*v1 + v2*v2 + v3*v3 + v4*v4 + v5*v5 + v6*v6 + v7*v7
	}

	// Handle remaining elements
	for ; i < capacity; i++ {
		v := float32(*(*T)(unsafe.Add(data, uintptr(i)*size)))
		sqrSum += v * v
	}

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
	data := memstruct.VectorDataPtrGet[T](vector)
	size := uintptr(memcore.SizeOf[T]())
	capacity := memstruct.VectorCapacityGet[T](vector)

	return blazeReduceVectorSumSquaredF64Go[T](data, size, capacity)
}

//go:inline
func blazeReduceVectorSumSquaredF64Go[T foundation.Numeric](
	data unsafe.Pointer, size uintptr, capacity uint64,
) float64 {
	var sqrSum float64
	var i uint64

	// Manually unrolled loop for stride 8
	for ; i+7 < capacity; i += 8 {
		v0 := float64(*(*T)(unsafe.Add(data, uintptr(i+0)*size)))
		v1 := float64(*(*T)(unsafe.Add(data, uintptr(i+1)*size)))
		v2 := float64(*(*T)(unsafe.Add(data, uintptr(i+2)*size)))
		v3 := float64(*(*T)(unsafe.Add(data, uintptr(i+3)*size)))
		v4 := float64(*(*T)(unsafe.Add(data, uintptr(i+4)*size)))
		v5 := float64(*(*T)(unsafe.Add(data, uintptr(i+5)*size)))
		v6 := float64(*(*T)(unsafe.Add(data, uintptr(i+6)*size)))
		v7 := float64(*(*T)(unsafe.Add(data, uintptr(i+7)*size)))

		sqrSum += v0*v0 + v1*v1 + v2*v2 + v3*v3 + v4*v4 + v5*v5 + v6*v6 + v7*v7
	}

	// Handle remaining elements
	for ; i < capacity; i++ {
		v := float64(*(*T)(unsafe.Add(data, uintptr(i)*size)))
		sqrSum += v * v
	}

	return sqrSum
}

/*
BlazeReduceVectorDotProductF32 computes the dot product (inner product) of two vectors in float32 precision.

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
func BlazeReduceVectorDotProductF32[T, U foundation.Numeric](aAddr, bAddr memcore.MarkRaw) float32 {
	aCapacity := memstruct.VectorCapacityGet[T](aAddr)
	bCapacity := memstruct.VectorCapacityGet[U](bAddr)

	if aCapacity != bCapacity {
		panic(fmt.Errorf("cannot compute dot product: capacity mismatch (a=%d, b=%d)",
			aCapacity, bCapacity))
	}

	aData := memstruct.VectorDataPtrGet[T](aAddr)
	bData := memstruct.VectorDataPtrGet[U](bAddr)
	aSize := uintptr(memcore.SizeOf[T]())
	bSize := uintptr(memcore.SizeOf[U]())
	capacity := aCapacity

	return blazeReduceDotProductF32Go[T, U](aData, bData, aSize, bSize, capacity)
}

//go:inline
func blazeReduceDotProductF32Go[T, U foundation.Numeric](
	aData, bData unsafe.Pointer, aSize, bSize uintptr, capacity uint64,
) float32 {
	var dot float32
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

		dot += a0*b0 + a1*b1 + a2*b2 + a3*b3 + a4*b4 + a5*b5 + a6*b6 + a7*b7
	}

	// Handle remaining elements
	for ; i < capacity; i++ {
		a := float32(*(*T)(unsafe.Add(aData, uintptr(i)*aSize)))
		b := float32(*(*U)(unsafe.Add(bData, uintptr(i)*bSize)))
		dot += a * b
	}

	return dot
}

/*
BlazeReduceVectorDotProductF64 computes the dot product (inner product) of two vectors in float64 precision.

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
func BlazeReduceVectorDotProductF64[T, U foundation.Numeric](aAddr, bAddr memcore.MarkRaw, output *float64) {
	aData := memstruct.VectorDataPtrGet[T](aAddr)
	bData := memstruct.VectorDataPtrGet[U](bAddr)
	n := memstruct.VectorCapacityGet[T](aAddr)

	// Validation: Dot product requires equal lengths.
	if n != memstruct.VectorCapacityGet[U](bAddr) {
		panic(fmt.Errorf("dot product capacity mismatch: %d vs %d", n, memstruct.VectorCapacityGet[U](bAddr)))
	}

	frame := internal.CurrentKernelFrame
	frame.Reset().
		WithDim(n, 0, 0).
		WithBuffer(0, aData, int64(memcore.SizeOf[T]())).
		WithBuffer(1, bData, int64(memcore.SizeOf[U]())).
		WithReturn(0, unsafe.Pointer(output)).
		WithFlags(internal.Flag_Contiguous)

	// Optimization: Check alignment for both buffers
	if memcore.IsAligned(aData, 32) && memcore.IsAligned(bData, 32) {
		frame.Flags |= internal.Flag_Aligned32
	}

	// Attempt SIMD execution; fallback to Go if unsupported
	if !simd.BlazeTryExecuteOperation(core.Blaze_Operation_Vector_Dot, frame, core.DTypeF64, core.BlazeDTypeGet[T](), core.BlazeDTypeGet[U]()) {
		blazeReduceDotProductF64Go[T, U](frame)
	}
}

//go:inline
func blazeReduceDotProductF64Go[T, U foundation.Numeric](
	frame *internal.BlazeKernelFrame,
) {
	n := frame.Dim[0]
	if n == 0 {
		return
	}

	ptrA := frame.Buffers[0].Ptr
	ptrB := frame.Buffers[1].Ptr
	strideA := uintptr(frame.Buffers[0].Stride)
	strideB := uintptr(frame.Buffers[1].Stride)

	var dot0, dot1 float64
	var i uintptr
	nCnv := uintptr(n)

	// Unrolled loop: 8 elements per iteration using dual accumulators
	for ; i+7 < nCnv; i += 8 {
		dot0 += float64(*(*T)(unsafe.Add(ptrA, (i+0)*strideA))) * float64(*(*U)(unsafe.Add(ptrB, (i+0)*strideB)))
		dot1 += float64(*(*T)(unsafe.Add(ptrA, (i+1)*strideA))) * float64(*(*U)(unsafe.Add(ptrB, (i+1)*strideB)))
		dot0 += float64(*(*T)(unsafe.Add(ptrA, (i+2)*strideA))) * float64(*(*U)(unsafe.Add(ptrB, (i+2)*strideB)))
		dot1 += float64(*(*T)(unsafe.Add(ptrA, (i+3)*strideA))) * float64(*(*U)(unsafe.Add(ptrB, (i+3)*strideB)))
		dot0 += float64(*(*T)(unsafe.Add(ptrA, (i+4)*strideA))) * float64(*(*U)(unsafe.Add(ptrB, (i+4)*strideB)))
		dot1 += float64(*(*T)(unsafe.Add(ptrA, (i+5)*strideA))) * float64(*(*U)(unsafe.Add(ptrB, (i+5)*strideB)))
		dot0 += float64(*(*T)(unsafe.Add(ptrA, (i+6)*strideA))) * float64(*(*U)(unsafe.Add(ptrB, (i+6)*strideB)))
		dot1 += float64(*(*T)(unsafe.Add(ptrA, (i+7)*strideA))) * float64(*(*U)(unsafe.Add(ptrB, (i+7)*strideB)))
	}

	accum := dot0 + dot1

	// Handle tail
	for ; i < nCnv; i++ {
		accum += float64(*(*T)(unsafe.Add(ptrA, i*strideA))) * float64(*(*U)(unsafe.Add(ptrB, i*strideB)))
	}

	if frame.Returns[0] != nil {
		*(*float64)(frame.Returns[0]) = accum
	}
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
	var sum float32
	BlazeReduceVectorSumF32[T](vector, &sum)
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
	var sum float64
	BlazeReduceVectorSumF64[T](vector, &sum)
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
	aCapacity := memstruct.VectorCapacityGet[T](vectorA)
	bCapacity := memstruct.VectorCapacityGet[U](vectorB)

	if aCapacity != bCapacity {
		panic(fmt.Errorf("cannot compute covariance: capacity mismatch (a=%d, b=%d)",
			aCapacity, bCapacity))
	}

	aData := memstruct.VectorDataPtrGet[T](vectorA)
	bData := memstruct.VectorDataPtrGet[U](vectorB)
	aSize := uintptr(memcore.SizeOf[T]())
	bSize := uintptr(memcore.SizeOf[U]())
	capacity := aCapacity

	var covariance float32
	var i uint64

	// Manually unrolled loop for stride 8
	for ; i+7 < capacity; i += 8 {
		a0 := float32(*(*T)(unsafe.Add(aData, uintptr(i+0)*aSize))) - meanA
		a1 := float32(*(*T)(unsafe.Add(aData, uintptr(i+1)*aSize))) - meanA
		a2 := float32(*(*T)(unsafe.Add(aData, uintptr(i+2)*aSize))) - meanA
		a3 := float32(*(*T)(unsafe.Add(aData, uintptr(i+3)*aSize))) - meanA
		a4 := float32(*(*T)(unsafe.Add(aData, uintptr(i+4)*aSize))) - meanA
		a5 := float32(*(*T)(unsafe.Add(aData, uintptr(i+5)*aSize))) - meanA
		a6 := float32(*(*T)(unsafe.Add(aData, uintptr(i+6)*aSize))) - meanA
		a7 := float32(*(*T)(unsafe.Add(aData, uintptr(i+7)*aSize))) - meanA

		b0 := float32(*(*U)(unsafe.Add(bData, uintptr(i+0)*bSize))) - meanB
		b1 := float32(*(*U)(unsafe.Add(bData, uintptr(i+1)*bSize))) - meanB
		b2 := float32(*(*U)(unsafe.Add(bData, uintptr(i+2)*bSize))) - meanB
		b3 := float32(*(*U)(unsafe.Add(bData, uintptr(i+3)*bSize))) - meanB
		b4 := float32(*(*U)(unsafe.Add(bData, uintptr(i+4)*bSize))) - meanB
		b5 := float32(*(*U)(unsafe.Add(bData, uintptr(i+5)*bSize))) - meanB
		b6 := float32(*(*U)(unsafe.Add(bData, uintptr(i+6)*bSize))) - meanB
		b7 := float32(*(*U)(unsafe.Add(bData, uintptr(i+7)*bSize))) - meanB

		covariance += a0*b0 + a1*b1 + a2*b2 + a3*b3 + a4*b4 + a5*b5 + a6*b6 + a7*b7
	}

	// Handle remaining elements
	for ; i < capacity; i++ {
		devA := float32(*(*T)(unsafe.Add(aData, uintptr(i)*aSize))) - meanA
		devB := float32(*(*U)(unsafe.Add(bData, uintptr(i)*bSize))) - meanB
		covariance += devA * devB
	}

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
	aCapacity := memstruct.VectorCapacityGet[T](vectorA)
	bCapacity := memstruct.VectorCapacityGet[U](vectorB)

	if aCapacity != bCapacity {
		panic(fmt.Errorf("cannot compute covariance: capacity mismatch (a=%d, b=%d)",
			aCapacity, bCapacity))
	}

	aData := memstruct.VectorDataPtrGet[T](vectorA)
	bData := memstruct.VectorDataPtrGet[U](vectorB)
	aSize := uintptr(memcore.SizeOf[T]())
	bSize := uintptr(memcore.SizeOf[U]())
	capacity := aCapacity

	var covariance float64
	var i uint64

	// Manually unrolled loop for stride 8
	for ; i+7 < capacity; i += 8 {
		a0 := float64(*(*T)(unsafe.Add(aData, uintptr(i+0)*aSize))) - meanA
		a1 := float64(*(*T)(unsafe.Add(aData, uintptr(i+1)*aSize))) - meanA
		a2 := float64(*(*T)(unsafe.Add(aData, uintptr(i+2)*aSize))) - meanA
		a3 := float64(*(*T)(unsafe.Add(aData, uintptr(i+3)*aSize))) - meanA
		a4 := float64(*(*T)(unsafe.Add(aData, uintptr(i+4)*aSize))) - meanA
		a5 := float64(*(*T)(unsafe.Add(aData, uintptr(i+5)*aSize))) - meanA
		a6 := float64(*(*T)(unsafe.Add(aData, uintptr(i+6)*aSize))) - meanA
		a7 := float64(*(*T)(unsafe.Add(aData, uintptr(i+7)*aSize))) - meanA

		b0 := float64(*(*U)(unsafe.Add(bData, uintptr(i+0)*bSize))) - meanB
		b1 := float64(*(*U)(unsafe.Add(bData, uintptr(i+1)*bSize))) - meanB
		b2 := float64(*(*U)(unsafe.Add(bData, uintptr(i+2)*bSize))) - meanB
		b3 := float64(*(*U)(unsafe.Add(bData, uintptr(i+3)*bSize))) - meanB
		b4 := float64(*(*U)(unsafe.Add(bData, uintptr(i+4)*bSize))) - meanB
		b5 := float64(*(*U)(unsafe.Add(bData, uintptr(i+5)*bSize))) - meanB
		b6 := float64(*(*U)(unsafe.Add(bData, uintptr(i+6)*bSize))) - meanB
		b7 := float64(*(*U)(unsafe.Add(bData, uintptr(i+7)*bSize))) - meanB

		covariance += a0*b0 + a1*b1 + a2*b2 + a3*b3 + a4*b4 + a5*b5 + a6*b6 + a7*b7
	}

	// Handle remaining elements
	for ; i < capacity; i++ {
		devA := float64(*(*T)(unsafe.Add(aData, uintptr(i)*aSize))) - meanA
		devB := float64(*(*U)(unsafe.Add(bData, uintptr(i)*bSize))) - meanB
		covariance += devA * devB
	}

	return covariance
}

/* BlazeReduceSpeedOfLight is now a "Hardened" calibration function. */
func BlazeReduceSpeedOfLight(frame *internal.BlazeKernelFrame) uint64 {
	frame.Reset()
	frame.WithDim(1, 0, 0)

	if !simd.BlazeTryExecuteOperation(core.Blaze_Operation_SpeedOfLight, frame, core.DTypeNone, core.DTypeNone) {
		panic("dispatch failed")
	}

	return frame.Dim[0]
}

/*
BlazeReduceMemoryThroughput measures memory throughput by performing sequential memory loads without computation.

This function uses the SpeedOfLightTest_Throughput assembly kernel which loads memory
into AVX2 registers without performing any arithmetic operations, making it ideal for
measuring raw memory bandwidth.

Use cases:
- Memory bandwidth benchmarking
- Cache performance analysis
- Memory subsystem stress testing

Time complexity: O(n) - single pass through vector
Space complexity: O(1) - no additional allocations

Prerequisites:
- Vector must be valid and initialized
- SIMD dispatch must be initialized

Edge cases:
- Works with any numeric type
- Requires AVX2 support for optimal performance
- Panics if kernel dispatch fails
*/
func BlazeReduceMemoryThroughput[T foundation.Numeric](vector memcore.MarkRaw) {
	data := memstruct.VectorDataPtrGet[T](vector)
	capacity := memstruct.VectorCapacityGet[T](vector)

	frame := internal.CurrentKernelFrame
	frame.Reset().
		WithDim(capacity, 0, 0).
		WithBuffer(0, data, int64(memcore.SizeOf[T]())).
		WithFlags(internal.Flag_Contiguous)

	if memcore.IsAligned(data, 32) {
		frame.Flags |= internal.Flag_Aligned32
	}

	if !simd.BlazeTryExecuteOperation(core.Blaze_Operation_SpeedOfLightThroughput, frame, core.DTypeNone, core.BlazeDTypeGet[T]()) {
		panic("dispatch failed")
	}
}
