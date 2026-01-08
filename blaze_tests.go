package blaze

import (
	"blaze/core"
	"blaze/reduce"
	"blaze/scalar"
	"blaze/simd"
	"unsafe"

	blazetesting "blaze/testing"
	"echo"
	"fmt"
	"foundation"
	foundationtesting "foundation/testing"
	"math/rand"
	"memarch"
	"memcore"
	"memforge"
	"memstruct"
	"testing"
)

func BlazeTestSuite(t *testing.T) {
	rng := rand.New(rand.NewSource(42))

	allocator := memforge.DynamicLinearAllocatorCreateFunction(
		uint64(20*memcore.MegaByte),
		blazetesting.DoubleGrowth,
	)
	defer memforge.DynamicLinearAllocatorDestroy(allocator)

	simd.BlazeSIMDDispatchInit()

	dimensions := []uint64{3, 128, 132, 384, 768, 1024, 1033}

	for _, dimension := range dimensions {
		t.Run(fmt.Sprintf("Dimension=%d", dimension), func(t *testing.T) {
			memforge.DynamicLinearAllocatorReset(allocator)

			// --- Sum Tests ---
			BlazeTestSumF64Path[float64](t, allocator, dimension, rng, core.DTypeF64, 1e-13)
			BlazeTestSumF64Path[float32](t, allocator, dimension, rng, core.DTypeF32, 1e-5)
			BlazeTestSumF32Path[float32](t, allocator, dimension, rng, core.DTypeF32, 1e-5)

			// --- Dot Product Tests ---
			// F64 · F64 -> F64
			BlazeTestDotProductF64Path[float64, float64](
				t, allocator, dimension, rng, core.DTypeF64, core.DTypeF64, 1e-13,
			)

			// F32 · F32 -> F64
			BlazeTestDotProductF64Path[float32, float32](
				t, allocator, dimension, rng, core.DTypeF32, core.DTypeF32, 1e-13,
			)

			// F32 · F64 -> F64
			BlazeTestDotProductF64Path[float32, float64](
				t, allocator, dimension, rng, core.DTypeF32, core.DTypeF64, 1e-13,
			)

			// --- Scalar Divide Tests ---
			// F64 / Scalar -> F64
			BlazeTestScalarVectorDivideF64Path[float64](
				t, allocator, dimension, rng, core.DTypeF64, 1e-13,
			)
			// F32 / Scalar -> F64
			BlazeTestScalarVectorDivideF64Path[float32](
				t, allocator, dimension, rng, core.DTypeF32, 1e-5,
			)

			// --- Scalar Multiply Tests ---
			// F64 * Scalar -> F64
			BlazeTestScalarVectorMultiplyF64Path[float64](
				t, allocator, dimension, rng, core.DTypeF64, 1e-13,
			)
			// F32 * Scalar -> F64
			BlazeTestScalarVectorMultiplyF64Path[float32](
				t, allocator, dimension, rng, core.DTypeF32, 1e-5,
			)
		})
	}
}

func BlazeTestScalarVectorMultiplyF64Path[T foundation.Numeric](
	t *testing.T,
	allocator memcore.MarkRaw,
	dimension uint64,
	rng *rand.Rand,
	inputDType core.BlazeDType,
	epsilon float64,
) {
	t.Helper()

	// 1. Create Source Vector
	vectorSrc, _ := memarch.MemArchVectorCreate[T](
		func(sizeBytes, alignment uint64) memcore.MarkRaw {
			return memforge.DynamicLinearAllocatorMallocUnsafe(allocator, sizeBytes, alignment)
		},
		dimension,
	)

	// 2. Create Destination Vectors (one for ASM, one for Go)
	vectorDstASM, _ := memarch.MemArchVectorCreate[float64](
		func(sizeBytes, alignment uint64) memcore.MarkRaw {
			return memforge.DynamicLinearAllocatorMallocUnsafe(allocator, sizeBytes, alignment)
		},
		dimension,
	)
	vectorDstGo, _ := memarch.MemArchVectorCreate[float64](
		func(sizeBytes, alignment uint64) memcore.MarkRaw {
			return memforge.DynamicLinearAllocatorMallocUnsafe(allocator, sizeBytes, alignment)
		},
		dimension,
	)

	// 3. Populate Source Data
	var zeroT T
	switch any(zeroT).(type) {
	case float64:
		memstruct.VectorSetFromSlice(vectorSrc, blazetesting.GenerateRandomVectorF64(dimension, rng))
	case float32:
		memstruct.VectorSetFromSlice(vectorSrc, blazetesting.GenerateRandomVectorF32(dimension, rng))
	default:
		t.Fatalf("unsupported type for test")
	}

	// 4. Generate Random Scalar
	randomScalar := rng.Float64()

	// 5. Run Baseline (ASM)
	scalar.BlazeScalarVectorMultiplyF64[T](vectorSrc, vectorDstASM, randomScalar)

	// 6. Force Fallback: Remove Kernel
	// Op: Vector_Scalar_Mul, Out: F64, In: inputDType
	oldKernel := simd.BlazeSIMDDispatchKernelRemove(
		core.Blaze_Operation_Vector_Scalar_Mul,
		core.DTypeF64,
		inputDType,
	)

	// 7. Run Go Fallback
	scalar.BlazeScalarVectorMultiplyF64[T](vectorSrc, vectorDstGo, randomScalar)

	// 8. Restore Kernel
	simd.BlazeSIMDDispatchKernelOverride(
		core.Blaze_Operation_Vector_Scalar_Mul,
		core.DTypeF64,
		oldKernel,
		inputDType,
	)

	// 9. Compare Vectors Element-wise
	ptrASM := memstruct.VectorDataPtrGet[float64](vectorDstASM)
	ptrGo := memstruct.VectorDataPtrGet[float64](vectorDstGo)

	sigStr := formatTestSignature(core.DTypeF64, inputDType)

	for i := uint64(0); i < dimension; i++ {
		valASM := *(*float64)(unsafe.Add(ptrASM, uintptr(i)*8))
		valGo := *(*float64)(unsafe.Add(ptrGo, uintptr(i)*8))

		AssertClose(t, valGo, valASM, epsilon, fmt.Sprintf("%s[idx=%d]", sigStr, i))
	}

	echo.On(core.BlazeUUID).
		Field("sig", sigStr).
		Field("dimension", dimension).
		Debug("tested scalar multiply")
}

/*
BlazeTestScalarVectorDivideF64Path tests the Scalar Vector Divide operation.
It runs the operation (Src / Scalar) using the ASM kernel, then forces the Go fallback,
and compares the resulting vectors element-by-element.
*/
func BlazeTestScalarVectorDivideF64Path[T foundation.Numeric](
	t *testing.T,
	allocator memcore.MarkRaw,
	dimension uint64,
	rng *rand.Rand,
	inputDType core.BlazeDType,
	epsilon float64,
) {
	t.Helper()

	// 1. Create Source Vector
	vectorSrc, _ := memarch.MemArchVectorCreate[T](
		func(sizeBytes, alignment uint64) memcore.MarkRaw {
			return memforge.DynamicLinearAllocatorMallocUnsafe(allocator, sizeBytes, alignment)
		},
		dimension,
	)

	// 2. Create Destination Vectors (one for ASM, one for Go)
	vectorDstASM, _ := memarch.MemArchVectorCreate[float64](
		func(sizeBytes, alignment uint64) memcore.MarkRaw {
			return memforge.DynamicLinearAllocatorMallocUnsafe(allocator, sizeBytes, alignment)
		},
		dimension,
	)
	vectorDstGo, _ := memarch.MemArchVectorCreate[float64](
		func(sizeBytes, alignment uint64) memcore.MarkRaw {
			return memforge.DynamicLinearAllocatorMallocUnsafe(allocator, sizeBytes, alignment)
		},
		dimension,
	)

	// 3. Populate Source Data
	var zeroT T
	switch any(zeroT).(type) {
	case float64:
		memstruct.VectorSetFromSlice(vectorSrc, blazetesting.GenerateRandomVectorF64(dimension, rng))
	case float32:
		memstruct.VectorSetFromSlice(vectorSrc, blazetesting.GenerateRandomVectorF32(dimension, rng))
	default:
		t.Fatalf("unsupported type for test")
	}

	// 4. Generate Random Scalar (Avoid 0.0 to prevent Inf comparison headaches)
	randomScalar := rng.Float64() + 0.1

	// 5. Run Baseline (ASM)
	// Assuming the function is exposed in 'vector_ops' or similar package
	scalar.BlazeScalarVectorDivideF64[T](vectorSrc, vectorDstASM, randomScalar)

	// 6. Force Fallback: Remove Kernel
	// Op: Vector_Scalar_Div, Out: F64, In: inputDType
	oldKernel := simd.BlazeSIMDDispatchKernelRemove(
		core.Blaze_Operation_Vector_Scalar_Div,
		core.DTypeF64,
		inputDType,
	)

	// 7. Run Go Fallback
	scalar.BlazeScalarVectorDivideF64[T](vectorSrc, vectorDstGo, randomScalar)

	// 8. Restore Kernel
	simd.BlazeSIMDDispatchKernelOverride(
		core.Blaze_Operation_Vector_Scalar_Div,
		core.DTypeF64,
		oldKernel,
		inputDType,
	)

	// 9. Compare Vectors Element-wise
	ptrASM := memstruct.VectorDataPtrGet[float64](vectorDstASM)
	ptrGo := memstruct.VectorDataPtrGet[float64](vectorDstGo)

	sigStr := formatTestSignature(core.DTypeF64, inputDType)

	for i := uint64(0); i < dimension; i++ {
		valASM := *(*float64)(unsafe.Add(ptrASM, uintptr(i)*8))
		valGo := *(*float64)(unsafe.Add(ptrGo, uintptr(i)*8))

		AssertClose(t, valGo, valASM, epsilon, fmt.Sprintf("%s[idx=%d]", sigStr, i))
	}

	echo.On(core.BlazeUUID).
		Field("sig", sigStr).
		Field("dimension", dimension).
		Debug("tested scalar divide")
}

/*
BlazeTestDotProductF64Path tests the Float64 dot product reduction.

It creates two vectors (A and B), populates them with random data, and compares
the optimized SIMD kernel execution against the Go fallback.
*/
func BlazeTestDotProductF64Path[T, U foundation.Numeric](
	t *testing.T,
	allocator memcore.MarkRaw,
	dimension uint64,
	rng *rand.Rand,
	dtypeA core.BlazeDType,
	dtypeB core.BlazeDType,
	epsilon float64,
) {
	t.Helper()

	// 1. Create Vector A
	vectorA, _ := memarch.MemArchVectorCreate[T](
		func(sizeBytes, alignment uint64) memcore.MarkRaw {
			return memforge.DynamicLinearAllocatorMallocUnsafe(allocator, sizeBytes, alignment)
		},
		dimension,
	)

	// 2. Create Vector B
	vectorB, _ := memarch.MemArchVectorCreate[U](
		func(sizeBytes, alignment uint64) memcore.MarkRaw {
			return memforge.DynamicLinearAllocatorMallocUnsafe(allocator, sizeBytes, alignment)
		},
		dimension,
	)

	// 3. Populate Vectors
	// We handle T and U separately to allow mixed-type tests later.
	var zeroT T
	switch any(zeroT).(type) {
	case float64:
		memstruct.VectorSetFromSlice(vectorA, blazetesting.GenerateRandomVectorF64(dimension, rng))
	case float32:
		memstruct.VectorSetFromSlice(vectorA, blazetesting.GenerateRandomVectorF32(dimension, rng))
	default:
		t.Fatalf("unsupported type A for test")
	}

	var zeroU U
	switch any(zeroU).(type) {
	case float64:
		memstruct.VectorSetFromSlice(vectorB, blazetesting.GenerateRandomVectorF64(dimension, rng))
	case float32:
		memstruct.VectorSetFromSlice(vectorB, blazetesting.GenerateRandomVectorF32(dimension, rng))
	default:
		t.Fatalf("unsupported type B for test")
	}

	// 4. Run Baseline (Optimized)
	var dotBaseline float64
	reduce.BlazeReduceVectorDotProductF64[T, U](vectorA, vectorB, &dotBaseline)

	// 5. Force Fallback: Remove Kernel
	// The registry keys for Dot Product are (Op, Out, InA, InB)
	oldKernel := simd.BlazeSIMDDispatchKernelRemove(
		core.Blaze_Operation_Vector_Dot,
		core.DTypeF64,
		dtypeA,
		dtypeB,
	)

	// 6. Run Go Fallback
	var dotGo float64
	reduce.BlazeReduceVectorDotProductF64[T, U](vectorA, vectorB, &dotGo)

	// 7. Restore Kernel
	simd.BlazeSIMDDispatchKernelOverride(
		core.Blaze_Operation_Vector_Dot,
		core.DTypeF64,
		oldKernel,
		dtypeA,
		dtypeB,
	)

	// 8. Assert
	sigStr := formatTestSignatureBinary(core.DTypeF64, dtypeA, dtypeB)
	echo.On(core.BlazeUUID).
		Field("sig", sigStr).
		Field("dimension", dimension).
		Debug("testing signature")

	AssertClose(t, dotGo, dotBaseline, epsilon, sigStr)
}

// --- Helpers ---

// BlazeTestSumF64Path tests the Float64 sum reduction path...
func BlazeTestSumF64Path[T foundation.Numeric](
	t *testing.T,
	allocator memcore.MarkRaw,
	dimension uint64,
	rng *rand.Rand,
	inputDType core.BlazeDType,
	epsilon float64,
) {
	t.Helper()

	vector, _ := memarch.MemArchVectorCreate[T](
		func(sizeBytes, alignment uint64) memcore.MarkRaw {
			return memforge.DynamicLinearAllocatorMallocUnsafe(
				allocator, sizeBytes, alignment,
			)
		},
		dimension,
	)

	var zero T
	switch any(zero).(type) {
	case float64:
		values := blazetesting.GenerateRandomVectorF64(dimension, rng)
		memstruct.VectorSetFromSlice(vector, values)
	case float32:
		values := blazetesting.GenerateRandomVectorF32(dimension, rng)
		memstruct.VectorSetFromSlice(vector, values)
	default:
		t.Fatalf("unsupported type for test")
		return
	}

	var sumBaseline float64
	reduce.BlazeReduceVectorSumF64[T](vector, &sumBaseline)

	oldKernel := simd.BlazeSIMDDispatchKernelRemove(
		core.Blaze_Operation_Vector_Sum,
		core.DTypeF64,
		inputDType,
	)

	var sumGo float64
	reduce.BlazeReduceVectorSumF64[T](vector, &sumGo)

	simd.BlazeSIMDDispatchKernelOverride(
		core.Blaze_Operation_Vector_Sum,
		core.DTypeF64,
		oldKernel,
		inputDType,
	)

	sigStr := formatTestSignature(core.DTypeF64, inputDType)
	echo.On(core.BlazeUUID).
		Field("sig", sigStr).
		Field("dimension", dimension).
		Debug("testing signature")

	AssertClose(t, sumBaseline, sumGo, epsilon, sigStr)
}

func BlazeTestSumF32Path[T foundation.Numeric](
	t *testing.T,
	allocator memcore.MarkRaw,
	dimension uint64,
	rng *rand.Rand,
	inputDType core.BlazeDType,
	epsilon float64,
) {
	t.Helper()

	vector, _ := memarch.MemArchVectorCreate[T](
		func(sizeBytes, alignment uint64) memcore.MarkRaw {
			return memforge.DynamicLinearAllocatorMallocUnsafe(
				allocator, sizeBytes, alignment,
			)
		},
		dimension,
	)

	var zero T
	switch any(zero).(type) {
	case float32:
		values := blazetesting.GenerateRandomVectorF32(dimension, rng)
		memstruct.VectorSetFromSlice(vector, values)
	default:
		t.Fatalf("unsupported input type for F32 sum test")
		return
	}

	var sumBaseline float32
	reduce.BlazeReduceVectorSumF32[T](vector, &sumBaseline)

	oldKernel := simd.BlazeSIMDDispatchKernelRemove(
		core.Blaze_Operation_Vector_Sum,
		core.DTypeF32,
		inputDType,
	)

	var sumGo float32
	reduce.BlazeReduceVectorSumF32[T](vector, &sumGo)

	simd.BlazeSIMDDispatchKernelOverride(
		core.Blaze_Operation_Vector_Sum,
		core.DTypeF32,
		oldKernel,
		inputDType,
	)

	sigStr := formatTestSignature(core.DTypeF32, inputDType)
	echo.On(core.BlazeUUID).
		Field("sig", sigStr).
		Field("dimension", dimension).
		Debug("testing signature")

	AssertClose(t, float64(sumGo), float64(sumBaseline), epsilon, sigStr)
}

func AssertClose(t *testing.T, sumGo, sumAsm float64, epsilon float64, sigStr string) {
	t.Helper()

	diff := sumGo - sumAsm
	if diff < 0 {
		diff = -diff
	}
	absGo := sumGo
	if absGo < 0 {
		absGo = -absGo
	}

	cond := diff <= epsilon || (absGo > 0 && diff/absGo <= epsilon)

	foundationtesting.Assert(
		cond,
		fmt.Sprintf("mismatch outside tolerance (%e) [sig=%s], go=%.20f, asm=%.20f, diff=%.20e", epsilon, sigStr, sumGo, sumAsm, diff),
		fmt.Sprintf("go & asm matched within tolerance [sig=%s], go=%.20f, asm=%.20f, diff=%.20e", sigStr, sumGo, sumAsm, diff),
		t,
	)
}

func formatTestSignature(out core.BlazeDType, in core.BlazeDType) string {
	outStr := formatDTypeString(out)
	inStr := formatDTypeString(in)
	return outStr + "<-" + inStr
}

// formatTestSignatureBinary handles 2 input types
func formatTestSignatureBinary(out core.BlazeDType, inA, inB core.BlazeDType) string {
	outStr := formatDTypeString(out)
	inAStr := formatDTypeString(inA)
	inBStr := formatDTypeString(inB)
	return outStr + "<-(" + inAStr + "·" + inBStr + ")"
}

func formatDTypeString(dt core.BlazeDType) string {
	switch dt {
	case core.DTypeF64:
		return "F64"
	case core.DTypeF32:
		return "F32"
	case core.DTypeI64:
		return "I64"
	case core.DTypeI32:
		return "I32"
	case core.DTypeI16:
		return "I16"
	case core.DTypeI8:
		return "I8"
	case core.DTypeU64:
		return "U64"
	case core.DTypeU32:
		return "U32"
	case core.DTypeU16:
		return "U16"
	case core.DTypeU8:
		return "U8"
	default:
		return "Unknown"
	}
}
