package blaze

import (
	"blaze/core"
	"blaze/reduce"
	"blaze/simd"

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
		uint64(10*memcore.MegaByte),
		blazetesting.DoubleGrowth,
	)
	defer memforge.DynamicLinearAllocatorDestroy(allocator)

	simd.BlazeSIMDDispatchInit()

	dimensions := []uint64{3, 128, 132, 384, 768, 1024, 1033}

	for _, dimension := range dimensions {
		t.Run(fmt.Sprintf("Dimension=%d", dimension), func(t *testing.T) {
			memforge.DynamicLinearAllocatorReset(allocator)

			BlazeTestSumF64Path[float64](t, allocator, dimension, rng, core.DTypeF64, 1e-14)
			BlazeTestSumF64Path[float32](t, allocator, dimension, rng, core.DTypeF32, 1e-6)
		})
	}
}

/*
BlazeTestSumF64Path tests the Float64 sum reduction path for a given input type.

This function encapsulates the test setup for comparing optimized kernel execution
against the pure Go fallback implementation. It creates a vector, populates it with
random data, runs both the baseline (optimized) and Go fallback paths, and compares
the results within the specified tolerance.

Time complexity: O(n) - where n is dimension
Space complexity: O(n) - allocates vector and data slice

Prerequisites:
- allocator must be a valid memory allocator
- dimension must be greater than 0
- rng must be initialized

Edge cases:
- Tests both Float64 -> Float64 and Float32 -> Float64 conversion paths
- Uses appropriate epsilon tolerance based on input type precision
*/
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

	sumBaseline := reduce.BlazeReduceVectorSumF64[T](vector)

	oldKernel := simd.BlazeSIMDDispatchKernelRemove(
		core.Blaze_Operation_Vector_Sum,
		core.DTypeF64,
		inputDType,
	)

	sumGo := reduce.BlazeReduceVectorSumF64[T](vector)

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
