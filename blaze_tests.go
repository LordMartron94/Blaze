package blaze

import (
	"blaze/core"
	"blaze/reduce"
	"blaze/simd"

	blazetesting "blaze/testing"
	"fmt"
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

	// Setup Manual Memory Allocator
	allocator := memforge.DynamicLinearAllocatorCreateFunction(
		uint64(10*memcore.MegaByte),
		blazetesting.DoubleGrowth,
	)
	defer memforge.DynamicLinearAllocatorDestroy(allocator)

	simd.BlazeSIMDDispatchInit()

	// Test various dimensions to trigger unrolled loops, tails, and scalar paths
	dimensions := []uint64{3, 128, 132, 384, 768, 1024, 1033}

	for _, dimension := range dimensions {
		t.Run(fmt.Sprintf("Dimension=%d", dimension), func(t *testing.T) {
			memforge.DynamicLinearAllocatorReset(allocator)

			// Create Vector
			vector, _ := memarch.MemArchVectorCreate[float64](
				func(sizeBytes, alignment uint64) memcore.MarkRaw {
					return memforge.DynamicLinearAllocatorMallocUnsafe(
						allocator, sizeBytes, alignment,
					)
				},
				dimension,
			)

			// Populate Vector
			values := blazetesting.GenerateRandomVectorF64(dimension, rng)
			memstruct.VectorSetFromSlice(vector, values)

			// 1. Establish Baseline (Uses Optimized AVX Kernel if available)
			sumBaseline := reduce.BlazeReduceVectorSumF64[float64](vector)

			// 2. Force Go Fallback
			// We remove the kernel for: Op=Sum, Out=F64, In=F64.
			// This forces TryExecute() to return false, triggering the Go path.
			oldKernel := simd.BlazeSIMDDispatchKernelRemove(
				core.Blaze_Operation_Vector_Sum,
				core.DTypeF64, // Output
				core.DTypeF64, // Input
			)

			// 3. Run Pure Go Implementation
			sumGo := reduce.BlazeReduceVectorSumF64[float64](vector)

			// 4. Restore Kernel (Crucial for subsequent tests)
			simd.BlazeSIMDDispatchKernelOverride(
				core.Blaze_Operation_Vector_Sum,
				core.DTypeF64,
				oldKernel,
				core.DTypeF64,
			)

			// 5. Compare Results
			AssertClose(t, sumBaseline, sumGo, 1e-14)
		})
	}
}

func AssertClose(t *testing.T, sumGo, sumAsm float64, epsilon float64) {
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
		fmt.Sprintf("mismatch outside tolerance (%e), go=%.20f, asm=%.20f, diff=%.20e", epsilon, sumGo, sumAsm, diff),
		fmt.Sprintf("go & asm matched within tolerance, go=%.20f, asm=%.20f, diff=%.20e", sumGo, sumAsm, diff),
		t,
	)
}
