package blaze

import (
	"blaze/core"
	"blaze/reduce"
	"blaze/simd"
	blazetesting "blaze/testing"
	"fmt"
	"foundation/benchmarking"
	"math/rand"
	"memarch"
	"memcore"
	"memforge"
	"memstruct"
	"runtime/debug"
	"testing"
)

func BenchmarkVectorSum(b *testing.B) {
	// 1. Initialize the library once for the entire benchmark suite
	simd.BlazeSIMDDispatchInit()

	dimensions := []uint64{128, 384, 768, 1024, 100_000, 1_000_000}
	methods := []string{"Go", "Asm"}

	for _, method := range methods {
		for _, dimension := range dimensions {
			b.Run(fmt.Sprintf("Dimension=%d/Method=%s", dimension, method), func(b *testing.B) {

				type benchData struct {
					vector    memcore.MarkRaw
					oldGC     int
					allocator memcore.MarkRaw
					restore   func()
				}

				flopsPerOp := float64(dimension - 1)
				bytesPerOp := float64(dimension * memcore.SizeOf[float64]())

				benchmarking.BenchmarkWithMetricsConfig(b,
					benchmarking.BenchmarkMetricsConfig{
						FLOPSPerOp: flopsPerOp,
						BytesPerOp: bytesPerOp,
					},
					// --- SETUP Phase ---
					func(b *testing.B) benchData {
						oldGC := debug.SetGCPercent(-1)
						rng := rand.New(rand.NewSource(42))

						// Use a larger initial capacity to avoid reallocations during setup
						allocator := memforge.DynamicLinearAllocatorCreateFunction(
							uint64(dimension*8+1024),
							blazetesting.DoubleGrowth,
						)

						vector, _ := memarch.MemArchVectorCreate[float64](
							func(sizeBytes, alignment uint64) memcore.MarkRaw {
								return memforge.DynamicLinearAllocatorMallocUnsafe(
									allocator,
									sizeBytes,
									alignment,
								)
							},
							dimension,
						)

						memstruct.VectorSetFromSlice(
							vector,
							blazetesting.GenerateRandomVectorF64(dimension, rng),
						)

						restore := func() {}

						// --- Explicitly Force Go fallback if requested ---
						if method == "Go" {
							// Remove the kernel for the specific signature: Sum(F64) -> F64
							old := simd.BlazeSIMDDispatchKernelRemove(
								core.Blaze_Operation_Vector_Sum,
								core.DTypeF64, // Output
								core.DTypeF64, // Input
							)

							restore = func() {
								// Restore the kernel so the "Asm" method run isn't broken
								simd.BlazeSIMDDispatchKernelOverride(
									core.Blaze_Operation_Vector_Sum,
									core.DTypeF64,
									old,
									core.DTypeF64,
								)
							}
						}

						return benchData{
							vector:    vector,
							oldGC:     oldGC,
							allocator: allocator,
							restore:   restore,
						}
					},
					// --- MEASUREMENT Phase ---
					func(d benchData, b *testing.B) {
						benchmarking.RunBatchedBenchmark(
							b,
							func(i int) {
								sum := reduce.BlazeReduceVectorSumF64[float64](d.vector)
								// Sink to prevent compiler from optimizing away the call
								if sum > 1e300 {
									fmt.Print("")
								}
							},
							blazetesting.DefaultMaxHeapGrowth,
							blazetesting.DefaultMaxHeapSize,
							blazetesting.DefaultMemoryCheckInterval,
						)
					},
					// --- TEARDOWN Phase ---
					func(d benchData, b *testing.B) {
						d.restore() // Crucial: Puts the Asm kernel back in the table
						memforge.DynamicLinearAllocatorDestroy(d.allocator)
						debug.SetGCPercent(d.oldGC)
					},
				)
			})
		}
	}
}
