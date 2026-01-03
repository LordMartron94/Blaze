package blaze

import (
	"blaze/core"
	"blaze/internal"
	"blaze/reduce"
	"blaze/simd"
	blazetesting "blaze/testing"
	"fmt"
	"foundation"
	"foundation/benchmarking"
	"math/rand"
	"memarch"
	"memcore"
	"memforge"
	"memstruct"
	"runtime"
	"runtime/debug"
	"testing"
)

func BenchmarkVectorSum(b *testing.B) {
	// 1. Initialize the library once for the entire benchmark suite
	simd.BlazeSIMDDispatchInit()

	// 2. Override kernel requirements to ensure ASM kernels always execute
	// This prevents benchmarks from unintentionally using Go fallback due to MinN constraints
	simd.BlazeSIMDDispatchOverrideRequirements(
		core.Blaze_Operation_Vector_Sum,
		core.DTypeF64, // Output
		0,             // MinN = 0 (no minimum dimension requirement)
		0,             // RequiredFlags = 0 (no flag requirements)
		core.DTypeF64, // Input: F64 -> F64 path
	)
	simd.BlazeSIMDDispatchOverrideRequirements(
		core.Blaze_Operation_Vector_Sum,
		core.DTypeF64, // Output
		0,             // MinN = 0 (no minimum dimension requirement)
		0,             // RequiredFlags = 0 (no flag requirements)
		core.DTypeF32, // Input: F32 -> F64 path
	)

	// 3. Clear overrides when done to prevent affecting other tests
	defer simd.BlazeSIMDDispatchClearAllRequirementOverrides()

	dimensions := []uint64{128, 384, 768, 1024, 100_000, 1_000_000}
	methods := []string{"Go", "Asm"}

	// Benchmark both F32 -> F64 and F64 -> F64 paths
	benchmarkVectorSumForType[float32](b, dimensions, methods, "F32")
	benchmarkVectorSumForType[float64](b, dimensions, methods, "F64")
}

var GlobalSink uint64

func BenchmarkVectorSpeedOfLight(b *testing.B) {
	simd.BlazeSIMDDispatchInit()

	b.Run("SpeedOfLight", func(b *testing.B) {
		type benchData struct {
			frame *internal.BlazeKernelFrame
		}

		benchmarking.BenchmarkWithMetrics(b,
			func(b *testing.B) benchData {
				return benchData{frame: new(internal.BlazeKernelFrame)}
			},
			func(d benchData, b *testing.B) {
				benchmarking.RunBatchedBenchmark(
					b,
					func(i int) {
						res := reduce.BlazeReduceSpeedOfLight(d.frame)
						GlobalSink = res
					},
					blazetesting.DefaultMaxHeapGrowth,
					blazetesting.DefaultMaxHeapSize,
					blazetesting.DefaultMemoryCheckInterval,
				)
			},
			func(d benchData, b *testing.B) {
				runtime.KeepAlive(GlobalSink)
			})
	})

	BenchmarkVectorMemoryThroughput(b)
}

func BenchmarkVectorMemoryThroughput(b *testing.B) {
	simd.BlazeSIMDDispatchInit()

	dimensions := []uint64{1024, 10_000, 100_000, 1_000_000, 10_000_000}

	for _, dimension := range dimensions {
		b.Run(fmt.Sprintf("Dimension=%d", dimension), func(b *testing.B) {
			type benchData struct {
				vector    memcore.MarkRaw
				oldGC     int
				allocator memcore.MarkRaw
			}

			bytesPerOp := float64(dimension * memcore.SizeOf[float64]())

			benchmarking.BenchmarkWithMetricsConfig(b,
				benchmarking.BenchmarkMetricsConfig{
					BytesPerOp: bytesPerOp,
				},
				// --- SETUP Phase ---
				func(b *testing.B) benchData {
					oldGC := debug.SetGCPercent(-1)
					rng := rand.New(rand.NewSource(42))

					// Use a larger initial capacity to avoid reallocations during setup
					allocator := memforge.DynamicLinearAllocatorCreateFunction(
						uint64(dimension*uint64(memcore.SizeOf[float64]())+1024),
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

					// Generate random data
					memstruct.VectorSetFromSlice(
						vector,
						blazetesting.GenerateRandomVectorF64(dimension, rng),
					)

					return benchData{
						vector:    vector,
						oldGC:     oldGC,
						allocator: allocator,
					}
				},
				// --- MEASUREMENT Phase ---
				func(d benchData, b *testing.B) {
					benchmarking.RunBatchedBenchmark(
						b,
						func(i int) {
							reduce.BlazeReduceMemoryThroughput[float64](d.vector)
						},
						blazetesting.DefaultMaxHeapGrowth,
						blazetesting.DefaultMaxHeapSize,
						blazetesting.DefaultMemoryCheckInterval,
					)
				},
				// --- TEARDOWN Phase ---
				func(d benchData, b *testing.B) {
					memforge.DynamicLinearAllocatorDestroy(d.allocator)
					debug.SetGCPercent(d.oldGC)
				},
			)
		})
	}
}

func benchmarkVectorSumForType[T foundation.Numeric](
	b *testing.B,
	dimensions []uint64,
	methods []string,
	inputTypeName string,
) {
	for _, method := range methods {
		for _, dimension := range dimensions {
			b.Run(fmt.Sprintf("InputType=%s/Dimension=%d/Method=%s", inputTypeName, dimension, method), func(b *testing.B) {

				type benchData struct {
					vector    memcore.MarkRaw
					oldGC     int
					allocator memcore.MarkRaw
					restore   func()
				}

				flopsPerOp := float64(dimension - 1)
				bytesPerOp := float64(dimension * memcore.SizeOf[T]())

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
							uint64(dimension*uint64(memcore.SizeOf[T]())+1024),
							blazetesting.DoubleGrowth,
						)

						vector, _ := memarch.MemArchVectorCreate[T](
							func(sizeBytes, alignment uint64) memcore.MarkRaw {
								return memforge.DynamicLinearAllocatorMallocUnsafe(
									allocator,
									sizeBytes,
									alignment,
								)
							},
							dimension,
						)

						// Generate data based on input type
						var zero T
						switch any(zero).(type) {
						case float32:
							memstruct.VectorSetFromSlice(
								vector,
								blazetesting.GenerateRandomVectorF32(dimension, rng),
							)
						case float64:
							memstruct.VectorSetFromSlice(
								vector,
								blazetesting.GenerateRandomVectorF64(dimension, rng),
							)
						default:
							b.Fatalf("unsupported type for benchmark")
						}

						restore := func() {}

						// --- Explicitly Force Go fallback if requested ---
						if method == "Go" {
							// Remove the kernel for the specific signature: Sum(T) -> F64
							inputDType := core.BlazeDTypeGet[T]()
							old := simd.BlazeSIMDDispatchKernelRemove(
								core.Blaze_Operation_Vector_Sum,
								core.DTypeF64, // Output (always F64)
								inputDType,    // Input (F32 or F64)
							)

							restore = func() {
								// Restore the kernel so the "Asm" method run isn't broken
								simd.BlazeSIMDDispatchKernelOverride(
									core.Blaze_Operation_Vector_Sum,
									core.DTypeF64,
									old,
									inputDType,
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
								sum := reduce.BlazeReduceVectorSumF64[T](d.vector)
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
