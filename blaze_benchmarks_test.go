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

var GlobalSink uint64
var GlobalSinkF32 float32
var GlobalSinkF64 float64

func BenchmarkVectorSum(b *testing.B) {
	// 1. Initialize the library once for the entire benchmark suite
	simd.BlazeSIMDDispatchInit()

	override := uint64(0)
	var overrideFlags internal.Flags = 0

	// 2. Override kernel requirements to ensure ASM kernels always execute
	// F64 Output Overrides
	simd.BlazeSIMDDispatchOverrideRequirements(
		core.Blaze_Operation_Vector_Sum,
		core.DTypeF64,
		&override, &overrideFlags,
		core.DTypeF64, // F64 -> F64
	)
	simd.BlazeSIMDDispatchOverrideRequirements(
		core.Blaze_Operation_Vector_Sum,
		core.DTypeF64,
		&override, &overrideFlags,
		core.DTypeF32, // F32 -> F64
	)

	// F32 Output Overrides (NEW)
	simd.BlazeSIMDDispatchOverrideRequirements(
		core.Blaze_Operation_Vector_Sum,
		core.DTypeF32,
		&override, &overrideFlags,
		core.DTypeF32, // F32 -> F32
	)

	// 3. Clear overrides when done
	defer simd.BlazeSIMDDispatchClearAllRequirementOverrides()

	dimensions := []uint64{128, 384, 768, 1024, 100_000, 1_000_000}
	methods := []string{"Go", "Asm"}

	// --- F64 Output Paths ---
	// F32(in) -> F64(out)
	benchmarkVectorSumGeneric[float32](
		b, dimensions, methods, "F32", "F64", core.DTypeF64,
		func(v memcore.MarkRaw, res *float64) { reduce.BlazeReduceVectorSumF64[float32](v, res) },
	)
	// F64(in) -> F64(out)
	benchmarkVectorSumGeneric[float64](
		b, dimensions, methods, "F64", "F64", core.DTypeF64,
		func(v memcore.MarkRaw, res *float64) { reduce.BlazeReduceVectorSumF64[float64](v, res) },
	)

	// --- F32 Output Paths ---
	// F32(in) -> F32(out) (NEW)
	benchmarkVectorSumGeneric[float32](
		b, dimensions, methods, "F32", "F32", core.DTypeF32,
		func(v memcore.MarkRaw, res *float32) { reduce.BlazeReduceVectorSumF32[float32](v, res) },
	)
}

func BenchmarkVectorDotProduct(b *testing.B) {
	simd.BlazeSIMDDispatchInit()

	override := uint64(0)
	var overrideFlags internal.Flags = 0

	// Override requirements for F64 · F64 -> F64
	simd.BlazeSIMDDispatchOverrideRequirements(
		core.Blaze_Operation_Vector_Dot,
		core.DTypeF64,
		&override, &overrideFlags,
		core.DTypeF64, // InA
		core.DTypeF64, // InB
	)
	defer simd.BlazeSIMDDispatchClearAllRequirementOverrides()

	dimensions := []uint64{128, 384, 768, 1024, 100_000, 1_000_000}
	methods := []string{"Go", "Asm"}

	// F64 · F64 -> F64
	benchmarkVectorDotProductGeneric[float64](
		b, dimensions, methods, "F64", "F64", core.DTypeF64,
		func(a, b memcore.MarkRaw, res *float64) {
			reduce.BlazeReduceVectorDotProductF64[float64, float64](a, b, res)
		},
	)
}

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
				benchmarking.BenchmarkMetricsConfig{BytesPerOp: bytesPerOp},
				func(b *testing.B) benchData {
					oldGC := debug.SetGCPercent(-1)
					rng := rand.New(rand.NewSource(42))
					allocator := memforge.DynamicLinearAllocatorCreateFunction(
						uint64(dimension*uint64(memcore.SizeOf[float64]())+1024),
						blazetesting.DoubleGrowth,
					)
					vector, _ := memarch.MemArchVectorCreate[float64](
						func(sizeBytes, alignment uint64) memcore.MarkRaw {
							return memforge.DynamicLinearAllocatorMallocUnsafe(allocator, sizeBytes, alignment)
						},
						dimension,
					)
					memstruct.VectorSetFromSlice(vector, blazetesting.GenerateRandomVectorF64(dimension, rng))
					return benchData{vector: vector, oldGC: oldGC, allocator: allocator}
				},
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
				func(d benchData, b *testing.B) {
					memforge.DynamicLinearAllocatorDestroy(d.allocator)
					debug.SetGCPercent(d.oldGC)
				},
			)
		})
	}
}

// benchmarkVectorSumGeneric handles both Input (T) and Output (U) types.
func benchmarkVectorSumGeneric[T foundation.Numeric, U foundation.Numeric](
	b *testing.B,
	dimensions []uint64,
	methods []string,
	inName string,
	outName string,
	outDType core.BlazeDType,
	reducer func(memcore.MarkRaw, *U),
) {
	for _, method := range methods {
		for _, dimension := range dimensions {
			b.Run(fmt.Sprintf("In=%s/Out=%s/Dim=%d/Method=%s", inName, outName, dimension, method), func(b *testing.B) {

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

						allocator := memforge.DynamicLinearAllocatorCreateFunction(
							uint64(dimension*uint64(memcore.SizeOf[T]())+1024),
							blazetesting.DoubleGrowth,
						)

						vector, _ := memarch.MemArchVectorCreate[T](
							func(sizeBytes, alignment uint64) memcore.MarkRaw {
								return memforge.DynamicLinearAllocatorMallocUnsafe(allocator, sizeBytes, alignment)
							},
							dimension,
						)

						// Initialize data
						var zero T
						switch any(zero).(type) {
						case float32:
							memstruct.VectorSetFromSlice(vector, blazetesting.GenerateRandomVectorF32(dimension, rng))
						case float64:
							memstruct.VectorSetFromSlice(vector, blazetesting.GenerateRandomVectorF64(dimension, rng))
						default:
							b.Fatalf("unsupported type for benchmark")
						}

						restore := func() {}

						// Force Go fallback if requested
						if method == "Go" {
							inputDType := core.BlazeDTypeGet[T]()
							old := simd.BlazeSIMDDispatchKernelRemove(
								core.Blaze_Operation_Vector_Sum,
								outDType, // Remove specific Output kernel
								inputDType,
							)
							restore = func() {
								simd.BlazeSIMDDispatchKernelOverride(
									core.Blaze_Operation_Vector_Sum,
									outDType,
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
								var sum U
								reducer(d.vector, &sum)

								// Efficient Global Sink (No IO, No Optimizing Away)
								switch s := any(sum).(type) {
								case float32:
									GlobalSinkF32 = s
								case float64:
									GlobalSinkF64 = s
								}
							},
							blazetesting.DefaultMaxHeapGrowth,
							blazetesting.DefaultMaxHeapSize,
							blazetesting.DefaultMemoryCheckInterval,
						)
					},
					// --- TEARDOWN Phase ---
					func(d benchData, b *testing.B) {
						d.restore()
						memforge.DynamicLinearAllocatorDestroy(d.allocator)
						debug.SetGCPercent(d.oldGC)
					},
				)
			})
		}
	}
}

// benchmarkVectorDotProductGeneric handles 2 Input vectors (A, B) and Scalar Output.
func benchmarkVectorDotProductGeneric[T foundation.Numeric](
	b *testing.B,
	dimensions []uint64,
	methods []string,
	inName string,
	outName string,
	outDType core.BlazeDType,
	reducer func(memcore.MarkRaw, memcore.MarkRaw, *float64),
) {
	for _, method := range methods {
		for _, dimension := range dimensions {
			b.Run(fmt.Sprintf("In=%s/Out=%s/Dim=%d/Method=%s", inName, outName, dimension, method), func(b *testing.B) {

				type benchData struct {
					vectorA   memcore.MarkRaw
					vectorB   memcore.MarkRaw
					oldGC     int
					allocator memcore.MarkRaw
					restore   func()
				}

				// Dot Product FLOPS: N mults + (N-1) adds ≈ 2*N
				flopsPerOp := float64(2 * dimension)
				// Bytes: Read Vector A + Read Vector B
				bytesPerOp := float64(2 * dimension * memcore.SizeOf[T]())

				benchmarking.BenchmarkWithMetricsConfig(b,
					benchmarking.BenchmarkMetricsConfig{
						FLOPSPerOp: flopsPerOp,
						BytesPerOp: bytesPerOp,
					},
					// --- SETUP Phase ---
					func(b *testing.B) benchData {
						oldGC := debug.SetGCPercent(-1)
						rng := rand.New(rand.NewSource(42))

						// Allocate space for 2 vectors
						allocator := memforge.DynamicLinearAllocatorCreateFunction(
							uint64(2*dimension*uint64(memcore.SizeOf[T]())+2048),
							blazetesting.DoubleGrowth,
						)

						// Vector A
						vecA, _ := memarch.MemArchVectorCreate[T](
							func(sizeBytes, alignment uint64) memcore.MarkRaw {
								return memforge.DynamicLinearAllocatorMallocUnsafe(allocator, sizeBytes, alignment)
							},
							dimension,
						)
						// Vector B
						vecB, _ := memarch.MemArchVectorCreate[T](
							func(sizeBytes, alignment uint64) memcore.MarkRaw {
								return memforge.DynamicLinearAllocatorMallocUnsafe(allocator, sizeBytes, alignment)
							},
							dimension,
						)

						// Initialize data
						var zero T
						switch any(zero).(type) {
						case float32:
							memstruct.VectorSetFromSlice(vecA, blazetesting.GenerateRandomVectorF32(dimension, rng))
							memstruct.VectorSetFromSlice(vecB, blazetesting.GenerateRandomVectorF32(dimension, rng))
						case float64:
							memstruct.VectorSetFromSlice(vecA, blazetesting.GenerateRandomVectorF64(dimension, rng))
							memstruct.VectorSetFromSlice(vecB, blazetesting.GenerateRandomVectorF64(dimension, rng))
						default:
							b.Fatalf("unsupported type for benchmark")
						}

						restore := func() {}

						// Force Go fallback if requested
						if method == "Go" {
							inputDType := core.BlazeDTypeGet[T]()
							// Kernel Key: (Op, Out, InA, InB)
							old := simd.BlazeSIMDDispatchKernelRemove(
								core.Blaze_Operation_Vector_Dot,
								outDType,
								inputDType,
								inputDType,
							)
							restore = func() {
								simd.BlazeSIMDDispatchKernelOverride(
									core.Blaze_Operation_Vector_Dot,
									outDType,
									old,
									inputDType,
									inputDType,
								)
							}
						}

						return benchData{
							vectorA:   vecA,
							vectorB:   vecB,
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
								var sum float64
								reducer(d.vectorA, d.vectorB, &sum)
								GlobalSinkF64 = sum
							},
							blazetesting.DefaultMaxHeapGrowth,
							blazetesting.DefaultMaxHeapSize,
							blazetesting.DefaultMemoryCheckInterval,
						)
					},
					// --- TEARDOWN Phase ---
					func(d benchData, b *testing.B) {
						d.restore()
						memforge.DynamicLinearAllocatorDestroy(d.allocator)
						debug.SetGCPercent(d.oldGC)
					},
				)
			})
		}
	}
}
