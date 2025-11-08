package blaze

import (
	"blaze/metric"
	"blaze/reduce"
	"blaze/scalar"
	"fmt"
	"foundation"
	"foundation/benchmarking"
	"math"
	"math/rand"
	"memarch"
	"memcore"
	"memforge"
	"memstruct"
	"runtime/debug"
	"testing"
)

func doubleGrowth(currentCap, needed uint64) uint64 {
	newSize := currentCap * 2
	if newSize < needed {
		newSize = needed
	}

	if newSize > uint64(1*memcore.GigaByte) {
		panic("way too much memory for a simple test")
	}

	return newSize
}

var growthFnID memcore.FunctionID = memcore.MemcoreFunctionRegisterTyped[memforge.GrowthStrategy](doubleGrowth)

// ────────────────────────────────────────────────────────────────
//   HELPERS
// ────────────────────────────────────────────────────────────────

func fillVectorRandom[T foundation.Numeric](mark memcore.MarkRaw, scale float64) {
	rnd := rand.New(rand.NewSource(42))
	for i := uint64(0); i < memstruct.VectorCapacityGet[T](mark); i++ {
		val := T(rnd.Float64() * scale)
		memstruct.VectorSetAtUnsafe(mark, i, val)
	}
}

// ────────────────────────────────────────────────────────────────
//   MAIN SUITE
// ────────────────────────────────────────────────────────────────

func BenchmarkBlazeVectorSuite(b *testing.B) {
	sizes := []uint64{64, 256, 1024, 4096, 16384}
	scales := []float64{1, 10, 100, 1000}

	for _, n := range sizes {
		for _, scale := range scales {
			group := fmt.Sprintf("N=%d/scale=%.0f", n, scale)

			b.Run(group+"/Float64", func(b *testing.B) {
				runVectorBench[float64](b, n, scale)
			})
			b.Run(group+"/Float32", func(b *testing.B) {
				runVectorBench[float32](b, n, scale)
			})
			b.Run(group+"/Int64", func(b *testing.B) {
				runVectorBench[int64](b, n, scale)
			})
			b.Run(group+"/Uint64", func(b *testing.B) {
				runVectorBench[uint64](b, n, scale)
			})
		}
	}
}

// ────────────────────────────────────────────────────────────────
//   PER-TYPE BENCH LOGIC
// ────────────────────────────────────────────────────────────────

func runVectorBench[T foundation.Numeric](b *testing.B, capacity uint64, scale float64) {
	type benchData struct {
		allocator memcore.MarkRaw
		vector    memcore.MarkRaw
		oldGC     int
	}

	benchmarking.BenchmarkWithMetrics(b,
		// SETUP
		func(b *testing.B) benchData {
			old := debug.SetGCPercent(-1)
			alloc := memforge.DynamicLinearAllocatorCreate(uint64(2*memcore.MegaByte), growthFnID)
			vec, _ := memarch.MemArchVectorCreate[T](
				func(size, align uint64) memcore.MarkRaw {
					return memforge.DynamicLinearAllocatorCalloc(alloc, size, align)
				},
				capacity,
			)
			fillVectorRandom[T](vec, scale)
			return benchData{alloc, vec, old}
		},

		// RUN
		func(d benchData, b *testing.B) {
			for i := 0; i < b.N; i++ {
				switch i % 7 {
				case 0:
					reduce.BlazeReduceVectorSumF64[T](d.vector)
				case 1:
					reduce.BlazeReduceVectorSumSquaredF64[T](d.vector)
				case 2:
					metric.BlazeMetricVectorMagnitudeF64[T](d.vector)
				case 3:
					metric.BlazeMetricVectorMagnitudeF32[T](d.vector)
				case 4:
					idx := uint64(i % int(capacity))
					memstruct.VectorItemGetAtUnsafe[T](d.vector, idx)
				case 5:
					idx := uint64(i % int(capacity))
					val := T(math.Mod(float64(i), scale))
					memstruct.VectorSetAtUnsafe(d.vector, idx, val)
				case 6:
					scalar.BlazeScalarVectorMultiplyF64[T](d.vector, d.vector, float64(1.001))

				}
			}
		},

		// CLEANUP
		func(d benchData, b *testing.B) {
			memforge.DynamicLinearAllocatorDestroy(d.allocator)
			debug.SetGCPercent(d.oldGC)
		},
	)
}

// ────────────────────────────────────────────────────────────────
//   NORMALIZATION BENCHMARKS
// ────────────────────────────────────────────────────────────────

func BenchmarkBlazeVectorSuite_Normalization(b *testing.B) {
	sizes := []uint64{128, 512, 2048}
	for _, n := range sizes {
		b.Run(fmt.Sprintf("Normalize/N=%d", n), func(b *testing.B) {
			type benchData struct {
				alloc memcore.MarkRaw
				src   memcore.MarkRaw
				dst   memcore.MarkRaw
				oldGC int
			}

			benchmarking.BenchmarkWithMetrics(b,
				func(b *testing.B) benchData {
					old := debug.SetGCPercent(-1)
					a := memforge.DynamicLinearAllocatorCreate(uint64(2*memcore.MegaByte), growthFnID)
					src, _ := memarch.MemArchVectorCreate[float64](
						func(size, align uint64) memcore.MarkRaw {
							return memforge.DynamicLinearAllocatorCalloc(a, size, align)
						}, n)
					dst, _ := memarch.MemArchVectorCreate[float64](
						func(size, align uint64) memcore.MarkRaw {
							return memforge.DynamicLinearAllocatorCalloc(a, size, align)
						}, n)
					fillVectorRandom[float64](src, 100)
					return benchData{a, src, dst, old}
				},
				func(d benchData, b *testing.B) {
					for i := 0; i < b.N; i++ {
						metric.BlazeMetricVectorNormalizedF64[float64](d.src, d.dst)
					}
				},
				func(d benchData, b *testing.B) {
					memforge.DynamicLinearAllocatorDestroy(d.alloc)
					debug.SetGCPercent(d.oldGC)
				},
			)
		})
	}
}

// ────────────────────────────────────────────────────────────────
//   EXTRA METRIC BENCHMARKS
// ────────────────────────────────────────────────────────────────

func BenchmarkBlazeVectorSuite_DotAndDistance(b *testing.B) {
	n := uint64(1024)
	b.Run(fmt.Sprintf("DotProduct/N=%d", n), func(b *testing.B) {
		type benchData struct {
			alloc memcore.MarkRaw
			a     memcore.MarkRaw
			bv    memcore.MarkRaw
			oldGC int
		}

		benchmarking.BenchmarkWithMetrics(b,
			func(b *testing.B) benchData {
				old := debug.SetGCPercent(-1)
				a := memforge.DynamicLinearAllocatorCreate(uint64(2*memcore.MegaByte), growthFnID)
				v1, _ := memarch.MemArchVectorCreate[float64](
					func(size, align uint64) memcore.MarkRaw {
						return memforge.DynamicLinearAllocatorCalloc(a, size, align)
					}, n)
				v2, _ := memarch.MemArchVectorCreate[float64](
					func(size, align uint64) memcore.MarkRaw {
						return memforge.DynamicLinearAllocatorCalloc(a, size, align)
					}, n)
				fillVectorRandom[float64](v1, 100)
				fillVectorRandom[float64](v2, 100)
				return benchData{a, v1, v2, old}
			},
			func(d benchData, b *testing.B) {
				for i := 0; i < b.N; i++ {
					reduce.BlazeReduceDotProductF64[float64, float64](d.a, d.bv)
				}
			},
			func(d benchData, b *testing.B) {
				memforge.DynamicLinearAllocatorDestroy(d.alloc)
				debug.SetGCPercent(d.oldGC)
			},
		)
	})
}
