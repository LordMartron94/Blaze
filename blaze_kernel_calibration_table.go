package blaze

import (
	"blaze/core"
	"blaze/reduce"
	"blaze/scalar"
	blazetesting "blaze/testing"
	"math/rand"
	"memarch"
	"memcore"
	"memstruct"
)

// CalibrationBatchSize determines how many distinct vector sets are processed
// per execution unit.
//
// We use 32 to ensure the working set size exceeds L1 Cache (~32KB-48KB).
// For Dim=384 (F64), 1 pair is 6KB. 32 pairs is 192KB.
// This forces the benchmark to measure L2/Memory bandwidth performance,
// preventing the scalar Go code from artificially benefiting from L1 residency
// and branch prediction memorization.
const CalibrationBatchSize = 32

// GetStandardTargets returns the official list of kernels that require calibration.
// Updated to use Batched Execution to reflect real-world memory access patterns.
func GetStandardTargets() []CalibrationTarget {
	var sinkF64 float64
	var sinkF32 float32

	return []CalibrationTarget{
		{
			Name:        "Sum_F64__F64",
			Operation:   core.Blaze_Operation_Vector_Sum,
			OutputDType: core.DTypeF64,
			InputDTypes: []core.BlazeDType{core.DTypeF64},
			CreateInputs: func(size int, allocFn func(uint64, uint64) memcore.MarkRaw, rng *rand.Rand) []memcore.MarkRaw {
				// Allocate Batch
				inputs := make([]memcore.MarkRaw, 0, CalibrationBatchSize)
				for i := 0; i < CalibrationBatchSize; i++ {
					vec, _ := memarch.MemArchVectorCreate[float64](allocFn, uint64(size))
					memstruct.VectorSetFromSlice(vec, blazetesting.GenerateRandomVectorF64(uint64(size), rng))
					inputs = append(inputs, vec)
				}
				return inputs
			},
			Execute: func(inputs []memcore.MarkRaw) {
				for i := 0; i < len(inputs); i++ {
					reduce.BlazeReduceVectorSumF64[float64](inputs[i], &sinkF64)
				}
			},
		},
		{
			Name:        "Sum_F32__F64",
			Operation:   core.Blaze_Operation_Vector_Sum,
			OutputDType: core.DTypeF64,
			InputDTypes: []core.BlazeDType{core.DTypeF32},
			CreateInputs: func(size int, allocFn func(uint64, uint64) memcore.MarkRaw, rng *rand.Rand) []memcore.MarkRaw {
				inputs := make([]memcore.MarkRaw, 0, CalibrationBatchSize)
				for i := 0; i < CalibrationBatchSize; i++ {
					vec, _ := memarch.MemArchVectorCreate[float32](allocFn, uint64(size))
					memstruct.VectorSetFromSlice(vec, blazetesting.GenerateRandomVectorF32(uint64(size), rng))
					inputs = append(inputs, vec)
				}
				return inputs
			},
			Execute: func(inputs []memcore.MarkRaw) {
				for i := 0; i < len(inputs); i++ {
					reduce.BlazeReduceVectorSumF64[float32](inputs[i], &sinkF64)
				}
			},
		},
		{
			Name:        "Sum_F32__F32",
			Operation:   core.Blaze_Operation_Vector_Sum,
			OutputDType: core.DTypeF32,
			InputDTypes: []core.BlazeDType{core.DTypeF32},
			CreateInputs: func(size int, allocFn func(uint64, uint64) memcore.MarkRaw, rng *rand.Rand) []memcore.MarkRaw {
				inputs := make([]memcore.MarkRaw, 0, CalibrationBatchSize)
				for i := 0; i < CalibrationBatchSize; i++ {
					vec, _ := memarch.MemArchVectorCreate[float32](allocFn, uint64(size))
					memstruct.VectorSetFromSlice(vec, blazetesting.GenerateRandomVectorF32(uint64(size), rng))
					inputs = append(inputs, vec)
				}
				return inputs
			},
			Execute: func(inputs []memcore.MarkRaw) {
				for i := 0; i < len(inputs); i++ {
					reduce.BlazeReduceVectorSumF32[float32](inputs[i], &sinkF32)
				}
			},
		},
		{
			Name:        "Dot_F64_F64__F64",
			Operation:   core.Blaze_Operation_Vector_Dot,
			OutputDType: core.DTypeF64,
			InputDTypes: []core.BlazeDType{core.DTypeF64, core.DTypeF64},
			CreateInputs: func(size int, allocFn func(uint64, uint64) memcore.MarkRaw, rng *rand.Rand) []memcore.MarkRaw {
				inputs := make([]memcore.MarkRaw, 0, CalibrationBatchSize*2)
				for i := 0; i < CalibrationBatchSize; i++ {
					vecA, _ := memarch.MemArchVectorCreate[float64](allocFn, uint64(size))
					vecB, _ := memarch.MemArchVectorCreate[float64](allocFn, uint64(size))
					memstruct.VectorSetFromSlice(vecA, blazetesting.GenerateRandomVectorF64(uint64(size), rng))
					memstruct.VectorSetFromSlice(vecB, blazetesting.GenerateRandomVectorF64(uint64(size), rng))
					inputs = append(inputs, vecA, vecB)
				}
				return inputs
			},
			Execute: func(inputs []memcore.MarkRaw) {
				// Iterate with Stride 2 (VecA, VecB)
				for i := 0; i < len(inputs); i += 2 {
					reduce.BlazeReduceVectorDotProductF64[float64, float64](inputs[i], inputs[i+1], &sinkF64)
				}
			},
		},
		{
			Name:        "Dot_F32_F32__F64",
			Operation:   core.Blaze_Operation_Vector_Dot,
			OutputDType: core.DTypeF64,
			InputDTypes: []core.BlazeDType{core.DTypeF32, core.DTypeF32},
			CreateInputs: func(size int, allocFn func(uint64, uint64) memcore.MarkRaw, rng *rand.Rand) []memcore.MarkRaw {
				inputs := make([]memcore.MarkRaw, 0, CalibrationBatchSize*2)
				for i := 0; i < CalibrationBatchSize; i++ {
					vecA, _ := memarch.MemArchVectorCreate[float32](allocFn, uint64(size))
					vecB, _ := memarch.MemArchVectorCreate[float32](allocFn, uint64(size))
					memstruct.VectorSetFromSlice(vecA, blazetesting.GenerateRandomVectorF32(uint64(size), rng))
					memstruct.VectorSetFromSlice(vecB, blazetesting.GenerateRandomVectorF32(uint64(size), rng))
					inputs = append(inputs, vecA, vecB)
				}
				return inputs
			},
			Execute: func(inputs []memcore.MarkRaw) {
				for i := 0; i < len(inputs); i += 2 {
					reduce.BlazeReduceVectorDotProductF64[float32, float32](inputs[i], inputs[i+1], &sinkF64)
				}
			},
		},
		{
			Name:        "Dot_F32_F64__F64",
			Operation:   core.Blaze_Operation_Vector_Dot,
			OutputDType: core.DTypeF64,
			InputDTypes: []core.BlazeDType{core.DTypeF32, core.DTypeF64},
			CreateInputs: func(size int, allocFn func(uint64, uint64) memcore.MarkRaw, rng *rand.Rand) []memcore.MarkRaw {
				inputs := make([]memcore.MarkRaw, 0, CalibrationBatchSize*2)
				for i := 0; i < CalibrationBatchSize; i++ {
					vecA, _ := memarch.MemArchVectorCreate[float32](allocFn, uint64(size))
					vecB, _ := memarch.MemArchVectorCreate[float64](allocFn, uint64(size))
					memstruct.VectorSetFromSlice(vecA, blazetesting.GenerateRandomVectorF32(uint64(size), rng))
					memstruct.VectorSetFromSlice(vecB, blazetesting.GenerateRandomVectorF64(uint64(size), rng))
					inputs = append(inputs, vecA, vecB)
				}
				return inputs
			},
			Execute: func(inputs []memcore.MarkRaw) {
				for i := 0; i < len(inputs); i += 2 {
					reduce.BlazeReduceVectorDotProductF64[float32, float64](inputs[i], inputs[i+1], &sinkF64)
				}
			},
		},
		{
			Name:        "Div_Scalar_F64__F64",
			Operation:   core.Blaze_Operation_Vector_Scalar_Div,
			OutputDType: core.DTypeF64,
			InputDTypes: []core.BlazeDType{core.DTypeF64},
			CreateInputs: func(size int, allocFn func(uint64, uint64) memcore.MarkRaw, rng *rand.Rand) []memcore.MarkRaw {
				inputs := make([]memcore.MarkRaw, 0, CalibrationBatchSize*2)
				for i := 0; i < CalibrationBatchSize; i++ {
					src, _ := memarch.MemArchVectorCreate[float64](allocFn, uint64(size))
					dst, _ := memarch.MemArchVectorCreate[float64](allocFn, uint64(size))
					memstruct.VectorSetFromSlice(src, blazetesting.GenerateRandomVectorF64(uint64(size), rng))
					inputs = append(inputs, src, dst)
				}
				return inputs
			},
			Execute: func(inputs []memcore.MarkRaw) {
				for i := 0; i < len(inputs); i += 2 {
					scalar.BlazeScalarVectorDivideF64[float64](inputs[i], inputs[i+1], 1.234)
				}
			},
		},
		{
			Name:        "Div_Scalar_F32__F64",
			Operation:   core.Blaze_Operation_Vector_Scalar_Div,
			OutputDType: core.DTypeF64,
			InputDTypes: []core.BlazeDType{core.DTypeF32},
			CreateInputs: func(size int, allocFn func(uint64, uint64) memcore.MarkRaw, rng *rand.Rand) []memcore.MarkRaw {
				inputs := make([]memcore.MarkRaw, 0, CalibrationBatchSize*2)
				for i := 0; i < CalibrationBatchSize; i++ {
					src, _ := memarch.MemArchVectorCreate[float32](allocFn, uint64(size))
					dst, _ := memarch.MemArchVectorCreate[float64](allocFn, uint64(size))
					memstruct.VectorSetFromSlice(src, blazetesting.GenerateRandomVectorF32(uint64(size), rng))
					inputs = append(inputs, src, dst)
				}
				return inputs
			},
			Execute: func(inputs []memcore.MarkRaw) {
				for i := 0; i < len(inputs); i += 2 {
					scalar.BlazeScalarVectorDivideF64[float32](inputs[i], inputs[i+1], 1.234)
				}
			},
		},
		{
			Name:        "Mul_Scalar_F64__F64",
			Operation:   core.Blaze_Operation_Vector_Scalar_Mul,
			OutputDType: core.DTypeF64,
			InputDTypes: []core.BlazeDType{core.DTypeF64},
			CreateInputs: func(size int, allocFn func(uint64, uint64) memcore.MarkRaw, rng *rand.Rand) []memcore.MarkRaw {
				inputs := make([]memcore.MarkRaw, 0, CalibrationBatchSize*2)
				for i := 0; i < CalibrationBatchSize; i++ {
					src, _ := memarch.MemArchVectorCreate[float64](allocFn, uint64(size))
					dst, _ := memarch.MemArchVectorCreate[float64](allocFn, uint64(size))
					memstruct.VectorSetFromSlice(src, blazetesting.GenerateRandomVectorF64(uint64(size), rng))
					inputs = append(inputs, src, dst)
				}
				return inputs
			},
			Execute: func(inputs []memcore.MarkRaw) {
				for i := 0; i < len(inputs); i += 2 {
					scalar.BlazeScalarVectorMultiplyF64[float64](inputs[i], inputs[i+1], 1.234)
				}
			},
		},
		{
			Name:        "Mul_Scalar_F32__F64",
			Operation:   core.Blaze_Operation_Vector_Scalar_Mul,
			OutputDType: core.DTypeF64,
			InputDTypes: []core.BlazeDType{core.DTypeF32},
			CreateInputs: func(size int, allocFn func(uint64, uint64) memcore.MarkRaw, rng *rand.Rand) []memcore.MarkRaw {
				inputs := make([]memcore.MarkRaw, 0, CalibrationBatchSize*2)
				for i := 0; i < CalibrationBatchSize; i++ {
					src, _ := memarch.MemArchVectorCreate[float32](allocFn, uint64(size))
					dst, _ := memarch.MemArchVectorCreate[float64](allocFn, uint64(size))
					memstruct.VectorSetFromSlice(src, blazetesting.GenerateRandomVectorF32(uint64(size), rng))
					inputs = append(inputs, src, dst)
				}
				return inputs
			},
			Execute: func(inputs []memcore.MarkRaw) {
				for i := 0; i < len(inputs); i += 2 {
					scalar.BlazeScalarVectorMultiplyF64[float32](inputs[i], inputs[i+1], 1.234)
				}
			},
		},
	}
}
