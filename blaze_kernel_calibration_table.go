package blaze

import (
	"blaze/core"
	"blaze/reduce"
	blazetesting "blaze/testing"
	"math/rand"
	"memarch"
	"memcore"
	"memstruct"
)

// -----------------------------------------------------------------------------
// Target Definitions (The Table)
// -----------------------------------------------------------------------------

// GetStandardTargets returns the official list of kernels that require calibration.
// This allows the test runner to simply ask for "everything" without knowing details.
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
				vec, _ := memarch.MemArchVectorCreate[float64](allocFn, uint64(size))
				memstruct.VectorSetFromSlice(vec, blazetesting.GenerateRandomVectorF64(uint64(size), rng))
				return []memcore.MarkRaw{vec}
			},
			Execute: func(inputs []memcore.MarkRaw) {
				reduce.BlazeReduceVectorSumF64[float64](inputs[0], &sinkF64)
			},
		},
		{
			Name:        "Sum_F32__F64",
			Operation:   core.Blaze_Operation_Vector_Sum,
			OutputDType: core.DTypeF64,
			InputDTypes: []core.BlazeDType{core.DTypeF32},
			CreateInputs: func(size int, allocFn func(uint64, uint64) memcore.MarkRaw, rng *rand.Rand) []memcore.MarkRaw {
				vec, _ := memarch.MemArchVectorCreate[float32](allocFn, uint64(size))
				memstruct.VectorSetFromSlice(vec, blazetesting.GenerateRandomVectorF32(uint64(size), rng))
				return []memcore.MarkRaw{vec}
			},
			Execute: func(inputs []memcore.MarkRaw) {
				reduce.BlazeReduceVectorSumF64[float32](inputs[0], &sinkF64)
			},
		},
		{
			Name:        "Sum_F32__F32",
			Operation:   core.Blaze_Operation_Vector_Sum,
			OutputDType: core.DTypeF32,
			InputDTypes: []core.BlazeDType{core.DTypeF32},
			CreateInputs: func(size int, allocFn func(uint64, uint64) memcore.MarkRaw, rng *rand.Rand) []memcore.MarkRaw {
				vec, _ := memarch.MemArchVectorCreate[float32](allocFn, uint64(size))
				memstruct.VectorSetFromSlice(vec, blazetesting.GenerateRandomVectorF32(uint64(size), rng))
				return []memcore.MarkRaw{vec}
			},
			Execute: func(inputs []memcore.MarkRaw) {
				reduce.BlazeReduceVectorSumF32[float32](inputs[0], &sinkF32)
			},
		},
		{
			Name:        "Dot_F64_F64__F64",
			Operation:   core.Blaze_Operation_Vector_Dot,
			OutputDType: core.DTypeF64,
			InputDTypes: []core.BlazeDType{core.DTypeF64, core.DTypeF64},
			CreateInputs: func(size int, allocFn func(uint64, uint64) memcore.MarkRaw, rng *rand.Rand) []memcore.MarkRaw {
				vecA, _ := memarch.MemArchVectorCreate[float64](allocFn, uint64(size))
				vecB, _ := memarch.MemArchVectorCreate[float64](allocFn, uint64(size))

				memstruct.VectorSetFromSlice(vecA, blazetesting.GenerateRandomVectorF64(uint64(size), rng))
				memstruct.VectorSetFromSlice(vecB, blazetesting.GenerateRandomVectorF64(uint64(size), rng))

				return []memcore.MarkRaw{vecA, vecB}
			},
			Execute: func(inputs []memcore.MarkRaw) {
				reduce.BlazeReduceVectorDotProductF64[float64, float64](inputs[0], inputs[1], &sinkF64)
			},
		},
		{
			Name:        "Dot_F32_F32__F64",
			Operation:   core.Blaze_Operation_Vector_Dot,
			OutputDType: core.DTypeF64,
			InputDTypes: []core.BlazeDType{core.DTypeF32, core.DTypeF32},
			CreateInputs: func(size int, allocFn func(uint64, uint64) memcore.MarkRaw, rng *rand.Rand) []memcore.MarkRaw {
				vecA, _ := memarch.MemArchVectorCreate[float32](allocFn, uint64(size))
				vecB, _ := memarch.MemArchVectorCreate[float32](allocFn, uint64(size))

				memstruct.VectorSetFromSlice(vecA, blazetesting.GenerateRandomVectorF32(uint64(size), rng))
				memstruct.VectorSetFromSlice(vecB, blazetesting.GenerateRandomVectorF32(uint64(size), rng))

				return []memcore.MarkRaw{vecA, vecB}
			},
			Execute: func(inputs []memcore.MarkRaw) {
				reduce.BlazeReduceVectorDotProductF64[float32, float32](inputs[0], inputs[1], &sinkF64)
			},
		},
		{
			Name:        "Dot_F32_F64__F64",
			Operation:   core.Blaze_Operation_Vector_Dot,
			OutputDType: core.DTypeF64,
			InputDTypes: []core.BlazeDType{core.DTypeF32, core.DTypeF64},
			CreateInputs: func(size int, allocFn func(uint64, uint64) memcore.MarkRaw, rng *rand.Rand) []memcore.MarkRaw {
				vecA, _ := memarch.MemArchVectorCreate[float32](allocFn, uint64(size))
				vecB, _ := memarch.MemArchVectorCreate[float64](allocFn, uint64(size))

				memstruct.VectorSetFromSlice(vecA, blazetesting.GenerateRandomVectorF32(uint64(size), rng))
				memstruct.VectorSetFromSlice(vecB, blazetesting.GenerateRandomVectorF64(uint64(size), rng))

				return []memcore.MarkRaw{vecA, vecB}
			},
			Execute: func(inputs []memcore.MarkRaw) {
				reduce.BlazeReduceVectorDotProductF64[float32, float64](inputs[0], inputs[1], &sinkF64)
			},
		},
	}
}
