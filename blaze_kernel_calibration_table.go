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
	var sumF64 float64
	var sumF32 float32

	return []CalibrationTarget{
		{
			Name:        "Sum_F64_F64",
			Operation:   core.Blaze_Operation_Vector_Sum,
			OutputDType: core.DTypeF64,
			InputDTypes: []core.BlazeDType{core.DTypeF64},
			CreateInputs: func(size int, allocFn func(uint64, uint64) memcore.MarkRaw, rng *rand.Rand) []memcore.MarkRaw {
				vec, _ := memarch.MemArchVectorCreate[float64](allocFn, uint64(size))
				memstruct.VectorSetFromSlice(vec, blazetesting.GenerateRandomVectorF64(uint64(size), rng))
				return []memcore.MarkRaw{vec}
			},
			Execute: func(inputs []memcore.MarkRaw) {
				reduce.BlazeReduceVectorSumF64[float64](inputs[0], &sumF64)
			},
		},
		{
			Name:        "Sum_F32_F64",
			Operation:   core.Blaze_Operation_Vector_Sum,
			OutputDType: core.DTypeF64,
			InputDTypes: []core.BlazeDType{core.DTypeF32},
			CreateInputs: func(size int, allocFn func(uint64, uint64) memcore.MarkRaw, rng *rand.Rand) []memcore.MarkRaw {
				vec, _ := memarch.MemArchVectorCreate[float32](allocFn, uint64(size))
				memstruct.VectorSetFromSlice(vec, blazetesting.GenerateRandomVectorF32(uint64(size), rng))
				return []memcore.MarkRaw{vec}
			},
			Execute: func(inputs []memcore.MarkRaw) {
				reduce.BlazeReduceVectorSumF64[float32](inputs[0], &sumF64)
			},
		},
		{
			Name:        "Sum_F32_F32",
			Operation:   core.Blaze_Operation_Vector_Sum,
			OutputDType: core.DTypeF32,
			InputDTypes: []core.BlazeDType{core.DTypeF32},
			CreateInputs: func(size int, allocFn func(uint64, uint64) memcore.MarkRaw, rng *rand.Rand) []memcore.MarkRaw {
				vec, _ := memarch.MemArchVectorCreate[float32](allocFn, uint64(size))
				memstruct.VectorSetFromSlice(vec, blazetesting.GenerateRandomVectorF32(uint64(size), rng))
				return []memcore.MarkRaw{vec}
			},
			Execute: func(inputs []memcore.MarkRaw) {
				reduce.BlazeReduceVectorSumF32[float32](inputs[0], &sumF32)
			},
		},
	}
}
