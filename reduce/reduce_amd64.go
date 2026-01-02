//go:build amd64

package reduce

import (
	"blaze/core"
	"blaze/internal"
)

func init() {
	internal.Register(internal.KernelManifest{
		// ---- Identity ----
		Op:     core.Blaze_Operation_Vector_Sum,
		Inputs: []core.BlazeDType{core.DTypeF64},
		Output: core.DTypeF64,

		// ---- Implementation ----
		Func: VectorSumF64iF64o__AVX2,

		// ---- Constraints ----
		RequiredISA: internal.ISA_AVX2,

		// Explicitly require the input buffer to be 32-byte aligned.
		RequiredFlags: internal.Flag_Aligned32,

		// ---- Strategy ----
		Priority: 20,
		MinN:     0,
	})
}

//go:noescape
func VectorSumF64iF64o__AVX2(frame *internal.BlazeKernelFrame)
