//go:build amd64

package reduce

import (
	"blaze/core"
	"blaze/internal"
)

func init() {

	// F64->F64
	internal.Register(internal.KernelManifest{
		// ---- Identity ----
		Op:     core.Blaze_Operation_Vector_Sum,
		Inputs: []core.BlazeDType{core.DTypeF64},
		Output: core.DTypeF64,

		// ---- Implementation ----
		Func: VectorSumF64iF64o__AVX2,

		// ---- Constraints ----
		RequiredISA:   internal.ISA_AVX2,
		RequiredFlags: internal.Flag_Aligned32 | internal.Flag_Contiguous,

		// ---- Strategy ----
		Priority: 20,
		MinN:     2300, // ASM only improves on Go performance when vectors become really big.
	})

	// F32->F64
	internal.Register(internal.KernelManifest{
		// ---- Identity ----
		Op:     core.Blaze_Operation_Vector_Sum,
		Inputs: []core.BlazeDType{core.DTypeF32},
		Output: core.DTypeF64,

		// ---- Implementation ----
		Func: VectorSumF32iF64o__AVX2,

		// ---- Constraints ----
		RequiredISA:   internal.ISA_AVX2,
		RequiredFlags: internal.Flag_Aligned32 | internal.Flag_Contiguous,

		// ---- Strategy ----
		Priority: 20,
		MinN:     1400, // ASM improves on Go performance a lot sooner due to conversions happening inside the Go fallback.
	})

	// SoL
	internal.Register(internal.KernelManifest{
		// ---- Identity ----
		Op:     core.Blaze_Operation_SpeedOfLight,
		Inputs: []core.BlazeDType{core.DTypeNone},
		Output: core.DTypeNone,

		// ---- Implementation ----
		Func: SpeedOfLightTest,

		// ---- Constraints ----
		RequiredISA:   internal.ISA_AVX2,
		RequiredFlags: internal.Flag_NoRequirements,

		// ---- Strategy ----
		Priority: 20,
		MinN:     0,
	})

	internal.Register(internal.KernelManifest{
		// ---- Identity ----
		Op:     core.Blaze_Operation_SpeedOfLightThroughput,
		Inputs: []core.BlazeDType{core.DTypeF64},
		Output: core.DTypeNone,

		// ---- Implementation ----
		Func: SpeedOfLightTest_Throughput,

		// ---- Constraints ----
		RequiredISA:   internal.ISA_AVX2,
		RequiredFlags: internal.Flag_Aligned32 | internal.Flag_Contiguous,

		// ---- Strategy ----
		Priority: 20,
		MinN:     0,
	})
}

//go:noescape
func VectorSumF64iF64o__AVX2(frame *internal.BlazeKernelFrame)

//go:noescape
func VectorSumF32iF64o__AVX2(frame *internal.BlazeKernelFrame)

//go:noescape
func SpeedOfLightTest(frame *internal.BlazeKernelFrame)

//go:noescape
func SpeedOfLightTest_Throughput(frame *internal.BlazeKernelFrame)
