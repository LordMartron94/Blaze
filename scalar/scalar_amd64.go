//go:build amd64

package scalar

import (
	"blaze/core"
	"blaze/internal"
)

func init() {
	// --- DIVIDE ---

	// F64->F64
	internal.Register(internal.KernelManifest{
		// ---- Identity ----
		Op:     core.Blaze_Operation_Vector_Scalar_Div,
		Inputs: []core.BlazeDType{core.DTypeF64},
		Output: core.DTypeF64,

		// ---- Implementation ----
		Func: ScalarVectorDivideF64iF64o_AVX2,

		// ---- Constraints ----
		RequiredISA:   internal.ISA_AVX2,
		RequiredFlags: internal.Flag_Aligned32 | internal.Flag_Contiguous,

		// ---- Strategy ----
		Priority: 20,
		MinN:     0,
	})

	// F32->F64
	internal.Register(internal.KernelManifest{
		// ---- Identity ----
		Op:     core.Blaze_Operation_Vector_Scalar_Div,
		Inputs: []core.BlazeDType{core.DTypeF32},
		Output: core.DTypeF64,

		// ---- Implementation ----
		Func: ScalarVectorDivideF32iF64o_AVX2,

		// ---- Constraints ----
		RequiredISA:   internal.ISA_AVX2,
		RequiredFlags: internal.Flag_Aligned32 | internal.Flag_Contiguous,

		// ---- Strategy ----
		Priority: 20,
		MinN:     0,
	})

	// --- Multiply ---

	// F64->F64
	internal.Register(internal.KernelManifest{
		// ---- Identity ----
		Op:     core.Blaze_Operation_Vector_Scalar_Mul,
		Inputs: []core.BlazeDType{core.DTypeF64},
		Output: core.DTypeF64,

		// ---- Implementation ----
		Func: ScalarVectorMultiplyF64iF64o_AVX2,

		// ---- Constraints ----
		RequiredISA:   internal.ISA_AVX2,
		RequiredFlags: internal.Flag_Aligned32 | internal.Flag_Contiguous,

		// ---- Strategy ----
		Priority: 20,
		MinN:     0,
	})

	// F32->F64
	internal.Register(internal.KernelManifest{
		// ---- Identity ----
		Op:     core.Blaze_Operation_Vector_Scalar_Mul,
		Inputs: []core.BlazeDType{core.DTypeF32},
		Output: core.DTypeF64,

		// ---- Implementation ----
		Func: ScalarVectorMultiplyF32iF64o_AVX2,

		// ---- Constraints ----
		RequiredISA:   internal.ISA_AVX2,
		RequiredFlags: internal.Flag_Aligned32 | internal.Flag_Contiguous,

		// ---- Strategy ----
		Priority: 20,
		MinN:     0,
	})
}

//go:noescape
func ScalarVectorDivideF64iF64o_AVX2(frame *internal.BlazeKernelFrame)

//go:noescape
func ScalarVectorDivideF32iF64o_AVX2(frame *internal.BlazeKernelFrame)

//go:noescape
func ScalarVectorMultiplyF64iF64o_AVX2(frame *internal.BlazeKernelFrame)

//go:noescape
func ScalarVectorMultiplyF32iF64o_AVX2(frame *internal.BlazeKernelFrame)
