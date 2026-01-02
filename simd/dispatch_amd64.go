//go:build amd64

package simd

import (
	"blaze/internal"
	_ "unsafe"

	"golang.org/x/sys/cpu"
)

func getCPUFlags() internal.CPUFlags {
	var flags internal.CPUFlags

	// SSE 4.2 (Baseline for many optimized text/hash ops)
	if cpu.X86.HasSSE42 {
		flags |= internal.ISA_SSE42
	}

	// AVX (Advanced Vector Extensions - 256-bit floats)
	if cpu.X86.HasAVX {
		flags |= internal.ISA_AVX
	}

	// AVX2 (Integer vectors + FMA3 usually)
	if cpu.X86.HasAVX2 {
		flags |= internal.ISA_AVX2
	}

	// FMA3 (Fused Multiply Add - Critical for DotProduct accuracy)
	if cpu.X86.HasFMA {
		flags |= internal.ISA_FMA3
	}

	// AVX-512 Foundation (512-bit ZMM registers)
	if cpu.X86.HasAVX512F {
		flags |= internal.ISA_AVX512F
	}

	// AVX-512 Byte/Word (Crucial for 8/16-bit integer ops)
	if cpu.X86.HasAVX512BW {
		flags |= internal.ISA_AVX512BW
	}

	// AVX-512 Double/Quad (Crucial for 64-bit ops)
	if cpu.X86.HasAVX512DQ {
		flags |= internal.ISA_AVX512DQ
	}

	// AVX-512 Vector Length (Allows AVX-512 features on 256-bit regs)
	// Prevents clock frequency throttling on some Intel chips.
	if cpu.X86.HasAVX512VL {
		flags |= internal.ISA_AVX512VL
	}

	return flags
}
