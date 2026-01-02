//go:build arm64

package simd

import (
	"blaze/internal"
	"golang.org/x/sys/cpu"
)

func getCPUFlags() internal.CPUFlags {
	var flags internal.CPUFlags

	// ASIMD is the Go name for NEON (Standard 128-bit SIMD)
	if cpu.ARM64.HasASIMD {
		flags |= internal.ISA_NEON
	}

	// SVE (Scalable Vector Extension - Variable length vectors)
	// Used in high-performance ARM chips (Fugaku, Graviton 3+, etc.)
	if cpu.ARM64.HasSVE {
		flags |= internal.ISA_SVE
	}

	return flags
}
