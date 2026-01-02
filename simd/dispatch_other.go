//go:build !amd64 && !arm64

package simd

import "blaze/internal"

func getCPUFlags() internal.CPUFlags {
	return internal.ISA_Generic
}
