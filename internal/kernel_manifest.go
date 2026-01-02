package internal

/* BlazeExecutionFn performs a Blaze operation. */
type BlazeExecutionFn func(frame *BlazeKernelFrame)

/* CPUFlags determine the specific architectural capacity needed for this kernel to be used. */
type CPUFlags uint64

/* KernelManifest defines a single compile-time contract for what a kernel provides and requires. */
type KernelManifest struct {
	Op     BlazeOperationID
	Inputs []BlazeDataType
	Output BlazeDataType

	Func          BlazeExecutionFn
	RequiredISA   CPUFlags
	RequiredFlags Flags
	Priority      int
	MinN          uint64
}

const (
	// ISA_Generic represents the baseline Go implementation (portable).
	// Always available, usually lowest priority (0).
	ISA_Generic CPUFlags = 0

	// ---- x86-64 Extensions (Intel/AMD) ----

	// ISA_SSE42: Streaming SIMD Extensions 4.2.
	// Baseline for most modern servers. 128-bit vectors.
	ISA_SSE42 CPUFlags = 1 << 0

	// ISA_AVX: Advanced Vector Extensions.
	// Moves to 256-bit floating point, but meant for float heavy loads.
	ISA_AVX CPUFlags = 1 << 1

	// ISA_AVX2: Has integer vector support + FMA3 (Fused Multiply-Add).
	// The sweet spot for most current deployments. 256-bit vectors.
	ISA_AVX2 CPUFlags = 1 << 2

	// ISA_FMA3: Explicit Fused Multiply-Add support (usually implied by AVX2).
	// Crucial for dot products and matrix mul accuracy.
	ISA_FMA3 CPUFlags = 1 << 3

	// ISA_AVX512F: AVX-512 Foundation.
	// 512-bit vectors (ZMM registers). Massive throughput potential.
	ISA_AVX512F CPUFlags = 1 << 4

	// ISA_AVX512BW: AVX-512 Byte and Word instructions.
	// Critical for 8-bit/16-bit integer quantization or image processing.
	ISA_AVX512BW CPUFlags = 1 << 5

	// ISA_AVX512DQ: AVX-512 Doubleword and Quadword.
	// Useful for 64-bit integer / float64 heavy workloads.
	ISA_AVX512DQ CPUFlags = 1 << 6

	// ISA_AVX512VL: Vector Length Extensions.
	// Allows using AVX-512 features on 128/256-bit registers (avoids frequency throttling).
	ISA_AVX512VL CPUFlags = 1 << 7

	// ---- ARM64 Extensions (Apple Silicon / Graviton) ----

	// ISA_NEON: Advanced SIMD.
	// Standard on all ARMv8. 128-bit vectors.
	ISA_NEON CPUFlags = 1 << 20

	// ISA_SVE: Scalable Vector Extension.
	// Variable vector length (hardware decided). Common in HPC ARM chips.
	ISA_SVE CPUFlags = 1 << 21
)
