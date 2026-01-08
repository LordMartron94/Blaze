// Package simd provides optimized assembly functions for operations.
package simd

import (
	"blaze/core"
	"blaze/internal"
	"echo"
	"foundation"
	"sort"
	"strings"
	"unsafe"
)

const (
	MaxOPS        = core.BLAZE_OPERATION_COUNT
	MaxSignatures = 4096
)

var dispatchTable [MaxOPS]unsafe.Pointer

type dispatchMap map[uint32]core.BlazeExecutionFn

type kernelRequirements struct {
	MinN          *uint64
	RequiredFlags *internal.Flags
}

var requirementOverrides map[uint32]kernelRequirements

/* BlazeSIMDDispatchKernelGet returns the kernel for the operation and types.*/
func BlazeSIMDDispatchKernelGet(op core.BlazeOperationID, out core.BlazeDType, inputs ...core.BlazeDType) core.BlazeExecutionFn {
	sig := internal.PackSignature(out, inputs...)

	return resolveFast(op, sig)
}

/* BlazeSIMDDispatchInit initializes the function table. */
func BlazeSIMDDispatchInit() {
	cpuFlags := getCPUFlags()
	candidates := groupManifests(internal.GetRegistry())

	for op, sigs := range candidates {
		for sig, manifests := range sigs {
			valid := filterByISA(manifests, cpuFlags)
			if len(valid) == 0 {
				continue
			}

			sortByPriority(valid)
			best := valid[0]

			logKernelSelection(op, sig, best)

			// Resolve initial effective requirements
			override, hasOverride := getRequirementOverride(sig)

			// Start with manifest defaults
			minN := best.MinN
			requiredFlags := best.RequiredFlags

			// Apply overrides if present (Partial Update Logic)
			if hasOverride {
				if override.MinN != nil {
					minN = *override.MinN
				}
				if override.RequiredFlags != nil {
					requiredFlags = *override.RequiredFlags
				}
			}

			// Optimization: If no runtime constraints exist, store raw function pointer
			if minN == 0 && requiredFlags == 0 {
				storeInTable(op, sig, best.Func)
				continue
			}

			// If constraints exist (either from manifest or override), wrap it.
			// We capture the initial state, but the wrapper checks for dynamic updates.
			initialMinN := minN
			initialFlags := requiredFlags

			constrainedFn := func(frame *internal.BlazeKernelFrame) {
				effectiveMinN := initialMinN
				effectiveFlags := initialFlags

				runtimeOverride, hasRuntimeOverride := getRequirementOverride(sig)
				if hasRuntimeOverride {
					if runtimeOverride.MinN != nil {
						effectiveMinN = *runtimeOverride.MinN
					}
					if runtimeOverride.RequiredFlags != nil {
						effectiveFlags = *runtimeOverride.RequiredFlags
					}
				}

				// 2. Fast Path check inside wrapper
				if effectiveMinN == 0 && effectiveFlags == 0 {
					best.Func(frame)
					return
				}

				// 3. Validation Logic
				if frame.Dim[0] >= effectiveMinN && (frame.Flags&effectiveFlags) == effectiveFlags {
					best.Func(frame)
				} else {
					frame.Flags |= internal.Flag_FailedConstraint
				}
			}
			storeInTable(op, sig, constrainedFn)
		}
	}
}

func resolveFast(op core.BlazeOperationID, sig uint32) core.BlazeExecutionFn {
	ptr := dispatchTable[op]

	if ptr == nil {
		return nil
	}

	dMap := *(*dispatchMap)(ptr)

	return dMap[sig]
}

func groupManifests(registry []internal.KernelManifest) map[core.BlazeOperationID]map[uint32][]internal.KernelManifest {
	grouped := make(map[core.BlazeOperationID]map[uint32][]internal.KernelManifest)

	for _, m := range registry {
		if _, ok := grouped[m.Op]; !ok {
			grouped[m.Op] = make(map[uint32][]internal.KernelManifest)
		}

		sig := internal.PackSignature(m.Output, m.Inputs...)
		grouped[m.Op][sig] = append(grouped[m.Op][sig], m)
	}

	return grouped
}

func filterByISA(candidates []internal.KernelManifest, cpuFlags internal.CPUFlags) []internal.KernelManifest {
	var valid []internal.KernelManifest

	for _, c := range candidates {
		if (c.RequiredISA & cpuFlags) == c.RequiredISA {
			valid = append(valid, c)
		}
	}

	return valid
}

func sortByPriority(candidates []internal.KernelManifest) {
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].Priority > candidates[j].Priority
	})
}

func storeInTable(op core.BlazeOperationID, sig uint32, fn core.BlazeExecutionFn) {
	ptr := dispatchTable[op]

	var dMap dispatchMap

	if ptr == nil {
		dMap = make(dispatchMap)
	} else {
		dMap = *(*dispatchMap)(ptr)
	}

	dMap[sig] = fn
	dispatchTable[op] = unsafe.Pointer(&dMap)
}

func getRequirementOverride(sig uint32) (kernelRequirements, bool) {
	if requirementOverrides == nil {
		return kernelRequirements{}, false
	}
	override, ok := requirementOverrides[sig]
	return override, ok
}

// --------------------------------------------------- API

/*
BlazeTryExecuteOperation locates the optimal kernel for the given operation and signature, then executes it using the provided frame.

It returns whether a suitable kernel was executed or not.
*/
func BlazeTryExecuteOperation(operation core.BlazeOperationID, frame *internal.BlazeKernelFrame, outDType core.BlazeDType, inputDTypes ...core.BlazeDType) bool {
	sig := internal.PackSignature(outDType, inputDTypes...)
	fn := resolveFast(operation, sig)

	if fn == nil {
		return false
	}

	frame.Flags &= ^internal.Flag_FailedConstraint

	fn(frame)

	return (frame.Flags & internal.Flag_FailedConstraint) == 0
}

/*
	BlazeSIMDDispatchKernelOverride replaces the kernel for a specific operation

and type combination.

It returns the previous kernel, allowing for temporary overrides
during tests (deferring the restoration).
*/
func BlazeSIMDDispatchKernelOverride(
	op core.BlazeOperationID,
	out core.BlazeDType,
	fn core.BlazeExecutionFn,
	inputs ...core.BlazeDType,
) core.BlazeExecutionFn {
	sig := internal.PackSignature(out, inputs...)

	ptr := dispatchTable[op]
	if ptr == nil {
		dMap := make(dispatchMap)
		dispatchTable[op] = unsafe.Pointer(&dMap)
		ptr = dispatchTable[op]
	}

	dMap := *(*dispatchMap)(ptr)
	oldFn := dMap[sig]

	if fn == nil {
		delete(dMap, sig)
	} else {
		dMap[sig] = fn
	}

	return oldFn
}

/*
	BlazeSIMDDispatchKernelRemove removes a kernel, forcing the system

to return false on TryExecute for that specific signature.
*/
func BlazeSIMDDispatchKernelRemove(
	op core.BlazeOperationID,
	out core.BlazeDType,
	inputs ...core.BlazeDType,
) core.BlazeExecutionFn {
	return BlazeSIMDDispatchKernelOverride(op, out, nil, inputs...)
}

/*
BlazeSIMDDispatchOverrideRequirements sets requirement overrides for a specific kernel signature.

This allows overriding MinN and RequiredFlags for kernels, which is useful for benchmarks
to ensure ASM kernels execute regardless of dimension constraints.

The override persists until explicitly cleared or the dispatch table is reinitialized.

Passing nil for a parameter means "do not override this specific requirement".
*/
func BlazeSIMDDispatchOverrideRequirements(
	op core.BlazeOperationID,
	out core.BlazeDType,
	minN *uint64,
	requiredFlags *internal.Flags,
	inputs ...core.BlazeDType,
) {
	sig := internal.PackSignature(out, inputs...)

	if requirementOverrides == nil {
		requirementOverrides = make(map[uint32]kernelRequirements)
	}

	current := requirementOverrides[sig]

	if minN != nil {
		current.MinN = minN
	}
	if requiredFlags != nil {
		current.RequiredFlags = requiredFlags
	}

	requirementOverrides[sig] = current
}

/*
BlazeSIMDDispatchClearRequirementOverride clears the requirement override for a specific kernel signature.
*/
func BlazeSIMDDispatchClearRequirementOverride(
	op core.BlazeOperationID,
	out core.BlazeDType,
	inputs ...core.BlazeDType,
) {
	sig := internal.PackSignature(out, inputs...)

	if requirementOverrides != nil {
		delete(requirementOverrides, sig)
	}
}

/*
BlazeSIMDDispatchClearAllRequirementOverrides clears all requirement overrides.

This is useful for test cleanup to prevent overrides from affecting subsequent tests.
*/
func BlazeSIMDDispatchClearAllRequirementOverrides() {
	requirementOverrides = nil
}

// Helper functions for kernel selection logging

func logKernelSelection(op core.BlazeOperationID, sig uint32, manifest internal.KernelManifest) {
	opName := core.BlazeOperationIDString(op)
	sigStr := formatSignature(manifest.Output, manifest.Inputs)
	isaStr := formatISA(manifest.RequiredISA)
	flagsStr := formatFlags(manifest.RequiredFlags)
	kernelName := getKernelName(manifest.Func)

	echo.On(core.BlazeUUID).
		Field("op", opName).
		Field("sig", sigStr).
		Field("kernel", kernelName).
		Field("isa", isaStr).
		Field("flags", flagsStr).
		Field("minN", manifest.MinN).
		Field("priority", manifest.Priority).
		Debug("selected kernel")
}

func formatSignature(out core.BlazeDType, inputs []core.BlazeDType) string {
	outStr := formatDType(out)
	if len(inputs) == 0 {
		return outStr
	}

	var inStrs []string
	for _, in := range inputs {
		inStrs = append(inStrs, formatDType(in))
	}

	return outStr + "<-" + strings.Join(inStrs, ",")
}

func formatDType(dt core.BlazeDType) string {
	switch dt {
	case core.DTypeF64:
		return "F64"
	case core.DTypeF32:
		return "F32"
	case core.DTypeI64:
		return "I64"
	case core.DTypeI32:
		return "I32"
	case core.DTypeI16:
		return "I16"
	case core.DTypeI8:
		return "I8"
	case core.DTypeU64:
		return "U64"
	case core.DTypeU32:
		return "U32"
	case core.DTypeU16:
		return "U16"
	case core.DTypeU8:
		return "U8"
	default:
		return "Unknown"
	}
}

func formatISA(isa internal.CPUFlags) string {
	if isa == internal.ISA_Generic {
		return "Generic"
	}

	var parts []string
	if isa&internal.ISA_SSE42 != 0 {
		parts = append(parts, "SSE42")
	}
	if isa&internal.ISA_AVX != 0 {
		parts = append(parts, "AVX")
	}
	if isa&internal.ISA_AVX2 != 0 {
		parts = append(parts, "AVX2")
	}
	if isa&internal.ISA_FMA3 != 0 {
		parts = append(parts, "FMA3")
	}
	if isa&internal.ISA_AVX512F != 0 {
		parts = append(parts, "AVX512F")
	}
	if isa&internal.ISA_AVX512BW != 0 {
		parts = append(parts, "AVX512BW")
	}
	if isa&internal.ISA_AVX512DQ != 0 {
		parts = append(parts, "AVX512DQ")
	}
	if isa&internal.ISA_AVX512VL != 0 {
		parts = append(parts, "AVX512VL")
	}
	if isa&internal.ISA_NEON != 0 {
		parts = append(parts, "NEON")
	}
	if isa&internal.ISA_SVE != 0 {
		parts = append(parts, "SVE")
	}

	if len(parts) == 0 {
		return "Generic"
	}
	return strings.Join(parts, "|")
}

func formatFlags(flags internal.Flags) string {
	if flags == 0 {
		return "None"
	}

	var parts []string
	if flags&internal.Flag_Aligned32 != 0 {
		parts = append(parts, "Aligned32")
	}
	if flags&internal.Flag_Contiguous != 0 {
		parts = append(parts, "Contiguous")
	}
	if flags&internal.Flag_Accumulate != 0 {
		parts = append(parts, "Accumulate")
	}

	if len(parts) == 0 {
		return "None"
	}
	return strings.Join(parts, "|")
}

func getKernelName(fn core.BlazeExecutionFn) string {
	if fn == nil {
		return "nil"
	}

	name := foundation.GetFunctionName(fn)
	// Extract just the function name from the full path
	// e.g., "blaze/reduce.VectorSumF32iF64o__AVX2" -> "VectorSumF32iF64o__AVX2"
	parts := strings.Split(name, ".")
	if len(parts) > 0 {
		return parts[len(parts)-1]
	}
	return name
}
