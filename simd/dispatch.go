// Package simd provides optimized assembly functions for operations.
package simd

import (
	"blaze/core"
	"blaze/internal"
	"sort"
	"unsafe"
)

const (
	MaxOPS        = core.BLAZE_OPERATION_COUNT
	MaxSignatures = 4096
)

var dispatchTable [MaxOPS]unsafe.Pointer

type dispatchMap map[uint32]core.BlazeExecutionFn

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

			if best.MinN == 0 && best.RequiredFlags == 0 {
				storeInTable(op, sig, best.Func)
				continue
			}

			// If it HAS constraints, wrap it.
			constrainedFn := func(frame *internal.BlazeKernelFrame) {
				if frame.Dim[0] >= best.MinN && (frame.Flags&best.RequiredFlags) == best.RequiredFlags {
					best.Func(frame)
				} else {
					frame.Flags |= internal.Flag_FailedConstraint
				}
			}
			storeInTable(op, sig, constrainedFn)
		}
	}
}

func compileHybridKernel(candidates []internal.KernelManifest) core.BlazeExecutionFn {
	best := candidates[0]

	var fallback core.BlazeExecutionFn
	for _, c := range candidates {
		if c.MinN == 0 && c.RequiredFlags == 0 {
			fallback = c.Func
			break
		}
	}

	if fallback == nil {
		panic("Configuration Error: High-perf kernel requires MinN but no scalar fallback exists!")
	}

	return func(frame *internal.BlazeKernelFrame) {
		if (frame.Flags & best.RequiredFlags) != best.RequiredFlags {
			fallback(frame)
			return
		}

		if frame.Dim[0] < best.MinN {
			fallback(frame)
			return
		}

		best.Func(frame)
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
