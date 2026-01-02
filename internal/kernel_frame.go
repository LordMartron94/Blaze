package internal

import "unsafe"

type Flags uint64

const (
	Flag_Aligned32        Flags = 1 << 0 // Input pointers are 32-byte aligned
	Flag_Contiguous       Flags = 1 << 1 // Stride is 1 element size (no gaps)
	Flag_Accumulate       Flags = 1 << 2 // Result += NewVal instead of Result = NewVal
	Flag_FailedConstraint Flags = 1 << 3
)

var CurrentKernelFrame *BlazeKernelFrame

func init() {
	CurrentKernelFrame = &BlazeKernelFrame{}
}

/*
BlazeKernelFrame is the universal data transfer object for all Blaze assembly kernels.

It acts as the single source of truth for the execution context.
The layout is optimized for 8-byte alignment and cache locality.
It is typically stack-allocated by the Go dispatcher.
*/
type BlazeKernelFrame struct {
	/* ---- Execution Bounds (The "Shape") ---- */

	/* Dim[0] is the primary loop counter. Dim[1] and Dim[2] are for Matrix/Tensor batching. */
	Dim [3]uint64

	/* ---- Configuration Flags ---- */
	/* Bitmask for runtime behavior (Aligned, Accumulate, etc.) */
	Flags Flags

	/* ---- Parameter Slots (The "Knobs") ---- */
	/* Generic scalars, bitmasks, or flags. Go must bit-cast float32/64 to uint64. */
	Params [6]uint64

	/* ---- Memory Aperture (The "Vectors") ---- */

	/* Buffer descriptors for Inputs and Outputs. Using 6 allows for complex ops (e.g., A+B+C -> D). */
	Buffers [6]BlazeBufferSlot

	/* ---- Scalar Returns (The "Reductions") ---- */

	/* Pointers to memory for scalar results like Sum, Min, or Max. These lack strides. */
	Returns [2]unsafe.Pointer

	/* ---- Opaque Context (The "Escape Hatch") ---- */

	/* Pointer to auxiliary structures like LUTs or large config blocks. */
	Opaque unsafe.Pointer

	/* ---- Extension Pad ---- */

	/* Reserved to maintain alignment and allow future growth without breaking the ABI. */
	_ [3]uint64
}

/*
	 BlazeBufferSlot describes a strided region of memory.
		It is used for both reading (Inputs) and writing (Outputs).
*/
type BlazeBufferSlot struct {
	/* Ptr is the base address of the raw data. */
	Ptr unsafe.Pointer

	/* Stride is the byte-offset between elements. Signed to allow reverse iteration. */
	Stride int64
}

/* Reset clears the frame for reuse, zeroing all sensitive pointers and bounds. */
func (f *BlazeKernelFrame) Reset() *BlazeKernelFrame {
	*f = BlazeKernelFrame{}
	return f
}

/* WithDim sets the execution dimensions. */
func (f *BlazeKernelFrame) WithDim(d0, d1, d2 uint64) *BlazeKernelFrame {
	f.Dim[0], f.Dim[1], f.Dim[2] = d0, d1, d2
	return f
}

/* WithFlags applies the provided bitmask. */
func (f *BlazeKernelFrame) WithFlags(flags Flags) *BlazeKernelFrame {
	f.Flags = flags
	return f
}

/* WithParam injects a uint64 parameter into the specified slot index. */
func (f *BlazeKernelFrame) WithParam(index int, val uint64) *BlazeKernelFrame {
	f.Params[index] = val
	return f
}

/* WithBuffer configures a specific memory aperture slot. */
func (f *BlazeKernelFrame) WithBuffer(index int, ptr unsafe.Pointer, stride int64) *BlazeKernelFrame {
	f.Buffers[index].Ptr = ptr
	f.Buffers[index].Stride = stride
	return f
}

/* WithReturn sets a scalar reduction target. */
func (f *BlazeKernelFrame) WithReturn(index int, ptr unsafe.Pointer) *BlazeKernelFrame {
	f.Returns[index] = ptr
	return f
}

/* WithOpaque sets the auxiliary context pointer. */
func (f *BlazeKernelFrame) WithOpaque(ptr unsafe.Pointer) *BlazeKernelFrame {
	f.Opaque = ptr
	return f
}
