package structure

import (
	"fmt"
	"foundation"
	"memcore"
	"memstruct"
)

// BlazeStructureMatrixTranspose transposes a matrix.
// It copies src[row, col] to dst[col, row].
// If src is M×N, dst must be N×M.
// For square matrices, src and dst may be the same for in-place transpose.
func BlazeStructureMatrixTranspose[T foundation.Numeric](
	srcAddr, dstAddr memcore.MarkRaw,
) {
	srcRows := memstruct.MatrixRowsGet[T](srcAddr)
	srcCols := memstruct.MatrixColsGet[T](srcAddr)
	dstRows := memstruct.MatrixRowsGet[T](dstAddr)
	dstCols := memstruct.MatrixColsGet[T](dstAddr)

	// Verify dimensions: if src is M×N, dst must be N×M
	if srcRows != dstCols || srcCols != dstRows {
		panic(fmt.Errorf(
			"BlazeStructureMatrixTranspose: dimension mismatch (src=%dx%d, dst=%dx%d)",
			srcRows, srcCols, dstRows, dstCols,
		))
	}

	// Handle in-place transpose for square matrices
	if srcAddr == dstAddr {
		if srcRows != srcCols {
			panic("BlazeStructureMatrixTranspose: in-place transpose only supported for square matrices")
		}
		transposeInPlace[T](srcAddr, srcRows)
		return
	}

	// Standard transpose: copy src[row, col] → dst[col, row]
	for row := uint64(0); row < srcRows; row++ {
		for col := uint64(0); col < srcCols; col++ {
			val := memstruct.MatrixItemGetAtUnsafe[T](srcAddr, row, col)
			memstruct.MatrixSetAtUnsafe(dstAddr, col, row, val)
		}
	}
}

// transposeInPlace performs an in-place transpose for square matrices.
// It swaps elements across the diagonal.
func transposeInPlace[T foundation.Numeric](
	matrixAddr memcore.MarkRaw,
	size uint64,
) {
	// For square matrices, we swap elements across the diagonal
	// We only iterate over the upper triangle to avoid swapping twice
	for row := uint64(0); row < size; row++ {
		for col := row + 1; col < size; col++ {
			val1 := memstruct.MatrixItemGetAtUnsafe[T](matrixAddr, row, col)
			val2 := memstruct.MatrixItemGetAtUnsafe[T](matrixAddr, col, row)
			memstruct.MatrixSetAtUnsafe(matrixAddr, col, row, val1)
			memstruct.MatrixSetAtUnsafe(matrixAddr, row, col, val2)
		}
	}
}

