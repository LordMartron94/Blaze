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

/*
BlazeStructureMatrixMultiplyF32 computes standard matrix multiplication C = A × B in float32 precision.

Standard matrix multiplication computes: C[i,j] = Σ(A[i,k] * B[k,j]) for k in [0, n)
where A is m×n, B is n×p, and C is m×p.

Use cases:
- Linear algebra operations
- Computing X^T X for covariance matrices
- Matrix transformations
- Statistical computations (multivariate analysis)

Time complexity: O(m * n * p) - standard matrix multiplication
Space complexity: O(1) - only accumulator variables used

Prerequisites:
- Matrix A must be m×n
- Matrix B must be n×p
- Matrix C (output) must be m×p
- All matrices must be valid and initialized

Edge cases:
- Panics if dimensions don't match (A.cols != B.rows)
- Panics if output matrix dimensions don't match (C.rows != A.rows or C.cols != B.cols)
- Works with any numeric type (int, float32, float64, etc.)

The implementation uses nested loops with cache-friendly access patterns.
Future optimizations could include SIMD vectorization or BLAS integration.
*/
func BlazeStructureMatrixMultiplyF32[T, U foundation.Numeric](
	matrixAAddr, matrixBAddr, matrixCAddr memcore.MarkRaw,
) {
	aRows := memstruct.MatrixRowsGet[T](matrixAAddr)
	aCols := memstruct.MatrixColsGet[T](matrixAAddr)
	bRows := memstruct.MatrixRowsGet[U](matrixBAddr)
	bCols := memstruct.MatrixColsGet[U](matrixBAddr)
	cRows := memstruct.MatrixRowsGet[float32](matrixCAddr)
	cCols := memstruct.MatrixColsGet[float32](matrixCAddr)

	// Validate dimensions: A (m×n) × B (n×p) → C (m×p)
	if aCols != bRows {
		panic(fmt.Errorf(
			"BlazeStructureMatrixMultiplyF32: dimension mismatch (A.cols=%d != B.rows=%d)",
			aCols, bRows,
		))
	}
	if cRows != aRows || cCols != bCols {
		panic(fmt.Errorf(
			"BlazeStructureMatrixMultiplyF32: output dimension mismatch (C=%dx%d, expected=%dx%d)",
			cRows, cCols, aRows, bCols,
		))
	}

	// Standard matrix multiplication: C[i,j] = Σ(A[i,k] * B[k,j])
	for i := uint64(0); i < aRows; i++ {
		for j := uint64(0); j < bCols; j++ {
			var sum float32 = 0
			for k := uint64(0); k < aCols; k++ {
				aVal := memstruct.MatrixItemGetAtUnsafe[T](matrixAAddr, i, k)
				bVal := memstruct.MatrixItemGetAtUnsafe[U](matrixBAddr, k, j)
				sum += float32(aVal) * float32(bVal)
			}
			memstruct.MatrixSetAtUnsafe(matrixCAddr, i, j, sum)
		}
	}
}

/*
BlazeStructureMatrixMultiplyF64 computes standard matrix multiplication C = A × B in float64 precision.

Standard matrix multiplication computes: C[i,j] = Σ(A[i,k] * B[k,j]) for k in [0, n)
where A is m×n, B is n×p, and C is m×p.

Use cases:
- Linear algebra operations
- Computing X^T X for covariance matrices
- Matrix transformations
- Statistical computations (multivariate analysis)

Time complexity: O(m * n * p) - standard matrix multiplication
Space complexity: O(1) - only accumulator variables used

Prerequisites:
- Matrix A must be m×n
- Matrix B must be n×p
- Matrix C (output) must be m×p
- All matrices must be valid and initialized

Edge cases:
- Panics if dimensions don't match (A.cols != B.rows)
- Panics if output matrix dimensions don't match (C.rows != A.rows or C.cols != B.cols)
- Works with any numeric type (int, float32, float64, etc.)

The implementation uses nested loops with cache-friendly access patterns.
Future optimizations could include SIMD vectorization or BLAS integration.
*/
func BlazeStructureMatrixMultiplyF64[T, U foundation.Numeric](
	matrixAAddr, matrixBAddr, matrixCAddr memcore.MarkRaw,
) {
	aRows := memstruct.MatrixRowsGet[T](matrixAAddr)
	aCols := memstruct.MatrixColsGet[T](matrixAAddr)
	bRows := memstruct.MatrixRowsGet[U](matrixBAddr)
	bCols := memstruct.MatrixColsGet[U](matrixBAddr)
	cRows := memstruct.MatrixRowsGet[float64](matrixCAddr)
	cCols := memstruct.MatrixColsGet[float64](matrixCAddr)

	// Validate dimensions: A (m×n) × B (n×p) → C (m×p)
	if aCols != bRows {
		panic(fmt.Errorf(
			"BlazeStructureMatrixMultiplyF64: dimension mismatch (A.cols=%d != B.rows=%d)",
			aCols, bRows,
		))
	}
	if cRows != aRows || cCols != bCols {
		panic(fmt.Errorf(
			"BlazeStructureMatrixMultiplyF64: output dimension mismatch (C=%dx%d, expected=%dx%d)",
			cRows, cCols, aRows, bCols,
		))
	}

	// Standard matrix multiplication: C[i,j] = Σ(A[i,k] * B[k,j])
	for i := uint64(0); i < aRows; i++ {
		for j := uint64(0); j < bCols; j++ {
			var sum float64 = 0
			for k := uint64(0); k < aCols; k++ {
				aVal := memstruct.MatrixItemGetAtUnsafe[T](matrixAAddr, i, k)
				bVal := memstruct.MatrixItemGetAtUnsafe[U](matrixBAddr, k, j)
				sum += float64(aVal) * float64(bVal)
			}
			memstruct.MatrixSetAtUnsafe(matrixCAddr, i, j, sum)
		}
	}
}
