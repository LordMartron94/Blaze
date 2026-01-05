# Blaze

Ultra-fast numeric computation library built on manual memory management.

## Overview

`blaze` is a high-performance numerical computing library for Go that provides vectorized operations, matrix computations, reductions, and scalar operations. It uses the manual memory model from `memcore`, `memforge`, `memarch`, and `memstruct` to achieve zero garbage collection overhead and predictable performance.

## Design Philosophy

- **Zero GC Overhead**: All operations work on manually managed memory
- **Type Generic**: Works with all numeric types (`foundation.Numeric`)
- **Functional API**: Operations are stateless functions, not methods
- **Multiple Precisions**: Supports both `float32` and `float64` precision for operations
- **Cache Friendly**: Optimized memory access patterns

## Packages

### `blaze/scalar`

Operations that apply a scalar value to each element of a vector or matrix.

**Vector Operations:**
- `BlazeScalarVectorAddF32` / `BlazeScalarVectorAddF64` - Add scalar to each element
- `BlazeScalarVectorSubtractF32` / `BlazeScalarVectorSubtractF64` - Subtract scalar from each element
- `BlazeScalarVectorMultiplyF32` / `BlazeScalarVectorMultiplyF64` - Multiply each element by scalar
- `BlazeScalarVectorDivideF32` / `BlazeScalarVectorDivideF64` - Divide each element by scalar
- `BlazeScalarVectorSetAllSequence` - Fill vector with arithmetic sequence (value[i] = initial + i*step)
- `BlazeScalarVectorClamp` - Clamp each element to [min, max] range (in-place)

**Matrix Operations:**
- `BlazeScalarMatrixAddF32` / `BlazeScalarMatrixAddF64` - Add scalar to each element
- `BlazeScalarMatrixSubtractF32` / `BlazeScalarMatrixSubtractF64` - Subtract scalar from each element
- `BlazeScalarMatrixMultiplyF32` / `BlazeScalarMatrixMultiplyF64` - Multiply each element by scalar
- `BlazeScalarMatrixDivideF32` / `BlazeScalarMatrixDivideF64` - Divide each element by scalar
- `BlazeScalarMatrixSetAllSequence` - Fill matrix with arithmetic sequence (value[i] = initial + i*step)
- `BlazeScalarMatrixClamp` - Clamp each element to [min, max] range (in-place)

**Example:**
```go
import "blaze/scalar"

// Multiply each element by 2.5 (using float32 precision)
scalar.BlazeScalarVectorMultiplyF32[int](
    inputVectorMark,  // source
    outputVectorMark, // destination
    2.5,              // scalar
)

// Add 10 to each element (using float64 precision)
scalar.BlazeScalarVectorAddF64[float64](
    inputVectorMark,
    outputVectorMark,
    10.0,
)
```

### `blaze/elementwise`

Operations performed element-by-element across multiple vectors or matrices.

**Important:** Element-wise operations perform operations on corresponding elements at the same position. For example, element-wise matrix multiplication multiplies `A[i,j] * B[i,j]` for each position. This is different from standard matrix multiplication (see `blaze/structure`).

**Vector Operations:**
- `BlazeElementWiseVectorAddF32` / `BlazeElementWiseVectorAddF64` - Element-wise addition
- `BlazeElementWiseVectorSubtractF32` / `BlazeElementWiseVectorSubtractF64` - Element-wise subtraction
- `BlazeElementWiseVectorMultiplyF32` / `BlazeElementWiseVectorMultiplyF64` - Element-wise multiplication
- `BlazeElementWiseVectorDivideF32` / `BlazeElementWiseVectorDivideF64` - Element-wise division

**Matrix Operations:**
- `BlazeElementWiseMatrixAddF32` / `BlazeElementWiseMatrixAddF64` - Element-wise addition (C[i,j] = A[i,j] + B[i,j])
- `BlazeElementWiseMatrixSubtractF32` / `BlazeElementWiseMatrixSubtractF64` - Element-wise subtraction (C[i,j] = A[i,j] - B[i,j])
- `BlazeElementWiseMatrixDivideF32` / `BlazeElementWiseMatrixDivideF64` - Element-wise division (C[i,j] = A[i,j] / B[i,j])

**Note:** Standard matrix multiplication (C = A × B) and matrix-vector multiplication are in `blaze/structure`, not `blaze/elementwise`.

**Example:**
```go
import "blaze/elementwise"

// Add two vectors element-wise: output = vec1 + vec2
elementwise.BlazeElementWiseVectorAddF64[float64](
    vec1Mark,        // first input
    vec2Mark,        // second input
    outputVectorMark, // destination
)

// Element-wise matrix addition: output[i,j] = mat1[i,j] + mat2[i,j]
elementwise.BlazeElementWiseMatrixAddF32[float32, float32](
    mat1Mark,
    mat2Mark,
    outputMatrixMark,
)
```

### `blaze/reduce`

Operations that reduce a collection of values to a single value.

**Operations:**
- `BlazeReduceVectorSumF32` / `BlazeReduceVectorSumF64` - Sum all elements
- `BlazeReduceVectorSumSquaredF32` / `BlazeReduceVectorSumSquaredF64` - Sum of squares (for magnitude calculations)
- `BlazeReduceVectorMin` - Find minimum value
- `BlazeReduceVectorMax` - Find maximum value
- `BlazeReduceVectorMinMax` - Find both minimum and maximum values in a single pass
- `BlazeReduceVectorMeanF32` / `BlazeReduceVectorMeanF64` - Calculate arithmetic mean (average)
- `BlazeReduceDotProductF32` / `BlazeReduceDotProductF64` - Compute dot product between two vectors
- `BlazeReduceCovarianceF32` / `BlazeReduceCovarianceF64` - Compute covariance between two vectors (requires pre-computed means)

**Example:**
```go
import "blaze/reduce"

// Calculate sum of all elements
sum := reduce.BlazeReduceVectorSumF64[float64](vectorMark)

// Find minimum value
min := reduce.BlazeReduceVectorMin[int](vectorMark)

// Find maximum value
max := reduce.BlazeReduceVectorMax[float32](vectorMark)

// Find both min and max in one pass
min, max := reduce.BlazeReduceVectorMinMax[float64](vectorMark)

// Calculate mean
mean := reduce.BlazeReduceVectorMeanF64[float64](vectorMark)

// Calculate dot product
dotProduct := reduce.BlazeReduceDotProductF64[float64, float64](vec1Mark, vec2Mark)

// Calculate covariance (requires pre-computed means)
meanA := reduce.BlazeReduceVectorMeanF64[float64](vecAMark)
meanB := reduce.BlazeReduceVectorMeanF64[float64](vecBMark)
covariance := reduce.BlazeReduceCovarianceF64[float64, float64](vecAMark, vecBMark, meanA, meanB)
```

### `blaze/compare`

Comparison operations between vectors.

**Operations:**
- `BlazeCompareVectorEqualTo` - Element-wise equality comparison with tolerance
- `BlazeCompareVectorGreaterThan` - Element-wise greater-than comparison
- `BlazeCompareVectorGreaterThanOrEqualTo` - Element-wise greater-than-or-equal comparison
- `BlazeCompareVectorSmallerThan` - Element-wise less-than comparison
- `BlazeCompareVectorSmallerThanOrEqualTo` - Element-wise less-than-or-equal comparison
- `BlazeCompareVectorSparseJaccardWeightedSimilarity` - Compute weighted Jaccard similarity for sparse vectors

**Example:**
```go
import "blaze/compare"

// Compare two vectors element-wise, storing boolean results
compare.BlazeCompareVectorGreaterThan[int, int](
    vec1Mark,
    vec2Mark,
    resultArrayMark, // stores bool results (must be Array[bool])
)

// Compare with tolerance for floating-point equality
compare.BlazeCompareVectorEqualTo[float64, float64](
    vec1Mark,
    vec2Mark,
    resultArrayMark,
    1e-9, // tolerance
)

// Compute weighted Jaccard similarity for sparse vectors
similarity := compare.BlazeCompareVectorSparseJaccardWeightedSimilarity[int, int](
    vecAMark,
    vecBMark,
    vecAWeights, // []float64
    vecBWeights, // []float64
    1e-6,        // tolerance for ID matching
)
```

### `blaze/metric`

Mathematical metrics and distance calculations.

**Operations:**
- `BlazeMetricVectorMagnitudeF32` / `BlazeMetricVectorMagnitudeF64` - Calculate vector magnitude (L2 norm)
- `BlazeMetricVectorNormalizedF32` / `BlazeMetricVectorNormalizedF64` - Normalize vector to unit length (magnitude = 1)

**Example:**
```go
import "blaze/metric"

// Calculate vector magnitude
magnitude := metric.BlazeMetricVectorMagnitudeF64[float64](vectorMark)

// Normalize vector to unit length
metric.BlazeMetricVectorNormalizedF64[float64](
    inputVectorMark,
    outputVectorMark,
)
```

### `blaze/structure`

Matrix-specific structural operations for linear algebra.

**Operations:**
- `BlazeStructureMatrixTranspose` - Transpose a matrix
- `BlazeStructureMatrixMultiplyF32` / `BlazeStructureMatrixMultiplyF64` - Standard matrix multiplication (C = A × B, where C[i,j] = Σ(A[i,k] * B[k,j]))
- `BlazeStructureMatrixMultiplyVectorF32` / `BlazeStructureMatrixMultiplyVectorF64` - Standard matrix-vector multiplication (result[i] = Σ(matrix[i,j] * vector[j]))

**Important:** These are standard linear algebra operations, not element-wise operations. For element-wise operations (where operations are performed on corresponding elements), see `blaze/elementwise`.

**Example:**
```go
import "blaze/structure"

// Transpose matrix: output = input^T
structure.BlazeStructureMatrixTranspose[float64](
    inputMatrixMark,
    outputMatrixMark,
)

// Standard matrix multiplication: output = mat1 × mat2
structure.BlazeStructureMatrixMultiplyF32[float32, float32](
    mat1Mark,
    mat2Mark,
    outputMatrixMark,
)

// Standard matrix-vector multiplication: output = matrix * vector
err := structure.BlazeStructureMatrixMultiplyVectorF64[float64, float64](
    matrixMark,
    vectorMark,
    outputVectorMark,
)
```

### `blaze/math`

Transcendental mathematical functions (numeric primitives).

This package provides mathematical functions that are building blocks for numerical computation.
These functions have no inherent domain-specific meaning and are pure calculus and numerical analysis primitives.

**Operations:**
- `BlazeMathBetaRegularizedIncompleteF32` / `BlazeMathBetaRegularizedIncompleteF64` - Regularized incomplete beta function I_x(a, b)
- `BlazeMathLogGammaAbsF32` / `BlazeMathLogGammaAbsF64` - Natural logarithm of absolute value of Gamma function: ln(|Γ(x)|)
- `BlazeMathGammaF32` / `BlazeMathGammaF64` - Gamma function: Γ(x)

**Example:**
```go
import "blaze/math"

// Compute I_0.5(2.0, 3.0) - regularized incomplete beta function
result := math.BlazeMathBetaRegularizedIncompleteF64(0.5, 2.0, 3.0)

// Compute ln(|Γ(5.0)|) - log gamma
logGamma := math.BlazeMathLogGammaAbsF64(5.0)

// Compute Γ(5.0) - gamma function
gamma := math.BlazeMathGammaF64(5.0)
```

**Note:** These numeric primitives are used by higher-level libraries (e.g., `statarch`) for domain-specific operations, but the functions themselves remain domain-agnostic.

## Precision Variants

Many operations provide precision variants:

1. **F32** (e.g., `BlazeScalarVectorAddF32`): Forces `float32` precision internally
2. **F64** (e.g., `BlazeScalarVectorAddF64`): Forces `float64` precision internally

Use F32/F64 variants when you need consistent precision regardless of input type, or when you want to control precision for performance/accuracy trade-offs. Some operations (like min/max) work directly with the input type without precision variants.

## Complete Example

```go
package main

import (
    "memcore"
    "memforge"
    "memarch"
    "blaze/scalar"
    "blaze/elementwise"
    "blaze/reduce"
)

func main() {
    // Create allocator
    allocator := memforge.FixedLinearAllocatorCreate(uint64(memcore.MegaByte))
    defer memforge.FixedLinearAllocatorDestroy(allocator)

    allocFn := func(sizeBytes, alignment uint64) memcore.MarkRaw {
        return memforge.FixedLinearAllocatorMalloc(allocator, sizeBytes, alignment)
    }

    // Create vectors
    vec1Mark, _ := memarch.MemArchVectorCreate[float64](allocFn, 1000)
    vec2Mark, _ := memarch.MemArchVectorCreate[float64](allocFn, 1000)
    resultMark, _ := memarch.MemArchVectorCreate[float64](allocFn, 1000)

    // Initialize vec1 with values (using memstruct)
    // ... populate vec1 ...

    // Scale vec1 by 2.5, store in result
    scalar.BlazeScalarVectorMultiplyF64[float64](vec1Mark, resultMark, 2.5)

    // Add vec1 and vec2 element-wise, store in result
    elementwise.BlazeElementWiseVectorAddF64[float64](vec1Mark, vec2Mark, resultMark)

    // Calculate sum of result
    sum := reduce.BlazeReduceVectorSumF64[float64](resultMark)
    _ = sum
}
```

## Performance Characteristics

- **Vectorized Operations**: Most operations process elements sequentially with minimal overhead
- **Cache Efficiency**: Uses contiguous memory from `memstruct.Vector` and `memstruct.Matrix`
- **Zero Allocations**: No heap allocations during computation
- **Type Conversions**: F32/F64 variants may involve type conversions; use appropriate precision for your needs

## Integration

`blaze` is built on top of:

```
memcore → memforge → memarch → memstruct → blaze
```

It requires:
- `memstruct.Vector` for vector operations
- `memstruct.Matrix` for matrix operations
- Properly initialized vectors/matrices (use `memarch` to create them)

## Use Cases

- **Machine Learning**: Feature scaling, normalization, distance calculations
- **Scientific Computing**: Numerical simulations, signal processing
- **Game Development**: 3D graphics, physics calculations
- **Financial Computing**: Risk calculations, portfolio optimization
- **Data Processing**: ETL pipelines, aggregations, transformations

## Safety Guidelines

⚠️ **Important:**

1. **Input Validation**: Ensure vectors/matrices have compatible dimensions for operations
2. **Memory Lifetime**: Input and output vectors/matrices must be valid during operations
3. **Type Consistency**: Use consistent numeric types (mixing may cause unexpected behavior)
4. **Precision**: Be aware of precision loss when using F32 variants with large numbers

## Future Directions

`blaze` is designed for extensibility. Potential future additions:
- More advanced matrix operations (LU decomposition, eigenvalues)
- GPU acceleration support
- Sparse matrix support
- Parallel computation utilities

## Why Blaze?

Traditional Go numerical libraries often:
- Trigger garbage collection during computation
- Have unpredictable performance characteristics
- Don't provide fine-grained control over precision
- Require copying data between different representations

`blaze` provides:
- Zero GC overhead
- Predictable, cache-friendly performance
- Explicit precision control
- Direct operation on manual memory structures

Use `blaze` when you need high-performance numerical computations with deterministic memory usage.
