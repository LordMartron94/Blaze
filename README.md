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
- `BlazeScalarVectorAdd` / `BlazeScalarVectorAddF32` / `BlazeScalarVectorAddF64`
- `BlazeScalarVectorSubtract` / `BlazeScalarVectorSubtractF32` / `BlazeScalarVectorSubtractF64`
- `BlazeScalarVectorMultiply` / `BlazeScalarVectorMultiplyF32` / `BlazeScalarVectorMultiplyF64`
- `BlazeScalarVectorDivide` / `BlazeScalarVectorDivideF32` / `BlazeScalarVectorDivideF64`
- `BlazeScalarVectorSetAllSequence` - Fill vector with arithmetic sequence

**Matrix Operations:**
- `BlazeScalarMatrixAdd` / `BlazeScalarMatrixAddF32` / `BlazeScalarMatrixAddF64`
- `BlazeScalarMatrixSubtract` / `BlazeScalarMatrixSubtractF32` / `BlazeScalarMatrixSubtractF64`
- `BlazeScalarMatrixMultiply` / `BlazeScalarMatrixMultiplyF32` / `BlazeScalarMatrixMultiplyF64`
- `BlazeScalarMatrixDivide` / `BlazeScalarMatrixDivideF32` / `BlazeScalarMatrixDivideF64`

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

**Vector Operations:**
- `BlazeElementwiseVectorAdd` / `BlazeElementwiseVectorAddF32` / `BlazeElementwiseVectorAddF64`
- `BlazeElementwiseVectorSubtract` / `BlazeElementwiseVectorSubtractF32` / `BlazeElementwiseVectorSubtractF64`
- `BlazeElementwiseVectorMultiply` / `BlazeElementwiseVectorMultiplyF32` / `BlazeElementwiseVectorMultiplyF64`
- `BlazeElementwiseVectorDivide` / `BlazeElementwiseVectorDivideF32` / `BlazeElementwiseVectorDivideF64`

**Matrix Operations:**
- `BlazeElementwiseMatrixAdd` / `BlazeElementwiseMatrixAddF32` / `BlazeElementwiseMatrixAddF64`
- `BlazeElementwiseMatrixSubtract` / `BlazeElementwiseMatrixSubtractF32` / `BlazeElementwiseMatrixSubtractF64`
- `BlazeElementwiseMatrixMultiply` / `BlazeElementwiseMatrixMultiplyF32` / `BlazeElementwiseMatrixMultiplyF64`
- `BlazeElementwiseMatrixDivide` / `BlazeElementwiseMatrixDivideF32` / `BlazeElementwiseMatrixDivideF64`

**Example:**
```go
import "blaze/elementwise"

// Add two vectors element-wise: output = vec1 + vec2
elementwise.BlazeElementwiseVectorAddF64[float64](
    vec1Mark,        // first input
    vec2Mark,        // second input
    outputVectorMark, // destination
)

// Multiply two matrices element-wise: output = mat1 * mat2
elementwise.BlazeElementwiseMatrixMultiplyF32[float32](
    mat1Mark,
    mat2Mark,
    outputMatrixMark,
)
```

### `blaze/reduce`

Operations that reduce a collection of values to a single value.

**Operations:**
- `BlazeReduceVectorSum` - Sum all elements
- `BlazeReduceVectorMin` - Find minimum value
- `BlazeReduceVectorMax` - Find maximum value
- `BlazeReduceVectorMean` - Calculate mean (average)

**Example:**
```go
import "blaze/reduce"

// Calculate sum of all elements
sum := reduce.BlazeReduceVectorSum[float64](vectorMark)

// Find minimum value
min := reduce.BlazeReduceVectorMin[int](vectorMark)

// Find maximum value
max := reduce.BlazeReduceVectorMax[float32](vectorMark)

// Calculate mean
mean := reduce.BlazeReduceVectorMean[float64](vectorMark)
```

### `blaze/compare`

Comparison operations between vectors.

**Operations:**
- `BlazeCompareVectorEqual` - Element-wise equality comparison
- `BlazeCompareVectorNotEqual` - Element-wise inequality comparison
- `BlazeCompareVectorGreater` - Element-wise greater-than comparison
- `BlazeCompareVectorGreaterEqual` - Element-wise greater-than-or-equal comparison
- `BlazeCompareVectorLess` - Element-wise less-than comparison
- `BlazeCompareVectorLessEqual` - Element-wise less-than-or-equal comparison

**Example:**
```go
import "blaze/compare"

// Compare two vectors element-wise, storing boolean results
compare.BlazeCompareVectorGreater[int](
    vec1Mark,
    vec2Mark,
    resultVectorMark, // stores bool results
)
```

### `blaze/metric`

Mathematical metrics and distance calculations.

**Operations:**
- `BlazeMetricVectorEuclideanDistance` - Calculate Euclidean distance between two vectors
- `BlazeMetricVectorManhattanDistance` - Calculate Manhattan (L1) distance
- `BlazeMetricVectorDotProduct` - Calculate dot product

**Example:**
```go
import "blaze/metric"

// Calculate Euclidean distance
distance := metric.BlazeMetricVectorEuclideanDistance[float64](vec1Mark, vec2Mark)

// Calculate dot product
dotProduct := metric.BlazeMetricVectorDotProduct[float64](vec1Mark, vec2Mark)
```

### `blaze/structure`

Matrix-specific structural operations.

**Operations:**
- `BlazeStructureMatrixTranspose` - Transpose a matrix
- Additional matrix operations for linear algebra

**Example:**
```go
import "blaze/structure"

// Transpose matrix: output = input^T
structure.BlazeStructureMatrixTranspose[float64](
    inputMatrixMark,
    outputMatrixMark,
)
```

## Precision Variants

Many operations provide three variants:

1. **Generic** (e.g., `BlazeScalarVectorAdd`): Uses the input type's precision
2. **F32** (e.g., `BlazeScalarVectorAddF32`): Forces `float32` precision internally
3. **F64** (e.g., `BlazeScalarVectorAddF64`): Forces `float64` precision internally

Use F32/F64 variants when you need consistent precision regardless of input type, or when you want to control precision for performance/accuracy trade-offs.

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
    elementwise.BlazeElementwiseVectorAddF64[float64](vec1Mark, vec2Mark, resultMark)

    // Calculate sum of result
    sum := reduce.BlazeReduceVectorSum[float64](resultMark)
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
