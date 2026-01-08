# Blaze Benchmark Report: Vector Dot Product

**Date:** January 2026
**Hardware Target:** AVX2 (x86_64)
**Baseline:** Go 1.2x Compiler (Auto-vectorization)

## Executive Summary

The Blaze SIMD kernels demonstrate a decisive performance advantage over the native Go compiler for all Dot Product operations involving medium-to-large vectors ($N >= 384$).

* **Peak Throughput:** The kernels successfully saturate the memory bandwidth, peaking at **~86 GB/s** for mixed-precision operations.
* **Speedup:** Asymptotic speedups range from **3x** (Standard F64) to **8x** (F32 Expansion).
* **Latency:** The `BlazeKernelFrame` abstraction introduces a constant overhead of ~80ns, making the Go fallback faster for small vectors ($N < 256$).

---

## 1. Standard Precision (`F64 · F64 → F64`)

This is the standard scientific computing workload. The SIMD kernel leverages `VFMADD231PD` with fused memory operands to maximize arithmetic density.

| Dimension ($N$) | Go (ns) | Blaze ASM (ns) | Speedup | Status |
| :--- | :--- | :--- | :--- | :--- |
| **128** | 91 | 173 | *0.52x* | 🔴 Overhead Dominated |
| **384** | 235 | 188 | **1.25x** | 🟢 Crossover Point |
| **1,024** | 613 | 222 | **2.76x** | 🚀 Bandwidth Limited |
| **100,000** | 57,790 | 19,190 | **3.01x** | 🔥 Peak (78 GB/s) |
| **1,000,000** | 660,410 | 326,753 | **2.02x** | ⚠️ RAM Latency Bound |

**Observation:** The Go compiler generates decent scalar code but fails to unroll loops effectively, capping out at ~25 GB/s. The ASM kernel hits ~78 GB/s, effectively maximizing the L2/L3 cache bandwidth.

---

## 2. Expansion Precision (`F32 · F32 → F64`)

This kernel accumulates single-precision data into a double-precision sum to prevent precision loss. This effectively benchmarks the CPU's ability to handle `VCVTPS2PD` (Vector Convert) instructions in a tight loop.

| Dimension ($N$) | Go (ns) | Blaze ASM (ns) | Speedup | Status |
| :--- | :--- | :--- | :--- | :--- |
| **128** | 130 | 179 | *0.72x* | 🔴 Overhead Dominated |
| **384** | 355 | 207 | **1.71x** | 🟢 Crossover Point |
| **1,024** | 924 | 275 | **3.36x** | 🚀 Strong Scaling |
| **100,000** | 88,874 | 11,002 | **8.08x** | 🔥 Massive Win |

**Observation:** This is the biggest win for Blaze. The Go compiler struggles immensely with the type conversion inside the loop, likely performing scalar `float32 -> float64` casts. The ASM kernel vectorizes the expansion, yielding an **8x speedup**.

---

## 3. Mixed Precision (`F32 · F64 → F64`)

A common pattern in Machine Learning (Weights X Inputs) or Statistics. This requires handling asymmetric memory strides (4 bytes vs 8 bytes).

| Dimension ($N$) | Go (ns) | Blaze ASM (ns) | Speedup | Status |
| :--- | :--- | :--- | :--- | :--- |
| **128** | 111 | 173 | *0.64x* | 🔴 Overhead Dominated |
| **384** | 295 | 197 | **1.50x** | 🟢 Crossover Point |
| **1,024** | 739 | 243 | **3.04x** | 🚀 Strong Scaling |
| **100,000** | 70,256 | 12,888 | **5.45x** | 🔥 Peak (86 GB/s) |

**Observation:** This kernel achieves the highest raw memory throughput (**86 GB/s**). Because the F32 vector requires less bandwidth, the CPU can fetch data faster relative to the compute operations, allowing for slightly higher operation throughput than the pure F64 kernel.

---

## Technical Analysis

### The Crossover Point
The "Overhead Tax" of the `BlazeKernelFrame` (stack allocation + dispatcher checks) is approximately **80ns**.
* **Small Vectors ($N < 256$):** The compute time is so short that this 80ns setup cost dominates. The Go compiler inlines the function, avoiding this cost.
* **Large Vectors ($N >= 384$):** The superior throughput of AVX2 quickly amortizes the setup cost.



### Memory Bandwidth Saturation
* **L2/L3 Cache Region ($N=100k$):** Blaze achieves **78-86 GB/s**. This is near the theoretical limit for a single core reading from L2/L3 on modern DDR4/DDR5 systems.
* **RAM Region ($N=1M$):** Performance drops to **~45-55 GB/s** as the dataset size (8MB - 16MB) exceeds the L3 cache, introducing main memory latency.

### Recommendation
The current implementation is optimal for general-purpose workloads.
* **Keep ASM** for all standard data processing.