// Package testing provides some utilities for testing & benchmarking.
package testing

import (
	"math/rand"
	"memcore"
)

func DoubleGrowth(currentCap, needed uint64) uint64 {
	newSize := currentCap * 2
	if newSize < needed {
		newSize = needed
	}

	if newSize > uint64(1*memcore.GigaByte) {
		panic("way too much memory for a simple test")
	}

	return newSize
}

// ────────────────────────────────────────────────────────────────
//   HELPERS
// ────────────────────────────────────────────────────────────────

func GenerateRandomVectorF32(dimension uint64, rng *rand.Rand) []float32 {
	vec := make([]float32, dimension)
	for i := range vec {
		vec[i] = rng.Float32()
	}
	return vec
}

func GenerateRandomVectorF64(dimension uint64, rng *rand.Rand) []float64 {
	vec := make([]float64, dimension)
	for i := range vec {
		vec[i] = rng.Float64()
	}
	return vec
}

// ────────────────────────────────────────────────────────────────
//   MAIN SUITE
// ────────────────────────────────────────────────────────────────

const (
	DefaultMaxHeapGrowth       = 500 * 1024 * 1024  // 500MB
	DefaultMaxHeapSize         = 1024 * 1024 * 1024 // 1GB
	DefaultMemoryCheckInterval = 50_000             // Check every 50k iterations
)
