package blaze

import (
	"blaze/core"
	"blaze/simd"
	blazetesting "blaze/testing"
	"encoding/json"
	"fmt"
	"foundation/benchmarking"
	"math/rand"
	"memcore"
	"memforge"
	"os"
	"time"
)

// -----------------------------------------------------------------------------
// Data Structures
// -----------------------------------------------------------------------------

// CalibrationTarget defines a specific kernel configuration to calibrate.
type CalibrationTarget struct {
	Name        string
	Operation   core.BlazeOperationID
	OutputDType core.BlazeDType
	InputDTypes []core.BlazeDType // Variadic to match SIMD signature

	// Data Factory: Creates inputs (1 vector for Sum, 2 for Dot, etc.)
	CreateInputs func(size int, allocFn func(uint64, uint64) memcore.MarkRaw, rng *rand.Rand) []memcore.MarkRaw

	// Execution Wrapper: Runs the kernel on the inputs
	Execute func(inputs []memcore.MarkRaw)
}

// CalibrationSuiteResult represents the aggregated output file.
type CalibrationSuiteResult struct {
	Timestamp   string                                  `json:"timestamp"`
	Description string                                  `json:"description"`
	Results     map[string]benchmarking.CrossoverResult `json:"results"`
}

// CalibrationContext is the standardized state passed to the foundation bencher.
type CalibrationContext struct {
	Inputs    []memcore.MarkRaw
	OldKernel core.BlazeExecutionFn
}

// ReportGenerator defines how the statistical summary is formatted for the console.
type ReportGenerator func(statsA, statsB []float64) string

// -----------------------------------------------------------------------------
// The Execution Engine
// -----------------------------------------------------------------------------

// RunCalibrationSuite executes the calibration process for the provided targets.
func RunCalibrationSuite(
	targets []CalibrationTarget,
	judge benchmarking.StatisticalStrategy,
	reporter ReportGenerator,
	outputFile string,
) {
	// 1. Global Initialization
	simd.BlazeSIMDDispatchInit()

	suiteResult := CalibrationSuiteResult{
		Timestamp:   time.Now().Format(time.RFC3339),
		Description: "Automated MinN Calibration for Blaze Kernels",
		Results:     make(map[string]benchmarking.CrossoverResult),
	}

	// 2. Setup Shared Allocator
	allocator := memforge.DynamicLinearAllocatorCreateFunction(
		uint64(1*memcore.MegaByte),
		blazetesting.DoubleGrowth,
	)
	defer memforge.DynamicLinearAllocatorDestroy(allocator)

	allocFn := func(sizeBytes, alignment uint64) memcore.MarkRaw {
		return memforge.DynamicLinearAllocatorMallocUnsafe(allocator, sizeBytes, alignment)
	}

	rng := rand.New(rand.NewSource(42))

	// 3. Process Targets
	for _, target := range targets {
		fmt.Printf("\n>>> Calibrating: %s <<<\n", target.Name)

		// A. Force ASM Execution (Global Override)
		// We override the requirements using the specific signature of the target
		simd.BlazeSIMDDispatchOverrideRequirements(
			target.Operation,
			target.OutputDType,
			0, 0, // Force MinN=0, Flags=0
			target.InputDTypes...,
		)

		// B. Define Benchmarking Hooks
		prepareGo := func(p benchmarking.Parameter) CalibrationContext {
			inputs := target.CreateInputs(int(p), allocFn, rng)

			// Force Go: Remove kernel from dispatcher for this signature
			old := simd.BlazeSIMDDispatchKernelRemove(
				target.Operation,
				target.OutputDType,
				target.InputDTypes...,
			)
			return CalibrationContext{Inputs: inputs, OldKernel: old}
		}

		cleanupGo := func(ctx CalibrationContext) {
			// Restore kernel
			simd.BlazeSIMDDispatchKernelOverride(
				target.Operation,
				target.OutputDType,
				ctx.OldKernel,
				target.InputDTypes...,
			)
		}

		prepareASM := func(p benchmarking.Parameter) CalibrationContext {
			// ASM is active due to the override in Step A
			return CalibrationContext{
				Inputs:    target.CreateInputs(int(p), allocFn, rng),
				OldKernel: nil,
			}
		}

		cleanupASM := func(ctx CalibrationContext) {}

		runWorkload := func(ctx CalibrationContext) {
			target.Execute(ctx.Inputs)
		}

		// C. Execute Binary Search
		result := benchmarking.FindCrossover(
			runWorkload, runWorkload,
			prepareGo, prepareASM,
			cleanupGo, cleanupASM,
			benchmarking.MeasureTime[CalibrationContext],
			judge,
			benchmarking.SearchConfig{
				MinParam: 10,
				MaxParam: 1_000_000,
				Samples:  100,
			},
		)

		// D. Save & Print
		suiteResult.Results[target.Name] = result
		printTargetReport(result, reporter)

		// E. Cleanup Global Override
		simd.BlazeSIMDDispatchClearAllRequirementOverrides()
	}

	// 4. Save to Disk
	saveSuiteReport(suiteResult, outputFile)
}

// -----------------------------------------------------------------------------
// IO Helpers
// -----------------------------------------------------------------------------

func printTargetReport(r benchmarking.CrossoverResult, reporter ReportGenerator) {
	if r.Indistinguishable {
		fmt.Printf("   Result: Indistinguishable [%d - %d]\n", r.LowIndex, r.HighIndex)
	} else {
		fmt.Printf("   Result: Crossover at ~%d elements\n", r.LowIndex)
	}

	fmt.Println("   Stats:")
	stats := reporter(r.FinalStatsA, r.FinalStatsB)
	fmt.Printf("      %s\n", stats)
}

func saveSuiteReport(report CalibrationSuiteResult, filename string) {
	file, err := os.Create(filename)
	if err != nil {
		fmt.Printf("Error creating report file: %v\n", err)
		return
	}
	defer file.Close()

	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")

	if err := encoder.Encode(report); err != nil {
		fmt.Printf("Error encoding report: %v\n", err)
		return
	}
	fmt.Printf("\nFull Calibration Suite saved to %s\n", filename)
}
