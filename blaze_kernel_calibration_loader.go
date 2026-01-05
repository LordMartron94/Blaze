package blaze

import (
	"blaze/core"
	"blaze/simd"
	"echo"
	"encoding/json"
	"fmt"
	"os"
)

/*
LoadAndApplyCalibrationProfile reads a calibration suite JSON file and updates
the SIMD dispatch table with the optimal MinN values found.

It applies the *exact* crossover point found during calibration to ensure maximum
granularity and fidelity to the specific hardware's performance characteristics.
*/
func LoadAndApplyCalibrationProfile(filePath string) error {
	// 1. Read and Parse the Suite JSON
	file, err := os.Open(filePath)
	if err != nil {
		return fmt.Errorf("failed to open calibration profile: %w", err)
	}
	defer file.Close()

	var suite CalibrationSuiteResult
	if err := json.NewDecoder(file).Decode(&suite); err != nil {
		return fmt.Errorf("failed to decode calibration profile: %w", err)
	}

	// 2. Get the list of known kernels
	targets := GetStandardTargets()
	targetMap := make(map[string]CalibrationTarget, len(targets))
	for _, t := range targets {
		targetMap[t.Name] = t
	}

	appliedCount := 0

	// 3. Iterate over results and apply overrides
	for name, result := range suite.Results {
		target, exists := targetMap[name]
		if !exists {
			echo.On(core.BlazeUUID).
				Field("kernel", name).
				Warning("calibration profile contains unknown kernel, skipping")
			continue
		}

		// Use the exact measured crossover for maximum accuracy on this architecture.
		robustMinN := uint64(result.LowIndex)

		// Apply the override.
		// we pass 'nil' for requiredFlags to ensure we don't accidentally clobber
		// the architecture requirements (like alignment) defined in the manifest.
		simd.BlazeSIMDDispatchOverrideRequirements(
			target.Operation,
			target.OutputDType,
			&robustMinN, // Override MinN
			nil,         // Keep existing Flags
			target.InputDTypes...,
		)

		appliedCount++

		echo.On(core.BlazeUUID).
			Field("kernel", name).
			Field("minN", robustMinN).
			Debug("applied calibration profile")
	}

	if appliedCount == 0 {
		return fmt.Errorf("no matching kernels found in calibration profile")
	}

	return nil
}
