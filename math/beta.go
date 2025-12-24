package math

import (
	"math"
)

const (
	// blazeMathBetaEpsilon determines the precision of the continued fraction convergence.
	// 3.0e-7 is standard for float32, 1.0e-15 is standard for float64.
	blazeMathBetaEpsilon = 1.0e-15

	// blazeMathBetaTiny is a very small number used to prevent division by zero
	// in Lentz's method. It should be roughly the square root of the machine epsilon.
	blazeMathBetaTiny = 1.0e-30

	// blazeMathBetaMaxIterations limits the loop to prevent infinite cycles in
	// non-converging cases (rare for valid inputs).
	blazeMathBetaMaxIterations = 200
)

/*
BlazeMathBetaRegularizedIncompleteF64 computes the regularized incomplete beta function I_x(a, b).

The regularized incomplete beta function is defined as:
I_x(a, b) = B(x; a, b) / B(a, b) = (1 / B(a, b)) * ∫(from 0 to x) t^(a-1) (1-t)^(b-1) dt

This is a numeric primitive fundamental to statistical computation.

Use cases:
- Cumulative Distribution Function (CDF) for Beta distributions
- Robust Quantile Estimation (Harrell-Davis Estimator)
- Student's t-distribution, F-distribution, and Binomial distribution calculations

Time complexity: O(N) where N is the number of iterations for convergence (typically < 20)
Space complexity: O(1) - strictly iterative, zero allocation

Prerequisites:
- x must be in range [0, 1]
- a > 0
- b > 0

Algorithm:
Uses the Modified Lentz's Method for evaluating the continued fraction expansion.
This approach is numerically stable and converges rapidly, even in the tails of the distribution.
It automatically handles the symmetry property I_x(a, b) = 1 - I_{1-x}(b, a) to optimize convergence.
*/
func BlazeMathBetaRegularizedIncompleteF64(x, a, b float64) float64 {
	if x < 0 || x > 1 {
		panic("BlazeMathBetaRegularizedIncompleteF64: x must be in [0, 1]")
	}
	if a <= 0 || b <= 0 {
		panic("BlazeMathBetaRegularizedIncompleteF64: a and b must be > 0")
	}

	// Trivial cases
	if x == 0 {
		return 0.0
	}
	if x == 1 {
		return 1.0
	}

	// Symmetry transformation:
	// The continued fraction converges much faster for x < (a+1)/(a+b+2).
	// If x is larger, we use the property I_x(a,b) = 1 - I_{1-x}(b,a).
	if x > (a+1.0)/(a+b+2.0) {
		return 1.0 - BlazeMathBetaRegularizedIncompleteF64(1.0-x, b, a)
	}

	// Compute the leading factor: exp(ln(x^a * (1-x)^b / B(a,b)))
	lgammaA := BlazeMathLogGammaAbsF64(a)
	lgammaB := BlazeMathLogGammaAbsF64(b)
	lgammaAB := BlazeMathLogGammaAbsF64(a + b)

	// factor = (x^a * (1-x)^b) / (a * Beta(a,b))
	// note: Beta(a,b) = Gamma(a)*Gamma(b) / Gamma(a+b)
	// We use log-space calculation for stability.
	logFactor := a*math.Log(x) + b*math.Log(1.0-x) + lgammaAB - lgammaA - lgammaB
	factor := math.Exp(logFactor) / a

	// Evaluate the continued fraction
	cf := blazeMathBetaLentzContinuedFractionF64(x, a, b)

	return factor * cf
}

/*
BlazeMathBetaRegularizedIncompleteF32 computes the regularized incomplete beta function I_x(a, b) in float32 precision.

See BlazeMathBetaRegularizedIncompleteF64 for implementation details.
*/
func BlazeMathBetaRegularizedIncompleteF32(x, a, b float32) float32 {
	return float32(BlazeMathBetaRegularizedIncompleteF64(float64(x), float64(a), float64(b)))
}

/*
blazeMathBetaLentzContinuedFractionF64 evaluates the continued fraction part of the Incomplete Beta Function.

Standard Continued Fraction Representation:
CF = 1 / (1 + d1 / (1 + d2 / ... ))

Coefficients d_m follow the pattern (Ref: Abramowitz & Stegun 26.5.8):
For m = 0, 1, 2...

	d_{2m+1} = - (a+m)(a+b+m)x / ((a+2m)(a+2m+1))
	d_{2m}   = m(b-m)x / ((a+2m-1)(a+2m))

This function uses the Modified Lentz's Method to evaluate this fraction,
which avoids the numerical instability of naive evaluation.
*/
func blazeMathBetaLentzContinuedFractionF64(x, a, b float64) float64 {
	// Initialize Lentz's method variables
	// f is the running approximation of the continued fraction
	f := 1.0
	C := 1.0
	D := 0.0

	// Ensure D starts non-zero (standard Lentz fix)
	if math.Abs(D) < blazeMathBetaTiny {
		D = blazeMathBetaTiny
	}

	// Iterate up to max iterations
	for j := 1; j <= blazeMathBetaMaxIterations; j++ {
		var d float64
		var m float64

		// Use integer math for 'm' to avoid floating point ambiguity in index logic
		// j is 1-based index: 1, 2, 3, 4...
		// m calculation:
		// If j is odd (1, 3, 5): j=2m+1 => m = (j-1)/2
		// If j is even (2, 4, 6): j=2m   => m = j/2

		if j%2 == 0 {
			// Even step: j = 2m
			m = float64(j) / 2.0
			// Numerator: m * (b - m) * x
			// Denominator: (a + 2m - 1) * (a + 2m)
			numerator := m * (b - m) * x
			denominator := (a + 2*m - 1) * (a + 2*m)
			d = numerator / denominator
		} else {
			// Odd step: j = 2m + 1
			m = float64(j-1) / 2.0
			// Numerator: - (a + m) * (a + b + m) * x
			// Denominator: (a + 2m) * (a + 2m + 1)
			numerator := -(a + m) * (a + b + m) * x
			denominator := (a + 2*m) * (a + 2*m + 1)
			d = numerator / denominator
		}

		// Calculate C and D (Lentz update)
		D = 1.0 + d*D
		if math.Abs(D) < blazeMathBetaTiny {
			D = blazeMathBetaTiny
		}

		C = 1.0 + d/C
		if math.Abs(C) < blazeMathBetaTiny {
			C = blazeMathBetaTiny
		}

		D = 1.0 / D
		delta := C * D
		f *= delta

		// Check for convergence
		if math.Abs(delta-1.0) < blazeMathBetaEpsilon {
			return f
		}
	}

	// If we hit max iterations, return the best estimate.
	return f
}
