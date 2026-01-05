package math

import "math"

// Lanczos approximation coefficients (g=7, n=9)
// Common double-precision coefficient set (Godfrey / NR-style).
var blazeMathLanczosCoeffs = []float64{
	0.99999999999980993,
	676.5203681218851,
	-1259.1392167224028,
	771.32342877765313,
	-176.61502916214059,
	12.507343278686905,
	-0.13857109526572012,
	9.9843695780195716e-6,
	1.5056327351493116e-7,
}

const (
	// blazeMathGammaG is the Lanczos scaling factor g
	blazeMathGammaG = 7.0

	// blazeMathLnSqrt2Pi is ln(sqrt(2*pi))
	blazeMathLnSqrt2Pi = 0.91893853320467274178032973640561763986139747363778

	// blazeMathLnMaxFloat64 is ln(math.MaxFloat64)
	blazeMathLnMaxFloat64 = 7.09782712893383973096e2 // ~709.78271289338397
)

/*
BlazeMathLogGammaAbsF64 computes ln(|Γ(x)|) using Lanczos (g=7) with reflection for x < 0.5.

Semantics:
- Poles at x = 0, -1, -2, ... => +Inf (since ln(|Γ(x)|) diverges)
- For NaN => NaN

Accuracy:
- Typically full float64 precision for x > 0 and good accuracy elsewhere except extremely close to poles.

This is a stats-grade primitive (Beta function, distributions, etc.).
*/
func BlazeMathLogGammaAbsF64(x float64) float64 {
	if math.IsNaN(x) {
		return math.NaN()
	}

	// Poles: 0, -1, -2, ...
	if x == 0.0 || (x < 0.0 && x == math.Floor(x)) {
		return math.Inf(1)
	}

	// Reflection for x < 0.5 to improve accuracy and handle negatives.
	if x < 0.5 {
		// ln(|Γ(x)|) = ln(pi) - ln(|sin(pi x)|) - ln(|Γ(1-x)|)
		s := math.Sin(math.Pi * x)
		// If s == 0, we are at/near a pole (should be caught above, but keep safe).
		if s == 0.0 {
			return math.Inf(1)
		}
		return math.Log(math.Pi) - math.Log(math.Abs(s)) - BlazeMathLogGammaAbsF64(1.0-x)
	}

	// Lanczos: evaluate ln Γ(x) for x >= 0.5
	z := x - 1.0

	// A(z) = c0 + Σ_{i=1..n-1} c_i/(z+i)
	agg := blazeMathLanczosCoeffs[0]
	for i := 1; i < len(blazeMathLanczosCoeffs); i++ {
		agg += blazeMathLanczosCoeffs[i] / (z + float64(i))
	}

	t := z + blazeMathGammaG + 0.5

	// ln Γ(x) ≈ ln(sqrt(2π)) + ln(A(z)) + (z+0.5)ln(t) - t
	// For x>=0.5, Γ(x) is positive, so ln(|Γ(x)|) == ln(Γ(x)).
	return blazeMathLnSqrt2Pi + math.Log(agg) + (z+0.5)*math.Log(t) - t
}

/*
BlazeMathGammaF64 computes Γ(x).

Semantics:
- Poles at x = 0, -1, -2, ... => NaN
- Overflows => +Inf or -Inf (sign-preserving for negative non-integers)
- NaN in => NaN out

Implementation:
- For x > 0: exp(LogGammaAbs)
- For x < 0 (non-integer): reflection Γ(x)=π/(sin(πx)Γ(1-x)), using LogGammaAbs for magnitude
*/
func BlazeMathGammaF64(x float64) float64 {
	if math.IsNaN(x) {
		return math.NaN()
	}
	if math.IsInf(x, 1) {
		return math.Inf(1)
	}
	if math.IsInf(x, -1) {
		// Γ(-∞) oscillates in sign and magnitude is not meaningful in float64.
		return math.NaN()
	}

	// Poles: 0, -1, -2, ...
	if x == 0.0 || (x < 0.0 && x == math.Floor(x)) {
		return math.NaN()
	}

	// Positive x: Γ(x) > 0
	if x > 0.0 {
		lg := BlazeMathLogGammaAbsF64(x)
		if lg > blazeMathLnMaxFloat64 {
			return math.Inf(1)
		}
		// Underflow to 0 is fine; exp handles it.
		return math.Exp(lg)
	}

	// Negative non-integer x: use reflection with sign from sin(pi x).
	s := math.Sin(math.Pi * x)
	if s == 0.0 {
		// Should be unreachable due to pole check, but keep safe.
		return math.NaN()
	}

	// ln|Γ(1-x)| (1-x > 1 for x<0, so this is safe and accurate)
	lg1 := BlazeMathLogGammaAbsF64(1.0 - x)

	// ln|Γ(x)| = ln(pi) - ln|sin(pi x)| - ln|Γ(1-x)|
	lg := math.Log(math.Pi) - math.Log(math.Abs(s)) - lg1

	// Overflow/underflow handling for exp(lg)
	if lg > blazeMathLnMaxFloat64 {
		// Sign is opposite sign of sin(pi x) in denominator? Actually:
		// Γ(x) = π / (sin(πx) Γ(1-x))
		// Γ(1-x) > 0 for x<0, so sign(Γ(x)) = sign(π / sin(πx)) = sign(sin(πx))^{-1} => same as sign(sin) because 1/sin.
		// In IEEE, dividing by negative yields negative.
		if s < 0 {
			return math.Inf(-1)
		}
		return math.Inf(1)
	}

	val := math.Exp(lg)
	// Apply sign: Γ(x) = (π / (sin(πx) Γ(1-x))) => sign is sign(1/sin(πx))
	if s < 0 {
		return -val
	}
	return val
}

// Float32 wrappers

func BlazeMathLogGammaAbsF32(x float32) float32 {
	return float32(BlazeMathLogGammaAbsF64(float64(x)))
}

func BlazeMathGammaF32(x float32) float32 {
	return float32(BlazeMathGammaF64(float64(x)))
}
