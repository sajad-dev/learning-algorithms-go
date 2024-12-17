package loss

import "math"

func MeenAbsoluteError(args ...[]float64) float64 {
	sum := 0.0
	for i, v := range args[0] {
		sum += math.Abs(v - args[1][i])
	}
	return sum / float64(len(args[0]))
}

func MeenSquaredError(args ...[]float64) float64 {
	sum := 0.0
	for i, v := range args[0] {
		sum += math.Pow(v-args[1][i], 2)
	}
	return sum / float64(len(args[0]))
}
