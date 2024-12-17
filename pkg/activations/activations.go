package activations

import "math"

func Relu(x float64) float64 {
	return math.Max(0, x)
}
func Sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}
