package nourn

type ActivationFunctions func(float64) float64

func Nourn(input []float64, weight []float64, bias float64, activation_functions ActivationFunctions) float64 {
	var sum float64
	for i, v := range input {
		sum += v * weight[i]
	}
	return activation_functions(sum + bias)
}
