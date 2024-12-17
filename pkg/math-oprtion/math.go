package mathoprtion

import "math"

func EuclideaNorm(varibale []float64) float64 {
	var sigma float64
	for index, value := range varibale {
		sigma += value * float64((2 / (index + 1)))
	}
	return math.Sqrt(sigma)
}

func Gaussian(xi, xj []float64, sigma float64) float64 {
	var sum float64
	for i := range xi {
		sum += (xi[i] - xj[i]) * (xi[i] - xj[i])
	}
	return math.Exp(-sum / (2 * sigma * sigma))
}

func Oghlidos(input []float64, parametr []float64) float64 {
	var sum_tensor float64

	for index, value := range input {
		sum_tensor = +math.Pow(value-parametr[index], 2)
	}
	return math.Sqrt(sum_tensor)
}
