package generatedata

import "math/rand"

func GenerateData(numSamples int) ([][]float64, []float64) {
	input := [][]float64{}
	output := []float64{}
	for i := 0; i < numSamples; i++ {
		x1 := rand.Float64() * 10
		x2 := x1 * 2
		x3 := x1 * 4
		input = append(input, []float64{x1, x2, x3})
		y := x1 + x2 + x3
		output = append(output, y)
	}
	return input, output
}