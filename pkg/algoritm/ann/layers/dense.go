package layers

import (
	"math/rand"

	"github.com/sajad-dev/learning-algorithms-go/pkg/algoritm/ann/nourn"
)

func (l *DenseLayer) DenseDerivative(X []float64, weight_num []int) float64 {
	h := 1e-3

	weight_copy := make([][][]float64, len(l.Weight))

	for i := range l.Weight {
		weight_copy[i] = make([][]float64, len(l.Weight[i]))
		for j := range l.Weight[i] {
			weight_copy[i][j] = append([]float64{}, l.Weight[i][j]...)
		}
	}

	weight_copy[weight_num[0]][weight_num[1]][weight_num[2]] += h
	return (l.DensOperrtion(X, weight_copy) - l.DensOperrtion(X, l.Weight)) / h
}

func (l *DenseLayer) GeneratePred(X []float64, weight_num []int) float64 {
	h := 1e-3

	weight_copy := make([][][]float64, len(l.Weight))
	for i := range l.Weight {
		weight_copy[i] = make([][]float64, len(l.Weight[i]))
		for j := range l.Weight[i] {
			weight_copy[i][j] = append([]float64{}, l.Weight[i][j]...)
		}
	}
	weight_copy[weight_num[0]][weight_num[1]][weight_num[2]] += h

	return l.DensOperrtion(X, weight_copy)
}

func (l *DenseLayer) GeneratePredB(X []float64, bias_num []int) float64 {
	h := 1e-3
	bias_copy := make([][]float64, len(l.Bias))
	for i := range l.Weight {
		bias_copy[i] = append([]float64{}, l.Bias[i]...)
	}

	bias_copy[bias_num[0]][bias_num[1]] += h
	return l.densOperrtionBias(X, bias_copy)
}

func (l *DenseLayer) DenseDerivativeBias(X []float64, bias_num []int) float64 {
	h := 1e-3
	bias_copy := make([][]float64, len(l.Bias))
	for i := range l.Weight {
		bias_copy[i] = append([]float64{}, l.Bias[i]...)
	}

	bias_copy[bias_num[0]][bias_num[1]] += h
	return (l.densOperrtionBias(X, bias_copy) - l.densOperrtionBias(X, l.Bias)) / h
}

func (l *DenseLayer) DensOperrtion(X []float64, weight [][][]float64) float64 {
	output := []float64{}

	for index, value := range weight[0] {
		output = append(output, nourn.Nourn([]float64{X[index]}, value, l.Bias[0][index],nourn.ActivationFunctions(l.OpertionLayers[0].Activation)))
	}
	input := output
	for index, value := range weight {
		if index != 0 {
			output := []float64{}
			for i, nournValue := range value {
				output = append(output, nourn.Nourn(input, nournValue, l.Bias[index][i],nourn.ActivationFunctions(l.OpertionLayers[index].Activation)))
			}

			input = output
		}
	}
	return output[0]
}

func (l *DenseLayer) densOperrtionBias(X []float64, bias [][]float64) float64 {
	output := []float64{}

	for index, value := range l.Weight[0] {
		output = append(output, nourn.Nourn([]float64{X[index]}, value, bias[0][index],nourn.ActivationFunctions(l.OpertionLayers[0].Activation)))
	}
	input := output
	for index, value := range l.Weight {
		if index != 0 {
			output := []float64{}
			for i, nournValue := range value {
				output = append(output, nourn.Nourn(input, nournValue, bias[index][i],nourn.ActivationFunctions(l.OpertionLayers[index].Activation)))
			}

			input = output
		}
	}
	return output[0]
}

func (l *DenseLayer) Dense(X []float64) []float64 {

	output := []float64{}
	for index, value := range l.Weight[0] {
		output = append(output, nourn.Nourn([]float64{X[index]}, value, l.Bias[0][index],nourn.ActivationFunctions(l.OpertionLayers[0].Activation)))
	}
	input := output
	for index, value := range l.Weight {
		if index != 0 {
			output := []float64{}
			for i, nournValue := range value {
				output = append(output, nourn.Nourn(input, nournValue, l.Bias[index][i],nourn.ActivationFunctions(l.OpertionLayers[index].Activation)))
			}

			input = output
		}
	}
	return input
}

func (l *DenseLayer) Installition(layers []int, input [][]float64, opertion_layears []*OpertionLayears) {
	l.layers = layers
	weight := [][]float64{}
	for i := 0; i < layers[0]; i++ {
		nourn := []float64{rand.Float64()}

		weight = append(weight, nourn)
	}
	l.Weight = append(l.Weight, weight)

	for ind, val := range layers {
		l.OpertionLayers = append(l.OpertionLayers, opertion_layears[ind])
		if ind != 0 {
			weight := [][]float64{}
			for i := 0; i < val; i++ {
				nourn := []float64{}
				for i := 0; i < layers[ind-1]; i++ {
					nourn = append(nourn, rand.Float64())
				}
				weight = append(weight, nourn)

			}
			l.Weight = append(l.Weight, weight)
		}

	}

	for _, val := range layers {
		bias := []float64{}
		for i := 0; i < val; i++ {
			bias = append(bias, rand.Float64()*0.01)
		}
		l.Bias = append(l.Bias, bias)
	}

}
