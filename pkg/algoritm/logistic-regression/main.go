package logisticregression

import (
	"math"
	"math/rand"
)

type LogisticRegression struct {
	Weight []float64
	Bias   float64
}

func (l *LogisticRegression) Installition(input []float64) {
	for i := 0; i < len(input); i++ {
		l.Weight = append(l.Weight, rand.Float64())
	}
}

func (l *LogisticRegression) Oprtion(input []float64) float64 {
	var output float64

	for index, value := range input {
		output += value * l.Weight[index]
	}
	output += l.Bias
	output = 1 / (1 + math.Exp(-output))
	if output > 1 {
		output = 1
	}

	return output

}

func (l *LogisticRegression) OprtionDerivative(input []float64, weight []float64, bias float64) float64 {
	var output float64

	for index, value := range input {
		output += value * weight[index]
	}

	output += l.Bias
	output = 1 / (1 + math.Exp(-output))
	if output > 1 {
		output = 1
	}
	return output

}
