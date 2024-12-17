package svm

import "math/rand"

type SvmV1 struct {
	Weight []float64
	Bias   float64
}

func (s *SvmV1) Installition(num int) {
	for i := 0; i < num; i++ {
		s.Weight = append(s.Weight, rand.Float64())
	}
	s.Bias = rand.Float64()
}

func (s *SvmV1) OpertionWeight(input []float64, weight []float64,bias float64) float64 {
	var sum float64
	for index, value := range input {
		sum += value * weight[index]
	}
	sum += bias

	return sum

}

func (s *SvmV1) Opertion(input []float64) float64 {
	var sum float64
	for index, value := range input {
		sum += value * s.Weight[index]
	}
	sum += s.Bias

	switch {
	case sum >= 0:
		return 1.0
	default:
		return -1.0
	}

}


