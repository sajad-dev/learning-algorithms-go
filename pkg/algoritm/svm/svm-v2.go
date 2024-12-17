package svm

import mathoprtion "github.com/sajad-dev/learning-algorithms-go/pkg/math-oprtion"

type SvmV2 struct {
	Alpha []float64
	Bias  float64
}

func (s *SvmV2) InstallitionV2(num int) {

	for i := 0; i < num; i++ {
		s.Alpha = append(s.Alpha, 0.0)
	}

	s.Bias = 0
}

func (model *SvmV2) OpertionV2(X_new []float64, X [][]float64, y []float64, sigma float64) float64 {
	var result float64
	// Compute the sum of alpha_i * y_i * K(x_i, x_new)
	for i := range model.Alpha {
		result += model.Alpha[i] * y[i] * mathoprtion.Gaussian(X_new, X[i], sigma)
	}
	// Add the bias term
	result += model.Bias

	// If the result is greater than 0, classify as +1, else -1
	if result >= 0 {
		return 1.0
	} else {
		return -1.0
	}
}
