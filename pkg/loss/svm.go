package loss

import (
	"math"

	"github.com/sajad-dev/learning-algorithms-go/pkg/algoritm/svm"
	mathoprtion "github.com/sajad-dev/learning-algorithms-go/pkg/math-oprtion"
)

func SVMLoss(y_true float64, y_pred float64, w []float64) float64 {
	return math.Max(0.0, 1-y_true*y_pred) + Regularization(w, 0.1)
}

func Regularization(w []float64, lambda float64) float64 {
	var sum float64
	for _, wi := range w {
		sum += wi * wi
	}
	return (lambda / 2) * sum
}

func SVMLossGaussian(model *svm.SvmV2, alpha []float64, X [][]float64, y []float64, C float64, sigma float64, bias float64) float64 {
	var loss float64
	n := len(X)

	for i := range alpha {
		loss += alpha[i]
	}
	loss = 0.5 * loss

	for i := 0; i < n; i++ {
		var sum float64
		for j := 0; j < n; j++ {
			sum += alpha[j] * y[j] * mathoprtion.Gaussian(X[i], X[j], sigma)
		}
		margin := 1 - y[i]*(sum+bias)
		if margin > 0 {
			loss += C * margin
		}
	}

	return loss
}
