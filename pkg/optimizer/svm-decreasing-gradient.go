package optimizer

import (
	"github.com/sajad-dev/learning-algorithms-go/pkg/algoritm/svm"
	"github.com/sajad-dev/learning-algorithms-go/pkg/loss"
)

func SvmGradianDecreasing(learning_rate float64, input [][]float64, y_true []float64, model *svm.SvmV1, loss func(float64, float64, []float64) float64) {
	h := 5e-3
	for ind, val := range input {
		for index, _ := range model.Weight {
			var weight_copy []float64
			weight_copy = append(weight_copy, model.Weight...)
			weight_copy[index] += h
			y_pred := model.OpertionWeight(val, model.Weight, model.Bias)
			y_pred_2 := model.OpertionWeight(val, weight_copy, model.Bias)

			dydw := (y_pred_2 - y_pred) / h
			dldy := (loss(y_pred+h, y_true[ind], model.Weight) - loss(y_pred, y_true[ind], model.Weight)) / h
			model.Weight[index] -= learning_rate * (dydw * dldy)

			y_pred = model.OpertionWeight(val, model.Weight, model.Bias)
			y_pred_2 = model.OpertionWeight(val, model.Weight, model.Bias+h)

			dydw = (y_pred_2 - y_pred) / h
			dldy = (loss(y_pred+h, y_true[ind], model.Weight) - loss(y_pred, y_true[ind], model.Weight)) / h
			model.Bias -= learning_rate * (dydw * dldy)

		}
	}
}

func GaussianSvmGradianDecreasing(model *svm.SvmV2, X [][]float64, y []float64, learningRate, C, sigma float64) {
	h := 1e-5
	for i := range model.Alpha {

		originalAlpha := model.Alpha[i]
		model.Alpha[i] += h
		lossPlus := loss.SVMLossGaussian(model, model.Alpha, X, y, C, sigma, model.Bias)

		model.Alpha[i] = originalAlpha - h
		lossMinus := loss.SVMLossGaussian(model, model.Alpha, X, y, C, sigma, model.Bias)

		gradAlpha := (lossPlus - lossMinus) / (2 * h)

		model.Alpha[i] -= learningRate * gradAlpha
	}

}
