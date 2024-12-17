package optimizer

import (
	logisticregression "github.com/sajad-dev/learning-algorithms-go/pkg/algoritm/logistic-regression"
)

func LogisticRegressionGradianDecreasing(learning_rate float64, model *logisticregression.LogisticRegression, y_ture []float64, y_pred []float64, input [][]float64, loss func([]float64, []float64) float64) {

	h := 5e-5
	for j, _ := range y_pred {
		y_pred_copy := []float64{}
		y_pred_copy = append(y_pred_copy, y_pred...)
		y_pred_copy[j] += h

		for index, _ := range model.Weight {

			var weg []float64
			weg = append(weg, model.Weight...)
			weg[index] += h
			dpdw := (model.OprtionDerivative(input[j], weg, model.Bias) - model.OprtionDerivative(input[j], model.Weight, model.Bias)) / h

			dldp := (loss(y_pred_copy, y_ture) - loss(y_pred, y_ture)) / h
			model.Weight[index] -= learning_rate * (dpdw * dldp)
		}
		var bias float64
		bias = model.Bias
		bias += h
		dpdw := (model.OprtionDerivative(input[j], model.Weight, bias) - model.OprtionDerivative(input[j], model.Weight, model.Bias)) / h

		dldp := (loss(y_pred_copy, y_ture) - loss(y_pred, y_ture)) / h

		model.Bias -= learning_rate * (dpdw * dldp)
	}
}
