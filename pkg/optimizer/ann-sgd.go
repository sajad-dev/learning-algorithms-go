package optimizer

import (
	"math/rand"
	"time"

	"github.com/sajad-dev/learning-algorithms-go/pkg/algoritm/ann/layers"
)

func AnnSGD(learning_rate float64, model *layers.DenseLayer, y_ture []float64, y_pred []float64, input [][]float64, loss func(args ...[]float64) float64) {
	h := 5e-3

	rand.Seed(time.Now().UnixMicro())
	rand_num := rand.Intn(len(y_pred))
	y_pred_copy := []float64{}
	y_pred_copy = append(y_pred_copy, y_pred...)
	for index, value := range model.Weight {
		for ind, val := range value {

			for i := range val {
				dzdw := ActivationDerivative(model.Weight[index][ind][i], model.OpertionLayers[index].Activation)

				dpdz := model.DenseDerivative(input[rand_num], []int{index, ind, i})

				y_pred_copy[rand_num] = model.GeneratePred(input[rand_num], []int{index, ind, i})

				dldp := (loss(y_pred_copy, y_ture) - loss(y_pred, y_ture)) / h

				model.Weight[index][ind][i] -= learning_rate * (dpdz * dldp * dzdw)

			}

			y_pred_copy[rand_num] = y_pred[rand_num]

			dzdb := ActivationDerivative(model.Bias[index][ind], model.OpertionLayers[index].Activation)

			dpdz := model.DenseDerivativeBias(input[rand_num], []int{index, ind})

			y_pred_copy[rand_num] = model.GeneratePredB(input[rand_num], []int{index, ind})

			dldp := (loss(y_pred_copy, y_ture) - loss(y_pred, y_ture)) / h

			model.Bias[index][ind] -= learning_rate * (dpdz * dldp * dzdb)

		}
	}

}
