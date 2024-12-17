package trainigloop

import (
	"fmt"

	"github.com/sajad-dev/learning-algorithms-go/pkg/algoritm/ann/layers"
	logisticregression "github.com/sajad-dev/learning-algorithms-go/pkg/algoritm/logistic-regression"
	"github.com/sajad-dev/learning-algorithms-go/pkg/algoritm/svm"
	"github.com/sajad-dev/learning-algorithms-go/pkg/loss"
	"github.com/sajad-dev/learning-algorithms-go/pkg/optimizer"
)

func FitAnn(input [][]float64, output []float64, learning_rate float64, layers_list []int, optimizer layers.OptimizerFuncation, activation []*layers.OpertionLayears, epoch int) *layers.DenseLayer {
	l := layers.DenseLayer{}
	l.Installition(layers_list, input, activation)
	for i := 0; i <= epoch; i++ {
		y_pred := []float64{}
		for _, value := range input {
			y_pred = append(y_pred, l.Dense(value)[0])
		}
		optimizer(learning_rate, &l, output, y_pred, input, loss.MeenAbsoluteError)

		fmt.Println(loss.MeenSquaredError(y_pred, output))
		if loss.MeenSquaredError(y_pred, output) < 0.005 {
			fmt.Println(i)
			break
		}
	}
	return &l

}

func FitSvmV1(input [][]float64, output []float64, epoch int) *svm.SvmV1 {
	model := svm.SvmV1{}
	model.Installition(2)

	for i := 0; i <= epoch; i++ {
		for _, val := range input {
			model.Opertion(val)
			optimizer.SvmGradianDecreasing(0.001, input, output, &model, loss.SVMLoss)
		}
	}
	return &model
}

func FitSvmV2(input [][]float64, output []float64, epoch int, C float64, sigma float64, learning_rate float64) *svm.SvmV2 {
	model := svm.SvmV2{}
	model.InstallitionV2(len(input))

	for i := 0; i <= epoch; i++ {
		optimizer.GaussianSvmGradianDecreasing(&model, input, output, learning_rate, C, sigma)
	}
	return &model
}

func FitLogisticRegression(input [][]float64, output []float64, epoch int, learning_rate float64) *logisticregression.LogisticRegression {
	model := logisticregression.LogisticRegression{}
	model.Installition(input[0])

	for i := 0; i <= epoch; i++ {
		y_pred := []float64{}
		for _, value := range input {
			y_pred = append(y_pred, model.Oprtion(value))
		}

		optimizer.LogisticRegressionGradianDecreasing(0.01, &model, output, y_pred, input, loss.LogisticRegressionLoss)
	}
	return &model
}
