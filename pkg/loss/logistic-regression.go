package loss

import (
	"math"
)

func LogisticRegressionLoss( y_pred []float64,y_true []float64) float64 {
	var output float64
	epsilon := 1e-15 
	for i, value := range y_true {
		output += value*math.Log(math.Max(y_pred[i], epsilon)) + 
			(1-value)*math.Log(math.Max(1-y_pred[i], epsilon))
	}
	return -output / float64(len(y_true))
}

