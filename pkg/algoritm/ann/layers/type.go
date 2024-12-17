package layers

type ActivationFunctions func(float64) float64

type OpertionLayears struct {
	Activation ActivationFunctions
}

type DenseLayer struct {
	Weight          [][][]float64
	Bias            [][]float64
	OpertionLayers []*OpertionLayears
	layers          []int
}

type DenseInterface interface {
	DenseDerivative([]float64, []int) float64
	DensOperrtion([]float64, [][][]float64) float64
	Dense([]float64) []float64
	Installition([]int, [][]float64, []*OpertionLayears)
}

var _ DenseInterface = (*DenseLayer)(nil)

type OptimizerFuncation func(learning_rate float64, model *DenseLayer, y_ture []float64, y_pred []float64, input [][]float64, loss func(args ...[]float64) float64)
