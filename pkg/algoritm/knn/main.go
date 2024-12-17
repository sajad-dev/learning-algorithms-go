package knn

import mathoprtion "github.com/sajad-dev/learning-algorithms-go/pkg/math-oprtion"

func Knn(input [][]float64, lable []float64, parametr []float64) float64 {
	output := map[string]float64{"class": lable[0], "d": mathoprtion.Oghlidos(input[0], parametr)}

	for index, value := range input {
		og := mathoprtion.Oghlidos(value, parametr)
		if og < output["d"] {
			output = map[string]float64{"class": lable[index], "d": og}
		}
	}

	return output["class"]

}
