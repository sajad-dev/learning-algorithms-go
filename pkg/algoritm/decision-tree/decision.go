package decisiontree

import (
	"fmt"
	"math"
)

type Node struct {
	Branches map[string]*Node
	class    string
	Name     string
}

type DecisionTree struct {
	Root *Node
}

type GinTop struct {
	Name  string
	Gain  float64
	Class string
}

func Entropy(probability []float64) float64 {

	var sum float64

	for _, value := range probability {
		if value > 0 {
			sum += value * (math.Log(value) / math.Log(2))

		}
	}
	return -sum
}

func getNums(input []map[string]string, field string, target string) map[string]map[string]int {
	var parametr = map[string]map[string]int{}
	for _, val := range input {
		tr := val[target]
		for key, value := range val {
			if _, exsis := parametr[value][tr]; exsis && key == field {
				parametr[value][tr]++
			}
			if _, exsis := parametr[value][tr]; !exsis && key == field {

				if _, exsis := parametr[value]; !exsis {
					parametr[value] = map[string]int{}
				}
				
				parametr[value][tr] = 1
			}
		}
	}
	return parametr
}

func getNumsVarible(input []map[string]string, field string, val string) int {
	var count int
	for _, value := range input {
		if value[field] == val {
			count++
		}
	}
	return count
}

func Gain(entropy []float64, probability []float64, entropyS float64) float64 {
	var sum float64

	for index, value := range entropy {
		sum += value * probability[index]
	}

	return entropyS - sum
}

func getTopGin(input []map[string]string, field string) GinTop {
	var fieldTopGin GinTop
	for key, _ := range input[0] {
		if key != field {
			parametr := getNums(input, key, field)
			entr := []float64{}
			pos := []float64{}
			for k, value := range parametr {
				if _, exsis := value["yes"]; exsis {
					entr = append(entr, Entropy([]float64{
						float64(value["yes"]) / float64(value["yes"]+value["no"]),
						float64(value["no"]) / float64(value["yes"]+value["no"])}))
				} else {
					entr = append(entr, 0)

				}
				pos = append(pos, float64(getNumsVarible(input, key, k))/float64(len(input)))
			}
			entS := Entropy([]float64{
				float64(getNumsVarible(input, field, "yes")) / float64(len(input)),
				float64(getNumsVarible(input, field, "no")) / float64(len(input))})

			gain := Gain(entr, pos, entS)
			if fieldTopGin.Gain < gain {
				fieldTopGin.Gain = gain
				fieldTopGin.Name = key

			}
			if gain == 0 {
				fieldTopGin.Class = input[0][field]
			}

		}
	}
	return fieldTopGin
}

func removeParametrMap(input []map[string]string, field string) []map[string]string {
	output := []map[string]string{}

	for _, value := range input {
		hashmap := map[string]string{}
		for key, val := range value {
			if key != field {
				hashmap[key] = val
			}
		}
		output = append(output, hashmap)

	}
	return output
}

func filterData(input []map[string]string, field string, parametr string) ([]map[string]string, []map[string]string) {
	output := []map[string]string{}
	removed := []map[string]string{}
	for _, value := range input {
		if value[field] == parametr {

			output = append(output, value)
		} else {
			removed = append(removed, value)
		}
	}
	return removeParametrMap(output, field), removed

}

func (d *DecisionTree) writeTree(input []map[string]string, field string, removed *[]map[string]string) *Node {
	gain := getTopGin(input, field)
	node := Node{Branches: map[string]*Node{}, Name: gain.Name}
	getEdge := getNums(input, node.Name, field)

	if gain.Gain == 0 {
		node.class = input[0][field]

		return &node
	}

	for key, _ := range getEdge {

		if gain.Gain != 0 {
			filter, removed := filterData(input, gain.Name, key)

			node.Branches[key] = d.writeTree(filter, field, &removed)

		}
	}

	return &node
}

func (d *DecisionTree) DecisionTreeFit(input []map[string]string, field string) {
	d.Root = d.writeTree(input, field, nil)
	// printTree(d.Root, 0)
}

func (d *DecisionTree) Predict(input map[string]string) {
	node := d.Root
	for {
		if node == nil || node.class != "" {
			break
		}
		value := input[node.Name]

		node = node.Branches[value]
	}
	fmt.Println(node.class)
}

func printTree(node *Node, level int) {
	if node == nil {
		return
	}

	prefix := ""
	for i := 0; i < level; i++ {
		prefix += "  "
	}
	fmt.Printf("%sNode: %s, Class: %s\n", prefix, node.Name, node.class)
	for key, branch := range node.Branches {
		fmt.Printf("%sBranch: %s\n", prefix, key)
		printTree(branch, level+1)
	}
}
