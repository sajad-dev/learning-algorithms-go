package qlearning

import (
	"fmt"
	"math/rand"
	"time"
)

type QModel struct {
	Environment [][]float64
	StartState  int
	TargetState int
	State       int
	BestAction  []string
}

func (q *QModel) Installtion(x int, y int) {
	environment := [][]float64{}

	for i := 0; i < y; i++ {
		width := []float64{}
		for j := 0; j < x; j++ {
			width = append(width, 0)
		}
		environment = append(environment, width)
	}
	q.Environment = environment
}

func (q *QModel) Policy(state int, epsilon float64) (int, string) {
	rand.Seed(time.Now().UnixMicro())
	actions := q.getActions(state)

	if rand.Float64() < epsilon {
		keys := make([]string, 0, len(actions))
		for k := range actions {
			keys = append(keys, k)
		}
		k := keys[rand.Intn(len(keys))]
		return actions[k], k
	}

	maxQ := -1.0
	best_actions := ""

	for action, state := range actions {
		y, x := q.stateToXY(state)
		if q.Environment[y][x] > maxQ {
			maxQ = q.Environment[y][x]
			best_actions = action
		}
	}
	return actions[best_actions], best_actions
}

func (q *QModel) Reward(state int) float64 {
	height := len(q.Environment)
	width := len(q.Environment[0])

	return float64(state) / float64(height*width-1)
}

func (q *QModel) Optimaize(epsilon float64, alpha float64, gama float64) {

	best_actions, best_actions_string := q.Policy(q.State, epsilon)
	q.BestAction = append(q.BestAction, best_actions_string)

	y_best_actions, x_best_actions := q.stateToXY(best_actions)
	y, x := q.stateToXY(q.State)

	new_value := q.Environment[y_best_actions][x_best_actions]
	current_state_value := q.Environment[y][x]
	q.Environment[y_best_actions][x_best_actions] += alpha * (q.Reward(best_actions) + gama*new_value - current_state_value)

	q.State = best_actions

}

func FitQLearning(alpha float64, gamma float64, epsilon float64, x int, y int, epoche int) ([]string, int, QModel) {
	model := QModel{State: 0, TargetState: 15, StartState: 0}
	model.Installtion(x, y)

	bestQ := []string{}
	bestStep := 10000

	for j := 0; j <= epoche; j++ {
		i := 0
		for true {
			i++

			model.Optimaize(epsilon, alpha, gamma)
			if model.Reward(model.State) == 1 {
				fmt.Printf("Episode %d: Steps %d, Final State %d, Reward %.2f\n", j, i, model.State, model.Reward(model.State))
				if i < bestStep {
					bestStep = i
					bestQ = model.BestAction
				}
				model.State = 0
				model.BestAction = []string{}
				break
			}

		}
	}
	return bestQ, bestStep, model

}
func Preadict(best_actions []string, model QModel) {
	state := model.State
	for _, value := range best_actions {
		actions := model.getActions(state)
		PrintMatris(state, len(model.Environment), len(model.Environment[0]))

		state = actions[value]

	}
	PrintMatris(state, len(model.Environment), len(model.Environment[0]))

	fmt.Println( best_actions)
}

func PrintMatris(state int, x int, y int) {

	for i := 0; i < y; i++ {
		for j := 0; j < x; j++ {
			if state/y == i && state%x == j {
				fmt.Print(1)
			} else {
				fmt.Print(0)
			}
			fmt.Print("  ")
		}
		fmt.Println("")
	}
	fmt.Println("")
}
