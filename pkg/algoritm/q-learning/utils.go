package qlearning

func (q *QModel) getState(x int, y int) int {
	return x + (y * len(q.Environment))
}

func (q *QModel) getActions(state int) map[string]int {
	actions := map[string]int{}

	height := len(q.Environment)
	width := len(q.Environment[0])
	if state%width != width-1 {
		actions["right"] = state + 1
	}
	if state%width != 0 {
		actions["left"] = state - 1
	}
	if state/width != height-1 {
		actions["botton"] = state + width
	}
	if state/width != 0 {
		actions["top"] = state - width
	}
	return actions
}

func (q *QModel) stateToXY(state int) (int, int) {
	width := len(q.Environment[0])
	return state / width, state % width
}
