from typing import List
import math
from parse_prompt import *
from prompts import *
import logging

# Get the logger
logger = logging.getLogger(__name__)


class Node:
    def __init__(self, state_id, state, trajectory=[], parent=None):
        self.state_id = state_id,
        self.trajectory = trajectory
        self.parent = parent
        self.children = []
        self.value = 0
        self.visits = 0
        self.state = state

    def uct(self, exploration_weight=1.0):
        if self.visits == 0:
            # return float('inf')
            return self.value
        return (self.value / self.visits) + exploration_weight * math.sqrt(math.log(self.parent.visits) / self.visits)

    def best_child(self):
        if not self.children:  # Check if children list is empty
            return None
        return max(self.children, key=lambda child: child.uct())

    def best_child_value(self):
        if not self.children:  # Check if children list is empty
            return None
        return max(self.children, key=lambda child: child.value)

    def update(self, reward: float):
        self.visits += 1
        self.value += reward

    def set_trajectory(self, action: str):
        self.trajectory.append(action)

    def __str__(self):
        return f"State id: {self.state_id}, Parent: {self.parent.state_id if self.parent else None}, Children: {self.children}, Trajectory: {self.trajectory}, value: {self.value}, vistis: {self.visits}, state: {self.state}"


def compare_with_truth(solution, truth_value):
    if solution == truth_value.replace("\n", "").replace("(", "").replace(")", ",")[:-1]:
        return True
    else:
        return False


def run_mcts(
    dataset: List[dict],
    max_iters: int
):
    num_items = len(dataset)
    num_success = 0  # Counter for successful solutions
    # cur_func_impl = None
    updated_items = []
    for i, item in enumerate(dataset):
        if i == 25:
            break
        state_id = 0
        solved_counter = 0
        solved = False
        logger.info(f"Processing item {i}")
        # Initialize root node
        initial_state, goal_state = parsePrompt(item['query'])
        # initial solution (for pass@1 metric)
        root = Node(state_id=state_id, state=initial_state)
        logger.info("Root:")
        logger.info(str(root))
        # Create depth 1 children
        logger.info("Creating initial children nodes")
        next_action_prompt = get_next_action_prompt(
            root.state, prev_action="No action")
        next_actions = getActions(next_action_prompt)
        for action in next_actions:
            state_id += 1
            next_state_prompt = get_next_state_prompt(initial_state, action)
            child_state = getNextState(next_state_prompt)
            child_node = Node(state_id=state_id,
                              state=child_state, parent=root)
            child_node.trajectory = list(root.trajectory)
            child_node.set_trajectory(action)
            # check if child node = goal state
            if check_goal_state_satisfied(child_state, goal_state):
                solved = True
                break
            logger.info("New node:")
            logger.info(str(child_node))
            root.children.append(child_node)
        if solved:
            solution_plan = ','.join(child_node.trajectory)
            item['solution_plan'] = solution_plan
            logger.info(f"Solution reached : {solution_plan}")
            continue
        for iter in range(max_iters):
            logger.info(f"Iteration {iter}")
            # Selection
            node = root
            while node.children:
                node = node.best_child()

            logger.info("Selection:")
            logger.info(str(node))

            if node.visits > 0:

                # Expansion
                logger.info("Expansion:")
                next_action_prompt = get_next_action_prompt(
                    node.state, node.trajectory[-1])

                next_actions = getActions(next_action_prompt)

                for action in next_actions:
                    state_id += 1
                    next_state_prompt = get_next_state_prompt(
                        initial_state, action)
                    child_state = getNextState(next_state_prompt)
                    child_node = Node(state_id=state_id,
                                      state=child_state, parent=node)
                    child_node.trajectory = list(node.trajectory)
                    child_node.set_trajectory(action)
                    node.children.append(child_node)
                    if check_goal_state_satisfied(child_state, goal_state):
                        solved = True
                        break
                    logger.info("New node:")
                    logger.info(str(child_node))
                if solved:
                    solution_plan = ','.join(child_node.trajectory)
                    item['solution_plan'] = solution_plan
                    logger.info(f"Solution reached : {solution_plan}")
                    break
            else:

                # Simulation
                logger.info("Simulation:")
                logger.info(str(node))
                reward_prompt = get_back_prop_prompt(node.state, goal_state)
                reward = getGoodness(reward_prompt)
                reward = (100 - reward)/1000
                logger.info(f"Reward: {reward}")
                # if reward = 1, goal state reached
                if reward == 1:
                    solved = True
                    solved_counter += 1
                    break
                node.update(reward)
                logger.info("Updated node:")
                logger.info(str(node))
                # backpropagation

                while node.parent:
                    parent = node.parent
                    parent.update(reward)
                    node = parent
                logger.info("values backpropagated")

        if solved:
            solution_plan = ','.join(node.trajectory)
            item['solution_plan'] = solution_plan
            logger.info(f"Solution reached : {solution_plan}")
        else:
            solution_plan = ','.join(root.best_child().trajectory)
            item['solution_plan'] = solution_plan
            logger.info(f"Interation exhausted, best solution : {
                        solution_plan}")
        updated_items.append(item)
        if compare_with_truth(item['solution_plan'], item['ground_truth']):
            num_success += 1

    return updated_items, num_success/num_items
