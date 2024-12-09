import re


class BlocksWorldGoalState:

    def __init__(self):
        self.current_state_status = dict()
        self.blocks_on_top_of_locations = dict()

    def add_block_positioning(self, block, location):
        self.blocks_on_top_of_locations[block] = location

    def get_goal_state_status(self):
        self.current_state_status["Blocks on top of locations"] = self.blocks_on_top_of_locations
        return self.current_state_status


class BlocksWorldState:

    def __init__(self):
        self.current_state_status = dict()
        self.clear_blocks = list()
        self.hand = None
        self.blocks_on_top_of_locations = dict()

    def add_block_to_clear(self, block):
        self.clear_blocks.append(block)

    def set_hand(self, block=None):
        self.hand = block

    def add_block_positioning(self, block, location):
        self.blocks_on_top_of_locations[block] = location

    def get_current_state_status(self):
        self.current_state_status["Clear blocks"] = self.clear_blocks
        self.current_state_status["Hand"] = self.hand
        self.current_state_status["Blocks on top of locations"] = self.blocks_on_top_of_locations
        return self.current_state_status


def get_initial_state(prompt):

    current_state = BlocksWorldState()

    clear_blocks = re.findall(r'the (\w+) block is clear', prompt)

    for clear_block in clear_blocks:
        current_state.add_block_to_clear(clear_block)

    if "the hand is empty" in prompt:
        current_state.set_hand(None)
    else:
        block_in_hand = re.search(r'the hand is holding the (\w+) block', prompt).group(1)
        current_state.set_hand(block_in_hand)

    positionings = re.findall(r'the (\w+) block is (on top of the|on the) (\w+)', prompt)

    for block, _, position in positionings:
        current_state.add_block_positioning(block, position)

    current_state_status = current_state.get_current_state_status()

    return current_state_status


def get_goal_state(prompt):

    goal_state = BlocksWorldGoalState()

    positionings = re.findall(r'the (\w+) block is (on top of the|on the) (\w+)', prompt)

    for block, _, position in positionings:
        goal_state.add_block_positioning(block, position)

    goal_state_status = goal_state.get_goal_state_status()

    return goal_state_status


def parsePrompt(original_prompt):

    # Extracting the initial conditions (between [STATEMENT] and "My goal is")
    initial_conditions_pattern = r"\[STATEMENT\](.*?)My goal is"
    initial_conditions_match = re.search(initial_conditions_pattern, original_prompt, re.DOTALL)

    if initial_conditions_match:
        initial_state_prompt = initial_conditions_match.group(1).strip()
    else:
        initial_state_prompt = None

    # Extracting the goal state (between "My goal is" and "My plan is")
    goal_state_pattern = r"My goal is(.*?)My plan is"
    goal_state_match = re.search(goal_state_pattern, original_prompt, re.DOTALL)

    if goal_state_match:
        goal_state_prompt = goal_state_match.group(1).strip()
    else:
        goal_state_prompt = None
    # print(initial_state_prompt)
    initial_state_status = get_initial_state(initial_state_prompt)
    goal_state_status = get_goal_state(goal_state_prompt)

    return initial_state_status, goal_state_status


def check_goal_state_satisfied(current_state_status, goal_state_status):
    for block, location in goal_state_status["Blocks on top of locations"].items():
        if block in current_state_status["Blocks on top of locations"].keys():
            if not current_state_status["Blocks on top of locations"][block] == location:
                return False
        else:
            return False
    return True


if __name__ == "__main__":

    original_prompt = "I am playing with a set of blocks where I need to arrange the blocks into stacks. Here are the actions I can do\n\nPick up a block\nUnstack a block from on top of another block\nPut down a block\nStack a block on top of another block\n\nI have the following restrictions on my actions:\nI can only pick up or unstack one block at a time.\nI can only pick up or unstack a block if my hand is empty.\nI can only pick up a block if the block is on the table and the block is clear. A block is clear if the block has no other blocks on top of it and if the block is not picked up.\nI can only unstack a block from on top of another block if the block I am unstacking was really on top of the other block.\nI can only unstack a block from on top of another block if the block I am unstacking is clear.\nOnce I pick up or unstack a block, I am holding the block.\nI can only put down a block that I am holding.\nI can only stack a block on top of another block if I am holding the block being stacked.\nI can only stack a block on top of another block if the block onto which I am stacking the block is clear.\nOnce I put down or stack a block, my hand becomes empty.\nOnce you stack a block on top of a second block, the second block is no longer clear.\n\n[STATEMENT]\nAs initial conditions I have that, the red block is clear, the blue block is clear, the orange block is clear, the hand is empty, the blue block is on top of the yellow block, the red block is on the table, the orange block is on the table and the yellow block is on the table.\nMy goal is to have that the blue block is on top of the orange block and the yellow block is on top of the red block.\n\nMy plan is as follows:\n\n[PLAN]"

    initial_state_status, goal_state_status = parsePrompt(original_prompt)

    print("Initial State:")
    print(initial_state_status)
    print()
    print("Goal State:")
    print(goal_state_status)

    # is_goal_state_satisfied = check_goal_state_satisfied(initial_state_status, goal_state_status)
    # print(is_goal_state_satisfied)
