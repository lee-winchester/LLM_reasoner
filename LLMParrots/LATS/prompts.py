import re
import json
import model
import logging

# Get the logger
logger = logging.getLogger(__name__)

expert_solving_blocksworld = "You are an expert in solving blocks world problems."

actions_prefix = "I am playing with a set of blocks where I need to arrange the blocks into stacks. Here are the actions I can do:"

allowed_actions = "Pick up a block.\nUnstack a block from on top of another block.\nPut down a block.\nStack a block on top of another block."

action_restrictions_prefix = "I have the following restrictions on my actions:"

action_restrictions = "I can only pick up or unstack one block at a time.\nI can only pick up or unstack a block if my hand is empty.\nI can only pick up a block if the block is on the table and the block is clear. A block is clear if the block has no other blocks on top of it and if the block is not picked up.\nI can only unstack a block from on top of another block if the block I am unstacking was really on top of the other block.\nI can only unstack a block from on top of another block if the block I am unstacking is clear.\nOnce I pick up or unstack a block, I am holding the block.\nI can only put down a block that I am holding.\nI can only stack a block on top of another block if I am holding the block being stacked.\nI can only stack a block on top of another block if the block onto which I am stacking the block is clear.\nOnce I put down or stack a block, my hand becomes empty.\nOnce you stack a block on top of a second block, the second block is no longer clear."

states_prefix = "The states of my blocks are represented as follows:"

allowed_states = "'Clear' is a set of blocks which are clear and free to move.\n'Hand' has the value of the block in hand. If no block in hand then it has value None.\n'Blocks on top of locations' is a dictionary of blocks in key value pairs, such that key is the block and value is the location on which the block is currently. location can be a another block or the table"

current_state_prefix = "We have the current state as follows:"

goal_state_prefix = "Goal state is as follows:"

next_action_prefix = "Next action to take:"

next_action_requirements = "Now, give me all actions possible on the current state. Actions should be one of the actions mentioned above and also should adhere to the restrictions mentioned above. I just need the actions and I don't need any explanations."

next_action_expected_format = "Give your actions as a comma separated string with '<NEXT_ACTIONS>' keyword before and '</NEXT_ACTIONS>' keyword after the actions. Every action should be in the format [ACTION][BLOCK]."

next_action_example_answer = "An example answer : <NEXT_ACTIONS>Pick up block blue, Unstack block red, Stack block yellow on block orange, Put down green block</NEXT_ACTIONS>."

next_state_requirements = "Based on the current state, the action given and the definition of the actions given above, create the new state in the same format as the current state. I need the 'Clear blocks', 'Hand' and 'Block on top of locations' for the new state we reach. Do no give me any explanations. I just need the new state."

next_state_expected_format = "Give the next state as string with '<NEXT_STATE>' keyword before and '</NEXT_STATE>' keyword after the next state. The next state should be in the format {'Clear blocks': [CLEAR_BLOCKS], 'Hand': [HAND], 'Blocks on top of locations': [BLOCKS_ON_TOP_OF_LOCATIONS]}. Ensure it is a valid JSON string"

next_state_example_answer = 'An example answer : <NEXT_STATE>{"Clear blocks": ["green", "yellow"], "Hand": "blue", "Blocks on top of locations": {"orange": "red", "red": "white", "white": "pink"}</NEXT_STATE>.'

back_prop_requirements = "As per the given possible actions and considering their restrictions, tell me how many such actions will be required to reach from current state to goal state. Do no give me any explainations."

back_prop_expected_format = "Give me strictly numeric answer with '<NUM_STEPS>' keyword before and '</NUM_STEPS>' keyword after the answer."

back_prop_example_answer = "An example answer : <NUM_STEPS>4</NUM_STEPS>"


def getActions(prompt):
    attempts = 3
    next_actions = None
    while attempts > 0:
        try:
            response = model.prompt_model(prompt)
            next_actions = re.findall(
                r"<NEXT_ACTIONS>(.*?)</NEXT_ACTIONS>", response)
            next_actions = [x for x in next_actions if len(x) > 0][0]
            break
        except:
            logger.warning("Parsing FAILED, trying again")
            attempts -= 1
    if next_actions:
        list_of_next_actions = [action.strip()
                                for action in next_actions.split(",")]
        return list_of_next_actions
    else:
        raise Exception(
            f"Cannot parse this response for next action {response}")


def getNextState(prompt):
    attempts = 3
    next_state = None
    while attempts > 0:
        try:
            response = model.prompt_model(prompt)
            next_state = re.findall(
                r"<NEXT_STATE>(.*?)</NEXT_STATE>", response)
            next_state = [x for x in next_state if len(x) > 0][0]
            break
        except:
            logger.warning("Parsing FAILED, trying again")
            attempts -= 1
    if next_state:
        clear_blocks = re.findall(r"\[(.*?)\]", next_state)[0]
        hand = re.findall(r'"Hand":(.*?),', next_state)[0].strip()
        blocks_on_top_of_locations = re.findall(r": {(.*?)}", next_state)[0]
        print(clear_blocks, hand, blocks_on_top_of_locations)
        return json.loads(
            f'{{"Clear blocks": [{clear_blocks}], "Hand": {
                hand}, "Blocks on top of locations": {{{blocks_on_top_of_locations}}}}}'
        )
    else:
        raise Exception(
            f"Cannot parse this response for next state {response}")


def getGoodness(prompt):
    attempts = 3
    num_steps = None
    while attempts > 0:
        try:
            response = model.prompt_model(prompt)
            num_steps = re.findall(r"<NUM_STEPS>(.*?)</NUM_STEPS>", response)
            num_steps = [x for x in num_steps if len(x) > 0][0]
            break
        except:
            logger.warning("Parsing FAILED, trying again")
            attempts -= 1
    if num_steps:
        return int(num_steps)
    else:
        raise Exception(f"Cannot parse this response for goodness {response}")


def get_next_action_prompt(current_state_status):
    return f"{expert_solving_blocksworld}\n\n{actions_prefix}\n\n{allowed_actions}\n\n{action_restrictions_prefix}\n{action_restrictions}\n\n{states_prefix}\n{allowed_states}\n\n{current_state_prefix}\n{current_state_status}\n\n{next_action_requirements}\n\n{next_action_expected_format}\n{next_action_example_answer}"


def get_next_state_prompt(current_state_status, action):
    return f"{expert_solving_blocksworld}\n\n{actions_prefix}\n\n{allowed_actions}\n\n{action_restrictions_prefix}\n{action_restrictions}\n\n{states_prefix}\n{allowed_states}\n\n{current_state_prefix}\n{current_state_status}\n\n{next_action_prefix}\n{action}\n\n{next_state_requirements}\n\n{next_state_expected_format}\n{next_state_example_answer}"


def get_back_prop_prompt(current_state_status, goal_state_status):
    return f"{expert_solving_blocksworld}\n\n{actions_prefix}\n\n{allowed_actions}\n\n{action_restrictions_prefix}\n{action_restrictions}\n\n{states_prefix}\n{allowed_states}\n\n{current_state_prefix}\n{current_state_status}\n\n{goal_state_prefix}\n{goal_state_status}\n\n{back_prop_requirements}\n\n{back_prop_expected_format}\n{back_prop_example_answer}"
