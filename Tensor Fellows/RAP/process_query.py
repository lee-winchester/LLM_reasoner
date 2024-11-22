import re
import json

def load_json(path):
    with open(path, 'r') as F:
        queries = json.load(F)
        
    return queries

def process_prompt(query):
    # Regex patterns
    init_pattern = (
        r"As initial conditions I have that, (.+?)\.\s*My goal is to have that"
    )
    goal_pattern = r"My goal is to have that (.+?)\.\s*My plan is as follows:"
    question_pattern = (
        r"As initial conditions I have that,.*?My plan is as follows:\n\n\[PLAN\]"
    )

    # Extract using regex
    init_match = re.search(init_pattern, query, re.DOTALL)
    goal_match = re.search(goal_pattern, query, re.DOTALL)
    question_match = re.search(question_pattern, query, re.DOTALL)

    # Validate matches
    init = init_match.group(1).strip()  # if init_match else "No match for init"
    goal = goal_match.group(1).strip()  # if goal_match else "No match for goal"
    question =  '\n[STATEMENT]\n' + question_match.group(
        0
    ).strip()  # if question_match else "No match for question"

    # Construct the dictionary
    example_temp = {"init": init, "goal": goal, "question": question}

    return example_temp


def process_query_data(query_dict):
    prompt = process_prompt(query_dict['query'])
    ground_truth_plan = '(unstack yellow orange)\n(put-down yellow)\n(pick-up orange)\n(stack orange red)\n'

    # Convert the string into a list by splitting by '\n', removing parentheses and empty strings
    ground_truth_plan = query_dict['ground_truth']
    ground_truth_action_list = [
        action.strip("()") for action in ground_truth_plan.strip().split("\n") if action
    ]
    
    return prompt, ground_truth_action_list


