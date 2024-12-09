import json
import mcts
import os
import logging
from datetime import datetime


INPUT_FILENAME = 'blocksworld.jsonl'
OUTPUT_FILENAME = 'blocksworld_sol.json.'
LOG_FOLDER = "logs"

if not os.path.exists(LOG_FOLDER):
    os.makedirs(LOG_FOLDER)

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename = f"{LOG_FOLDER}/log_{timestamp}.log"

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(message)s',
    # Log to file and console
    handlers=[logging.FileHandler(log_filename), logging.StreamHandler()]
)

# Example usage
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    print("Main")
    logger.info("Starting inference ...")
    with open(INPUT_FILENAME, 'r') as file:
        logger.info("Reading dataset...")
        dict_list = [json.loads(line) for line in file]
        solutions, acc = mcts.run_mcts(dict_list, max_iters=20)

    print("Accuracy = ", acc)
    logger.info(f"Accuracy = {acc}")
    with open(OUTPUT_FILENAME, 'w') as json_file:
        json.dump(solutions, json_file, indent=4)
