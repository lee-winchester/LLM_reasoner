# LLM-Reasoners Demo

This repository provides a demonstration of using Large Language Models (LLMs) to solve the Blocks World problem using Reasoning-as-Planning (RAP). The project includes steps to preprocess model outputs, calculate exact match accuracy, and evaluate the performance of LLMs like LLama and Gemini using RAP in a Blocks World setting.

## Overview

The goal is to test LLMs on the Blocks World dataset by generating plans to achieve specified configurations and comparing these plans with ground-truth actions. The workflow includes:
- **Model Integration:** Using pre-trained LLMs to generate block configuration transitions.
- **Action Planning:** Employing RAP frameworks to plan actions to reach a goal configuration.
- **Evaluation Metrics:** Calculating exact match accuracy after preprocessing the model outputs.

## Key Features

- **Blocks World RAP Implementation:** A reasoning framework that uses block states instead of action histories to determine the next state transition.
- **Preprocessing for Exact Match:** Removal of irrelevant words (e.g., "the," "block," "from") to semantically align generated and ground-truth actions.
- **Results Evaluation:** Automated pipeline to compute exact match accuracy for model-generated plans.

## Workflow

### 1. Setup

The RAP implementation relies on the `reasoners` library and a compatible pre-trained LLM (e.g., LLama-2). A pre-configured Blocks World environment is loaded using the provided scripts.

### 2. Model Integration

The pre-trained Llama model is integrated using the `ExLlamaModel` interface. Model outputs are generated for each test query, and RAP is used to determine the next actions based on the current block state.

### 3. Execution

For each query in the dataset:
1. **Prompt Processing:** Queries are parsed into initial block configurations and goals using the `process_query_data` function.
2. **Action Generation:** The RAP framework generates actions to transition between states using the LLM's output.
3. **Trace Logging:** Model-generated actions (traces) and the ground-truth actions are stored in a `.jsonl` file.

### 4. Evaluation

The evaluation involves:
1. **Preprocessing Outputs:** Removing unnecessary words (e.g., "the," "block") using regular expressions to ensure semantic alignment with ground truth.
2. **Exact Match Check:** Comparing the preprocessed traces with ground truth on a step-by-step basis:
    - **Mismatch in Length:** If the trace length differs from the ground truth, the instance is marked as incorrect.
    - **Mismatch in Steps:** If any step in the trace differs from the ground truth after cleaning, the instance is marked as incorrect.
3. **Accuracy Calculation:** The ratio of correctly matched instances to the total number of queries determines the exact match accuracy.

### 5. Results

The pipeline outputs the accuracy of the RAP framework with the tested LLM. The final accuracy is calculated as:

```python
accuracy = sum(exact_matches) / len(exact_matches)
accuracy_percentage = accuracy * 100
```

## Code Files

1. **`RAP_llama.ipynb/RAP_gemini.ipynb`:** Contains the full demonstration, including model setup, RAP implementation, and evaluation.
2. **`rap-llama-results.jsonl`:** Stores the model-generated traces and ground-truth actions for each query.
3. **`blocksworld.jsonl`:** Input dataset containing Blocks World queries.
4. **`process_query.py`:** Utilities for query parsing and preprocessing.

## How Results Are Calculated

- **Generated Trace:** The RAP framework generates a sequence of actions (trace) for each query.
- **Ground-Truth Comparison:** The trace is compared with the ground-truth actions after cleaning to remove stopwords.
- **Exact Match Criteria:** A query is considered a match if:
  - The length of the generated trace matches the ground truth.
  - Each action in the trace matches the corresponding ground-truth action after cleaning.
- **Accuracy:** The proportion of queries with exact matches is calculated as the exact match accuracy.

## Sample Output

An example of a result entry in `rap-llama-results.jsonl`:

```json
{
  "trace": ["pick-up yellow block", "stack yellow block on top of blue block"],
  "ground_truth": ["pick-up yellow", "stack yellow blue"]
}
```

Accuracy calculation:

- If the trace matches the ground truth for all steps, the instance is marked as correct.
- Final accuracy is the percentage of correct instances in the dataset.
