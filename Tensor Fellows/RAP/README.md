Here’s a sample **`README.md`** for your project, summarizing the key details and guiding potential users:

```markdown
# LLM-Reasoners Demo

## Setup

This project is implemented in Python and leverages a unified interface for interacting with Large Language Models (LLMs) using **ExLlamaModel** or other supported providers.

### Prerequisites

1. Python 3.8 or higher
2. CUDA-enabled GPU
3. Required Python libraries (install with `pip install -r requirements.txt`)

### Initialize the Model

By default, the notebook uses **ExLlamaModel**. You can set up other model providers, such as HuggingFace or OpenAI models, as detailed in the code.

```python
model = ExLlamaModel(
    model_dir="TheBloke/Llama-2-7b-Chat-GPTQ",
    device=torch.device("cuda:0"),
    max_batch_size=1,
    max_new_tokens=200,
    max_seq_length=2048,
)
```

---

## Dataset

The Blocksworld dataset is utilized for demonstration purposes, loaded via `blocksworld.jsonl`. It contains block manipulation examples used to create prompts for in-context learning. This is a preprocessed version of task_1_plan_generation.json

```python
queries = load_json('./blocksworld.jsonl')
example_prompt, ground_truth_action_list = process_query_data(queries[0])
```

---

## RAP (Reasoning with Action Plans)

We implement the RAP approach as demonstrated in the [demo.ipynb](https://github.com/maitrix-org/llm-reasoners/blob/main/demo.ipynb) 
Key classes:
1. **`BlocksWorldModelRAP`**: Handles state transitions and terminal state checks.
2. **`BWConfigRAP`**: Manages action generation and reward computation.

---

## Algorithms and Search

The demo uses **Monte Carlo Tree Search (MCTS)** to generate reasoning plans with depth and iteration limits.

```python
algorithm = MCTS(depth_limit=4, disable_tqdm=False, output_trace_in_each_iter=True, n_iters=10)
```

---

## Evaluation

The evaluator integrates custom prompts and domain configurations to benchmark reasoning performance against ground-truth plans.

```python
evaluator = BWEvaluator(
    config_file="./examples/CoT/blocksworld/data/bw_config.yaml",
    domain_file="./examples/CoT/blocksworld/data/generated_domain.pddl",
    data_path="./examples/CoT/blocksworld/data/split_v1/split_v1_step_4_data.json",
    init_prompt=prompt,
)
```

---

## Results and Comparison

The final step compares the generated tree-traced plan with the ground-truth plan to evaluate the algorithm's reasoning accuracy.

```python

world_model = BlocksWorldModelRAP(base_model=model, prompt=prompt, max_steps=4)
config = BWConfigRAP(base_model=model, prompt=prompt)

algorithm = MCTS(depth_limit=4, disable_tqdm=False, output_trace_in_each_iter=True, n_iters=10)

reasoner_rap = Reasoner(world_model=world_model, search_config=config, search_algo=algorithm)
result_rap = reasoner_rap(example_prompt)

print(result_rap.trace[1], ground_truth_action_list)

>> (['unstack the yellow block from on top of the orange block',
  'put down the yellow block',
  'pick up the orange block',
  'stack the orange block on top of the red block']
  
 ['unstack yellow orange',
  'put-down yellow',
  'pick-up orange',
  'stack orange red'])

```


---

