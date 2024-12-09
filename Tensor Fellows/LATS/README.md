# Language Agent Tree Search (LATS) Implementation for BlocksWorld

## Setup
### Ollama Langchain
#### Prerequisites

1. Python 3.8 or higher
2. Download [Ollama](https://github.com/ollama/ollama?tab=readme-ov-file) for Local Server
3. ollama pull llama3.1

### HuggingFace
#### Prerequisites

1. Python 3.8 or higher
2. CUDA-enabled A100 GPUs

### Initialize the Model
#### Ollama Langchain
```python
from langchain_ollama import ChatOllama
llm = ChatOllama(model="llama3.1")
```
#### HuggingFace
```python
def initialize_llama(system_prompt):
  hf_token = "" # REPLACE WITH PERSONAL TOKEN

  nest_asyncio.apply()

  from transformers import AutoTokenizer

  tokenizer = AutoTokenizer.from_pretrained(
      "meta-llama/Meta-Llama-3.1-8B-Instruct",
      token=hf_token,
  )

  stopping_ids = [
      tokenizer.eos_token_id,
      tokenizer.convert_tokens_to_ids("<|eot_id|>"),
  ]

  llm = HuggingFaceLLM(
      model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
      system_prompt = system_prompt,
      max_new_tokens=256,
      model_kwargs={
          "token": hf_token,
          "torch_dtype": torch.bfloat16,
      },
      generate_kwargs={
          "do_sample": True,
          "temperature": 0.6,
          "top_p": 0.9,
      },
      tokenizer_name=tokenizer,
      tokenizer_kwargs={"token": hf_token},
      stopping_ids=stopping_ids,
  )
  
  return llm
```

## Dataset

The Blocksworld dataset is utilized for demonstration purposes, loaded via `dataset/task_1_plan_generation.json`. It contains block manipulation examples used to create prompts for in-context learning.

```python
with open("dataset/task_1_plan_generation.json", 'r') as file:
    data = json.load(file)
```

## LATS (Language Agent Tree Search) 

We implement the LATS approach in Ollama Langchain as demonstrated in the [lats.ipynb](https://github.com/langchain-ai/langgraph/blob/main/docs/docs/tutorials/lats/lats.ipynb) and in huggingface as demonstrated in the [llama-index](https://docs.llamaindex.ai/en/stable/examples/agent/lats_agent/).

Key classes for Ollama Langchain:
1. **`TreeState`**: Represents the Monte Carlo Tree Search (MCTS) structure which contains all of the proposed reasoning paths.
2. **`Node`**: Represents a single step of reasoning within the Monte Carlo Tree Search (MCTS) structure.
3. **`Reflection`**: Manages reward computation for each action step proposal.

Key classes for HuggingFace:
1. **`LATS`**: Defines the Agent worker that performs a step of Language Agent Tree Search, interface to prompt the model, and tool definition to score the reasoning.

## Algorithms and Search

The demo uses **Monte Carlo Tree Search (MCTS)** to generate reasoning plans with depth and iteration limits.

### Ollama and Langchain
```python
def lats(state: TreeState, N=5):
    print("Starting LATS...")
    state = generate_initial_response(state)
    config = RunnableConfig(configurable={"N": N})
    
    iteration = 0
    while should_loop(state, N):
        iteration += 1
        print(f"Iteration: {iteration}")
        state = expand(state, config)
        
    # After search, get best solution
    print("Search complete.")
    solution_node = state["root"].get_best_solution()
    best_trajectory = solution_node.get_trajectory(include_reflections=False)
    return best_trajectory[-1]
```

### HuggingFace
```python
def call(self, initial, goal):
        response = self.agent.chat(initial + "\n[GOAL]\n\n" + goal)
        return response
```

## Results and Comparison

The final step compares the generated tree-traced plan with the ground-truth plan to evaluate the algorithm's reasoning accuracy.
### Ollama and Langchain
```python
input_text = "I am playing with a set of blocks where I need to arrange the blocks into stacks. Here are the actions I can do\n\nPick up a block\nUnstack a block from on top of another block\nPut down a block\nStack a block on top of another block\n\nI have the following restrictions on my actions:\nI can only pick up or unstack one block at a time.\nI can only pick up or unstack a block if my hand is empty.\nI can only pick up a block if the block is on the table and the block is clear. A block is clear if the block has no other blocks on top of it and if the block is not picked up.\nI can only unstack a block from on top of another block if the block I am unstacking was really on top of the other block.\nI can only unstack a block from on top of another block if the block I am unstacking is clear.\nOnce I pick up or unstack a block, I am holding the block.\nI can only put down a block that I am holding.\nI can only stack a block on top of another block if I am holding the block being stacked.\nI can only stack a block on top of another block if the block onto which I am stacking the block is clear.\nOnce I put down or stack a block, my hand becomes empty.\nOnce you stack a block on top of a second block, the second block is no longer clear.\n\n[STATEMENT]\nAs initial conditions I have that, the red block is clear, the blue block is clear, the yellow block is clear, the hand is empty, the blue block is on top of the orange block, the red block is on the table, the orange block is on the table and the yellow block is on the table.\nMy goal is to have that the orange block is on top of the blue block.\n\nMy plan is as follows:\n\n[PLAN]\nunstack the blue block from on top of the orange block\nput down the blue block\npick up the orange block\nstack the orange block on top of the blue block\n[PLAN END]\n\n"

state = {"input": input_text}
final_response = lats(state, N=3)

print(final_response.content, ground_truth_plan)

>> (['Pick up the blue block',
  'Unstack the red block from the blue block',
  'Put down the blue block',
  'Pick up the yellow block'
  'Unstack the orange block from the yellow block'
  'Put down the yellow block'
  'Stack the orange block on top of the red block']
  
 ['unstack yellow orange',
  'put-down yellow',
  'pick-up orange',
  'stack orange red'])
```
