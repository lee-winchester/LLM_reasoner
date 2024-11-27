# RAP Implementation with Llama-3.1-8B-Instruct

This repository implements Reasoning-Aware Planning (RAP) using the Hugging Face `Llama-3.1-8B-Instruct` model. The `hf_model` configuration leverages private models using `hf_tokens` for secure access.

---

## Key Updates
1. **Model Configuration**:
   - The `Reasoners/lm/hf_model` is configured to support `hf_tokens` for the private model `Llama-3.1-8B-Instruct`.
   - Removed unused models:
     - `anthropic_model`
     - `exlllama_model`
     - `gemini_model`
     - `llama2_model`
     - `llama3_model`
     - `llama_Cpp_model`
     - `llama_model`
     - `openai_hf_model`
     - `openai_model`
   - Focused exclusively on integrating `Llama-3.1-8B-Instruct`.

2. **Streamlined Dependencies**:
   - Removed excess model implementations to simplify the codebase and ensure smooth operation.

3. **Enhanced Usability**:
   - Made the necessary changes to enable the `Llama-3.1-8B-Instruct` model to run seamlessly.

4. **Interactive Demonstration**:
   - Open the `demo.ipynb` notebook and run the cells directly for an interactive RAP demonstration.

---

## Changes in Examples
The folder `Examples/RAP/blocksworld` has been updated to align with the new configuration and to work with `Llama-3.1-8B-Instruct`.

---

## Run Instructions

We provide scripts to reproduce the results of RAP using `Llama-3.1-8B-Instruct`. To reproduce results or experiment with other models, follow these steps:

1. **Rune the demo.ipynb**
   ```bash
   ./demo.ipynb

1. **Run the RAP Script**:
   ```bash
   ./examples/RAP/blocksworld/test_rap_llama2.sh
