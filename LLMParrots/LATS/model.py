from transformers import AutoTokenizer, LlamaForCausalLM
import torch
import logging
import os

model_name = "meta-llama/Llama-3.1-8B-Instruct"
hf_token = "TOKEN"

logger = logging.getLogger(__name__)

model_folder = "saved_model"
if not os.path.exists(model_folder):
    os.makedirs(model_folder)

try:
    model = LlamaForCausalLM.from_pretrained(model_folder)
    tokenizer = AutoTokenizer.from_pretrained(model_folder)
except:
    model = LlamaForCausalLM.from_pretrained(model_name, token=hf_token)
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    # model.save_pretrained(model_folder)
    # tokenizer.save_pretrained(model_folder)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

if torch.cuda.device_count() > 1:
    # print(f"Using {torch.cuda.device_count()} GPUs!")
    logger.info(f"Using {torch.cuda.device_count()} GPUs!")
    model = torch.nn.DataParallel(model)

def prompt_model(prompt, character_count=None):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    generate_ids = model.module.generate(inputs.input_ids, max_new_tokens = 500)
    output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    if character_count:
        return output[len(prompt):].strip()[:character_count]
    return output[len(prompt):].strip()