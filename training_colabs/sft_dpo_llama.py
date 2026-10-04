# !pip install -U transformers huggingface_hub accelerate peft torchao

import os
from huggingface_hub import login

hf_token = os.getenv("HF_TOKEN")

if not hf_token:
    raise ValueError("CRITICAL: HF_TOKEN environment variable not found!")

login(token=hf_token)

import torch 
from torch.utils.data import DataLoader,Dataset
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset


model_id = "meta-llama/Llama-3.2-3B"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16, 
    device_map="cuda"
)

class SFTAnthropicChatDataset(Dataset):
  def __init__(self, hf_dataset):
    self.hf_dataset = hf_dataset
    self.tokenizer = tokenizer

    if self.tokenizer.pad_token_id is None:
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    self.max_len = 1024


  def __len__(self):
    return len(self.hf_dataset['train'])

  def __getitem__(self, idx):
    row = self.hf_dataset['train'][idx]
    chosen_text = row['chosen']

    if "\n\nAssistant:" in chosen_text:
        prompt_text, completion_text = chosen_text.rsplit("\n\nAssistant:", 1)
        prompt_text += "\n\nAssistant:"  # Attach tag back to the prompt side
    else:
        prompt_text = ""
        completion_text = chosen_text
      
    prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
    completion_ids = self.tokenizer.encode(completion_text, add_special_tokens=False)

    max_prompt_len = self.max_len - min(len(completion_ids), self.max_len // 2)
    if len(prompt_ids) > max_prompt_len:
        prompt_ids = prompt_ids[-max_prompt_len:]

    full_ids = (prompt_ids + completion_ids)[:self.max_len]

    prompt_len = min(len(prompt_ids), len(full_ids))
    labels = [-100] * prompt_len + full_ids[prompt_len:]

    pad_len = self.max_len - len(full_ids)
    input_ids = full_ids + [self.tokenizer.pad_token_id] * pad_len
    attention_mask = [1]* len(full_ids) + [0] * pad_len
    labels = labels + [-100] * pad_len
    
    return {
        'input_ids': torch.tensor(input_ids, dtype=torch.long),
        'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
        'labels': torch.tensor(labels, dtype=torch.long)
    }


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,               
    lora_alpha=32,      
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"] 
)

peft_model = get_peft_model(model, lora_config)

peft_model.print_trainable_parameters()

peft_model.gradient_checkpointing_enable()

loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

# hh-rlhf
dataset = load_dataset("Anthropic/hh-rlhf")
chat_dataset = SFTAnthropicChatDataset(dataset)

# TODO update batch size to 32,16, or 8
dataLoader = DataLoader(chat_dataset, batch_size=1, shuffle=False)

optmizer = AdamW(peft_model.parameters())
peft_model.train()

for batch in dataLoader:
    inputs = batch['input_ids'].to(device)
    labels = batch['labels'].to(device)
    mask = batch['attention_mask'].to(device)
    
    logits = peft_model(inputs, attention_mask=mask).logits
    
    optmizer.zero_grad()
    
    shift_logits = logits[..., :-1, :].contiguous()
    # Take all batches, all sequences except the first one
    shift_labels = labels[..., 1:].contiguous()
    flat_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_labels = shift_labels.view(-1)
    
    output = loss_fn(flat_logits, flat_labels)
    output.backward()
    
    optmizer.step()

    print(f"SFT Batch Loss: {output.item():.4f}")
    break  # Stops after one batch to prove it works

peft_model.save_pretrained("./sft_llama_adapter")

import gc
import torch

# Safely delete any model variables if they exist in memory
for var_name in ['model', 'peft_model', 'model_base', 'frozen_model_base', 'frozen_model']:
    if var_name in globals():
        del globals()[var_name]

gc.collect()
torch.cuda.empty_cache()
print("VRAM successfully cleared!")

class DPOAnthropicChatDataset(Dataset):
  def __init__(self, hf_dataset):
    self.hf_dataset = hf_dataset
    self.tokenizer = tokenizer

    if self.tokenizer.pad_token_id is None:
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    self.max_len = 1024


  def __len__(self):
    return len(self.hf_dataset['train'])

  def __getitem__(self, idx):
    row = self.hf_dataset['train'][idx]
    
    # Pull the chosen and rejected text directly from the HF row
    # todo: update the limit with flash attention, RoPE, or YaRN
    chosen_ids = self.tokenizer.encode(row['chosen'])[:self.max_len]
    rejected_ids = self.tokenizer.encode(row['rejected'])[:self.max_len]

    actual_c_len = len(chosen_ids)
    pad_len = self.max_len - actual_c_len

    chosen_ids += [self.pad_token_id] * pad_len

    chosen_attention_mask = [1] * actual_c_len + [0] * pad_len

    actual_r_len = len(rejected_ids)
    pad_len = self.max_len - actual_r_len
    rejected_ids += [self.pad_token_id] * pad_len
    rejected_attention_mask = [1] * actual_r_len + [0] * pad_len

    
    return {
        'chosen': torch.tensor(chosen_ids, dtype=torch.long),
        'rejected': torch.tensor(rejected_ids, dtype=torch.long),
        'chosen_attention_mask': torch.tensor(chosen_attention_mask, dtype=torch.long),
        'rejected_attention_mask': torch.tensor(rejected_attention_mask, dtype=torch.long),
    }

from peft import PeftModel

frozen_model_base = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map="cuda"
)

frozen_model = PeftModel.from_pretrained(frozen_model_base, "./sft_llama_adapter")
# Set frozen model to evaluation mode (crucial for reference models)
frozen_model.eval()

model_base = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map="cuda"
)
model = PeftModel.from_pretrained(
    model_base, "./sft_llama_adapter", is_trainable=True
)

model.gradient_checkpointing_enable()

beta = 0.1

chat_dataset = DPOAnthropicChatDataset(dataset)
# TODO update batch size to 32,16, or 8
dataLoader = DataLoader(chat_dataset, batch_size=1, shuffle=True)

from tqdm import tqdm
import matplotlib.pyplot as plt
from IPython.display import clear_output
import torch.nn as nn
import torch
from torch.optim import AdamW

optmizer = AdamW(model.parameters())

num_epochs = 10
loss_history = [] 
max_steps = 100  

for epoch in range(num_epochs):
    print(f"\nStarting Epoch {epoch + 1}/{num_epochs}")
    loop = tqdm(dataLoader, desc="Processing Batches")

    for step, batch in enumerate(loop):
        if step >= max_steps - 1:
            break

        # Dynamically route inputs to wherever the active model lives
        chosen = batch['chosen'].to(model.device)
        rejected = batch['rejected'].to(model.device)
        chosen_attention_mask = batch['chosen_attention_mask'].to(model.device)
        rejected_attention_mask = batch['rejected_attention_mask'].to(model.device)

        # Dynamically route copies of inputs to wherever the frozen model lives
        chosen_f = batch['chosen'].to(frozen_model.device)
        rejected_f = batch['rejected'].to(frozen_model.device)
        chosen_attention_mask_f = batch['chosen_attention_mask'].to(frozen_model.device)
        rejected_attention_mask_f = batch['rejected_attention_mask'].to(frozen_model.device)

        # --- Active Model Forward Passes ---
        chosen_logits = model(chosen, attention_mask=chosen_attention_mask).logits
        rejected_logits = model(rejected, attention_mask=rejected_attention_mask).logits

        # --- Frozen Model Forward Passes (Must be in no_grad!) ---
        with torch.no_grad():
            frozen_chosen_logits = frozen_model(chosen_f, attention_mask=chosen_attention_mask_f).logits
            f_rejected_logits = frozen_model(rejected_f, attention_mask=rejected_attention_mask_f).logits
            
            # Move the frozen logits back to the active model's device to calculate loss together
            frozen_chosen_logits = frozen_chosen_logits.to(model.device)
            f_rejected_logits = f_rejected_logits.to(model.device)

        # --- Chosen Math ---
        shifted_chosen_logits = chosen_logits[:, :-1, :]
        f_shifted_chosen_logits = frozen_chosen_logits[:, :-1, :]
        shifted_chosen_labels = chosen[:, 1:]

        chosen_probs = nn.functional.log_softmax(shifted_chosen_logits, dim=-1)
        frozen_chosen_probs = nn.functional.log_softmax(f_shifted_chosen_logits, dim=-1)

        chain_chosen_probs = torch.gather(chosen_probs, dim=-1, index=shifted_chosen_labels.unsqueeze(-1)).squeeze(-1)
        frozen_chain_chosen_probs = torch.gather(frozen_chosen_probs, dim=-1, index=shifted_chosen_labels.unsqueeze(-1)).squeeze(-1)

        shifted_chosen_mask = chosen_attention_mask[:, 1:]
        masked_chosen_probs = chain_chosen_probs * shifted_chosen_mask
        chosen_sentence_prob = masked_chosen_probs.sum(dim=-1)
        
        masked_f_chosen_probs = frozen_chain_chosen_probs * shifted_chosen_mask
        frozen_chosen_sentence_prob = masked_f_chosen_probs.sum(dim=-1)

        chosen_loss = beta * (chosen_sentence_prob - frozen_chosen_sentence_prob)

        # --- Rejected Math ---
        shifted_rejected_logits = rejected_logits[:, :-1, :]
        f_shifted_rejected_logits = f_rejected_logits[:, :-1, :]
        shifted_rejected_labels = rejected[:, 1:]

        rejected_probs = nn.functional.log_softmax(shifted_rejected_logits, dim=-1)
        f_rejected_probs = nn.functional.log_softmax(f_shifted_rejected_logits, dim=-1)

        chain_rejected_probs = torch.gather(rejected_probs, dim=-1, index=shifted_rejected_labels.unsqueeze(-1)).squeeze(-1)
        f_chain_rejected_probs = torch.gather(f_rejected_probs, dim=-1, index=shifted_rejected_labels.unsqueeze(-1)).squeeze(-1)

        shifted_rejected_mask = rejected_attention_mask[:, 1:]
        masked_rejected_probs = chain_rejected_probs * shifted_rejected_mask
        rejected_sentence_probs = masked_rejected_probs.sum(dim=-1)

        masked_f_rejected_probs = f_chain_rejected_probs * shifted_rejected_mask
        f_rejected_sentence_probs = masked_f_rejected_probs.sum(dim=-1)
        
        rejected_loss = beta * (rejected_sentence_probs - f_rejected_sentence_probs)

        # --- Optimization ---
        optmizer.zero_grad()
        loss = -nn.functional.logsigmoid(chosen_loss - rejected_loss).mean()
        loss.backward()
        optmizer.step()

        loss_history.append(loss.item())

        if step % 100 == 0 and step > 0:
            clear_output(wait=True)
            plt.figure(figsize=(10, 5))
            plt.plot(loss_history, color='blue', label='Margin Loss')
            plt.title('Reward Model Training Loss')
            plt.xlabel('Training Steps')
            plt.ylabel('Loss')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            plt.show()

model.save_pretrained("./dpo_llama_adapter")
print("Successfully saved DPO adapter to ./dpo_llama_adapter")

import csv
import json
import time
import pandas as pd
import google.generativeai as genai
from tqdm import tqdm
import os

# 1. Fetch the key from the system environment
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("CRITICAL: GEMINI_API_KEY environment variable not found!")

# 2. Authenticate
genai.configure(api_key=api_key)
judge = genai.GenerativeModel("gemini-3.8-flash")

csv_filename = "phase_4_4_evaluation.csv"
fieldnames = ["Prompt", "DPO_Response", "SFT_Response", "Winner", "Judge_Reasoning"]

# 2. Initialize the CSV file with headers
with open(csv_filename, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

print(f"Initialized {csv_filename}. Starting row-by-row streaming evaluation...")

# 3. Stream through dataset and write each row immediately
test_split = dataset['test']

for i in tqdm(range(100), desc="Evaluating Prompts"):
    # Extract prompt
    chosen_text = test_split[i]['chosen']
    prompt_text = chosen_text.rsplit("\n\nAssistant:", 1)[0] + "\n\nAssistant:"
    
    # Generate model outputs
    inputs = tokenizer(prompt_text, return_tensors="pt")
    active_inputs = inputs.to(model.device)
    frozen_inputs = inputs.to(frozen_model.device)
    
    with torch.no_grad():
        dpo_out = model.generate(**active_inputs, max_new_tokens=150, do_sample=True, pad_token_id=tokenizer.eos_token_id)
        sft_out = frozen_model.generate(**frozen_inputs, max_new_tokens=150, do_sample=True, pad_token_id=tokenizer.eos_token_id)
    
    input_len = inputs['input_ids'].shape[1]
    dpo_resp = tokenizer.decode(dpo_out[0][input_len:], skip_special_tokens=True)
    sft_resp = tokenizer.decode(sft_out[0][input_len:], skip_special_tokens=True)
    
    # Evaluate with Gemini
    judge_prompt = f"""
    You are an AI judge. Which response is more helpful and accurate?
    Prompt: {prompt_text}
    Response A: {dpo_resp}
    Response B: {sft_resp}
    
    Return ONLY a raw JSON object: {{"reasoning": "...", "winner": "A", "or": "B", "or": "Tie"}}
    """
    
    try:
        api_response = judge.generate_content(
            judge_prompt, generation_config={"response_mime_type": "application/json"}
        )
        eval_data = json.loads(api_response.text)
        winner = eval_data.get("winner", "Tie")
        reasoning = eval_data.get("reasoning", "")
    except Exception as e:
        winner = "Error"
        reasoning = str(e)

    # Append row directly to disk
    with open(csv_filename, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow({
            "Prompt": prompt_text,
            "DPO_Response": dpo_resp,
            "SFT_Response": sft_resp,
            "Winner": winner,
            "Judge_Reasoning": reasoning
        })

    time.sleep(4.1)  # Respect free-tier rate limits

# 4. Load saved CSV into a Pandas DataFrame for analysis
df = pd.read_csv(csv_filename)
print("\n--- Final Results Loaded into DataFrame ---")
print(df["Winner"].value_counts())

