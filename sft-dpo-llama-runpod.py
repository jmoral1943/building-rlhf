#!/usr/bin/env python
# coding: utf-8

# In[1]:




# In[ ]:

# In[3]:




# In[ ]:




# In[5]:



import os
from dotenv import load_dotenv
from huggingface_hub import login
import google.generativeai as genai

# This automatically finds the .env file and loads the keys into os.environ
load_dotenv()

# Abstracted authentication
hf_token = os.getenv("HF_TOKEN")
gemini_key = os.getenv("GEMINI_API_KEY")

login(token=hf_token)
genai.configure(api_key=gemini_key)


# # SFT

# In[ ]:

import os
import gc
import csv
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login
import google.generativeai as genai
from tqdm import tqdm
import pandas as pd

# --- 0. ENVIRONMENT SETUP ---
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
load_dotenv()

hf_token = os.getenv("HF_TOKEN")
gemini_key = os.getenv("GEMINI_API_KEY")

login(token=hf_token)
genai.configure(api_key=gemini_key)

model_id = "meta-llama/Llama-3.2-3B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

dataset = load_dataset("Anthropic/hh-rlhf")

# ==========================================
# PHASE 1: SFT (Supervised Fine-Tuning)
# ==========================================
sft_adapter_path = "./sft_llama_adapter"

if os.path.exists(sft_adapter_path):
    print(f"\n[INFO] Found existing SFT adapter at '{sft_adapter_path}'. Skipping SFT training!")
else:
    print(f"\n[INFO] No SFT adapter found. Starting SFT training...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map="cuda", use_cache=False
    )

    class SFTAnthropicChatDataset(Dataset):
        def __init__(self, hf_dataset):
            self.hf_dataset = hf_dataset
            self.tokenizer = tokenizer
            self.max_len = 1024

        def __len__(self):
            return len(self.hf_dataset['train'])

        def __getitem__(self, idx):
            row = self.hf_dataset['train'][idx]
            chosen_text = row['chosen']
            if "\n\nAssistant:" in chosen_text:
                prompt_text, completion_text = chosen_text.rsplit("\n\nAssistant:", 1)
                prompt_text += "\n\nAssistant:"
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
            
            return {
                'input_ids': torch.tensor(full_ids + [self.tokenizer.pad_token_id] * pad_len, dtype=torch.long),
                'attention_mask': torch.tensor([1] * len(full_ids) + [0] * pad_len, dtype=torch.long),
                'labels': torch.tensor(labels + [-100] * pad_len, dtype=torch.long)
            }

    lora_config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj", "v_proj"])
    peft_model = get_peft_model(model, lora_config)
    peft_model.gradient_checkpointing_enable()

    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    sft_dataloader = DataLoader(SFTAnthropicChatDataset({'train': dataset['train'].select(range(10000))}), batch_size=32, shuffle=False)
    optimizer = AdamW(peft_model.parameters(), fused=True)
    peft_model.train()

    for epoch in range(1):
        loop = tqdm(sft_dataloader, desc="SFT Processing Batches")
        for step, batch in enumerate(loop):
            logits = peft_model(batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device)).logits
            optimizer.zero_grad()
            loss = loss_fn(logits[..., :-1, :].contiguous().view(-1, logits.size(-1)), batch['labels'].to(device)[..., 1:].contiguous().view(-1))
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())

    peft_model.save_pretrained(sft_adapter_path)
    del model, peft_model, optimizer
    gc.collect()
    torch.cuda.empty_cache()


# ==========================================
# PHASE 2: DPO (Direct Preference Optimization)
# ==========================================
dpo_adapter_path = "./dpo_llama_adapter"

if os.path.exists(dpo_adapter_path):
    print(f"\n[INFO] Found existing DPO adapter at '{dpo_adapter_path}'. Skipping DPO training!")
    # Load models directly for evaluation phase
    frozen_model_base = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="cuda:1", attn_implementation="flash_attention_2")
    frozen_model = PeftModel.from_pretrained(frozen_model_base, sft_adapter_path)
    frozen_model.eval()

    model_base = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="flash_attention_2", use_cache=False)
    model = PeftModel.from_pretrained(model_base, dpo_adapter_path)
    model.eval()
else:
    print(f"\n[INFO] No DPO adapter found. Starting DPO training...")
    class DPOAnthropicChatDataset(Dataset):
        def __init__(self, hf_dataset):
            self.hf_dataset = hf_dataset
            self.tokenizer = tokenizer
            self.max_len = 1024

        def __len__(self):
            return len(self.hf_dataset['train'])

        def __getitem__(self, idx):
            row = self.hf_dataset['train'][idx]
            c_ids = self.tokenizer.encode(row['chosen'])[:self.max_len]
            r_ids = self.tokenizer.encode(row['rejected'])[:self.max_len]
            
            c_pad = self.max_len - len(c_ids)
            r_pad = self.max_len - len(r_ids)
            
            return {
                'chosen': torch.tensor(c_ids + [self.tokenizer.pad_token_id] * c_pad, dtype=torch.long),
                'rejected': torch.tensor(r_ids + [self.tokenizer.pad_token_id] * r_pad, dtype=torch.long),
                'chosen_attention_mask': torch.tensor([1] * len(c_ids) + [0] * c_pad, dtype=torch.long),
                'rejected_attention_mask': torch.tensor([1] * len(r_ids) + [0] * r_pad, dtype=torch.long),
            }

    frozen_model_base = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="cuda:1", attn_implementation="flash_attention_2")
    frozen_model = PeftModel.from_pretrained(frozen_model_base, sft_adapter_path)
    frozen_model.eval()

    model_base = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="flash_attention_2", use_cache=False)
    model = PeftModel.from_pretrained(model_base, sft_adapter_path, is_trainable=True)
    model.gradient_checkpointing_enable()

    beta = 0.1
    dpo_dataloader = DataLoader(DPOAnthropicChatDataset({'train': dataset['train'].select(range(5000))}), batch_size=4, shuffle=True)
    optimizer = AdamW(model.parameters(), fused=True)

    for epoch in range(1):
        loop = tqdm(dpo_dataloader, desc="Processing DPO Batches")
        for step, batch in enumerate(loop):
            chosen = batch['chosen'].to(model.device)
            rejected = batch['rejected'].to(model.device)
            c_mask = batch['chosen_attention_mask'].to(model.device)
            r_mask = batch['rejected_attention_mask'].to(model.device)

            c_logits = model(chosen, attention_mask=c_mask).logits
            r_logits = model(rejected, attention_mask=r_mask).logits

            with torch.no_grad():
                f_c_logits = frozen_model(batch['chosen'].to(frozen_model.device), attention_mask=batch['chosen_attention_mask'].to(frozen_model.device)).logits.to(model.device)
                f_r_logits = frozen_model(batch['rejected'].to(frozen_model.device), attention_mask=batch['rejected_attention_mask'].to(frozen_model.device)).logits.to(model.device)

            c_loss = beta * ((torch.gather(nn.functional.log_softmax(c_logits[:, :-1, :], dim=-1), -1, chosen[:, 1:].unsqueeze(-1)).squeeze(-1) * c_mask[:, 1:]).sum(dim=-1) - 
                             (torch.gather(nn.functional.log_softmax(f_c_logits[:, :-1, :], dim=-1), -1, chosen[:, 1:].unsqueeze(-1)).squeeze(-1) * c_mask[:, 1:]).sum(dim=-1))
            
            r_loss = beta * ((torch.gather(nn.functional.log_softmax(r_logits[:, :-1, :], dim=-1), -1, rejected[:, 1:].unsqueeze(-1)).squeeze(-1) * r_mask[:, 1:]).sum(dim=-1) - 
                             (torch.gather(nn.functional.log_softmax(f_r_logits[:, :-1, :], dim=-1), -1, rejected[:, 1:].unsqueeze(-1)).squeeze(-1) * r_mask[:, 1:]).sum(dim=-1))

            optimizer.zero_grad()
            loss = -nn.functional.logsigmoid(c_loss - r_loss).mean()
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())

    model.save_pretrained(dpo_adapter_path)
    print(f"Successfully saved DPO adapter to {dpo_adapter_path}")


# ==========================================
# PHASE 3: LLM-As-a-Judge Evaluation
# ==========================================
csv_filename = "phase_4_4_evaluation.csv"
fieldnames = ["Prompt", "DPO_Response", "SFT_Response", "Winner", "Judge_Reasoning"]

if os.path.exists(csv_filename):
    df_existing = pd.read_csv(csv_filename)
    completed_count = len(df_existing)
    print(f"\n[INFO] Found existing evaluation CSV with {completed_count} evaluated rows.")
else:
    completed_count = 0
    with open(csv_filename, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

if completed_count >= 100:
    print("[INFO] Evaluation already fully completed!")
    df = pd.read_csv(csv_filename)
    print("\n--- Final Results Loaded into DataFrame ---")
    print(df["Winner"].value_counts())
else:
    print(f"\n[INFO] Resuming Gemini LLM-as-a-Judge evaluation from row {completed_count}...")
    judge = genai.GenerativeModel("gemini-3.8-flash")
    test_split = dataset['test']

    for i in tqdm(range(completed_count, 100), desc="Evaluating Prompts", initial=completed_count, total=100):
        chosen_text = test_split[i]['chosen']
        prompt_text = chosen_text.rsplit("\n\nAssistant:", 1)[0] + "\n\nAssistant:"
        
        inputs = tokenizer(prompt_text, return_tensors="pt")
        with torch.no_grad():
            dpo_out = model.generate(**inputs.to(model.device), max_new_tokens=150, do_sample=True, pad_token_id=tokenizer.eos_token_id)
            sft_out = frozen_model.generate(**inputs.to(frozen_model.device), max_new_tokens=150, do_sample=True, pad_token_id=tokenizer.eos_token_id)
        
        input_len = inputs['input_ids'].shape[1]
        dpo_resp = tokenizer.decode(dpo_out[0][input_len:], skip_special_tokens=True)
        sft_resp = tokenizer.decode(sft_out[0][input_len:], skip_special_tokens=True)
        
        judge_prompt = f"""
        You are an AI judge. Which response is more helpful and accurate?
        Prompt: {prompt_text}
        Response A: {dpo_resp}
        Response B: {sft_resp}
        Return ONLY a raw JSON object: {{"reasoning": "...", "winner": "A", "or": "B", "or": "Tie"}}
        """
        
        try:
            api_response = judge.generate_content(judge_prompt, generation_config={"response_mime_type": "application/json"})
            eval_data = json.loads(api_response.text)
            winner = eval_data.get("Winner", eval_data.get("winner", "Tie"))
            reasoning = eval_data.get("reasoning", "")
        except Exception as e:
            winner = "Error"
            reasoning = str(e)

        with open(csv_filename, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow({
                "Prompt": prompt_text, "DPO_Response": dpo_resp, "SFT_Response": sft_resp, "Winner": winner, "Judge_Reasoning": reasoning
            })

        time.sleep(4.1)

    df = pd.read_csv(csv_filename)
    print("\n--- Final Results Loaded into DataFrame ---")
    print(df["Winner"].value_counts())