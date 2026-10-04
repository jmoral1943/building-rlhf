# !pip install tiktoken

import torch
from torch.utils.data import DataLoader,Dataset
import torch.nn as nn
from torch.optim import AdamW

import tiktoken

from transformers import GPT2Config, GPT2LMHeadModel

from datasets import load_dataset


class ChatDataset(Dataset):
  def __init__(self, textArr):
    self.textArr = textArr
    self.enc = tiktoken.encoding_for_model("gpt-4o")

    self.pad_token_id = 0

  def __len__(self):
    return len(self.textArr)

  def __getitem__(self, idx):
    prompt_tokens = self.enc.encode(self.textArr[idx][0])
    chosen_tokens = self.enc.encode(self.textArr[idx][1])
    rejected_tokens = self.enc.encode(self.textArr[idx][2])


    chosen = prompt_tokens + chosen_tokens

    rejected = prompt_tokens + rejected_tokens

    return {
      "chosen": torch.tensor(chosen, dtype=torch.long),
      "rejected": torch.tensor(rejected, dtype=torch.long)
    }

class AnthropicChatDataset(Dataset):
  def __init__(self, hf_dataset):
    self.hf_dataset = hf_dataset
    self.enc = tiktoken.encoding_for_model("gpt-4o")

    self.pad_token_id = 0

    self.max_len = 128


  def __len__(self):
    return len(self.hf_dataset['train'])

  def __getitem__(self, idx):
    row = self.hf_dataset['train'][idx]
    
    # Pull the chosen and rejected text directly from the HF row
    # todo: update the limit with flash attention, RoPE, or YaRN
    chosen_ids = self.enc.encode(row['chosen'])[:self.max_len]
    rejected_ids = self.enc.encode(row['rejected'])[:self.max_len]

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

enc = tiktoken.encoding_for_model("gpt-4o")

# Ask it for its exact dictionary size
exact_vocab_size = enc.n_vocab

config = GPT2Config(
    vocab_size=exact_vocab_size,
    n_positions=128,
    n_embd=64,
    n_layer=2,
    n_head=4
)

model = GPT2LMHeadModel(config)
frozen_model = GPT2LMHeadModel(config)

model.to(device)
frozen_model.to(device)

# Set frozen model to evaluation mode (crucial for reference models)
frozen_model.eval()

beta = 0.1

# Load all helpfulness/harmless subsets (share the same schema)
dataset = load_dataset("Anthropic/hh-rlhf")
chat_dataset = AnthropicChatDataset(dataset)
dataLoader = DataLoader(chat_dataset, batch_size=8, shuffle=True)


from tqdm import tqdm
import matplotlib.pyplot as plt
from IPython.display import clear_output

optmizer = AdamW(model.parameters())

num_epochs = 10
loss_history = [] # Array to store the loss over time
max_steps = 100  # Stop after 100 steps

for epoch in range(num_epochs):
  print(f"\nStarting Epoch {epoch + 1}/{num_epochs}")

  loop = tqdm(dataLoader, desc="Processing Batches")

  for step, batch in enumerate(loop):
    # todo remove for real steps
    # if step >= max_steps - 1:
    #   break

    chosen = batch['chosen'].to(device)
    rejected = batch['rejected'].to(device)
    chosen_attention_mask = batch['chosen_attention_mask'].to(device)
    rejected_attention_mask = batch['rejected_attention_mask'].to(device)

    chosen_logits = model(chosen, attention_mask=chosen_attention_mask).logits
    frozen_chosen_logits = frozen_model(chosen, attention_mask=chosen_attention_mask).logits

    shifted_chosen_logits = chosen_logits[:, :-1, :]
    f_shifted_chosen_logits = frozen_chosen_logits[:, :-1, :]
    shifted_chosen_labels = chosen[:, 1:]

    chosen_probs = nn.functional.log_softmax(shifted_chosen_logits, dim=-1)
    frozen_chosen_probs = nn.functional.log_softmax(f_shifted_chosen_logits, dim=-1)

    chain_chosen_probs = torch.gather(chosen_probs, dim=-1, index=shifted_chosen_labels.unsqueeze(-1)).squeeze(-1)
    frozen_chain_chosen_probs = torch.gather(frozen_chosen_probs, dim=-1, index=shifted_chosen_labels.unsqueeze(-1)).squeeze(-1)

    # 1. Shift the mask to match the shifted labels
    shifted_chosen_mask = chosen_attention_mask[:, 1:]

    # 2. Multiply the probabilities by the mask (padding becomes 0.0)
    masked_chosen_probs = chain_chosen_probs * shifted_chosen_mask

    # 3. Sum only the real tokens
    chosen_sentence_prob = masked_chosen_probs.sum(dim=-1)
   
    masked_f_chosen_probs = frozen_chain_chosen_probs * shifted_chosen_mask

    frozen_chosen_sentence_prob = masked_f_chosen_probs.sum(dim=-1)

    chosen_loss = beta * (chosen_sentence_prob - frozen_chosen_sentence_prob)



    rejected_logits = model(rejected, attention_mask=rejected_attention_mask).logits
    f_rejected_logits = frozen_model(rejected, attention_mask=rejected_attention_mask).logits

    shifted_rejected_logits = rejected_logits[:, :-1, :]
    f_shifted_rejected_logits = f_rejected_logits[:, :-1, :]
    shifted_rejected_labels = rejected[:, 1:]

    rejected_probs = nn.functional.log_softmax(shifted_rejected_logits, dim=-1)
    f_rejected_probs = nn.functional.log_softmax(f_shifted_rejected_logits, dim=-1)

    chain_rejected_probs = torch.gather(rejected_probs, dim=-1, index=shifted_rejected_labels.unsqueeze(-1)).squeeze(-1)
    f_chain_rejected_probs = torch.gather(f_rejected_probs, dim=-1, index=shifted_rejected_labels.unsqueeze(-1)).squeeze(-1)

    # 1. Shift the mask to match the shifted labels
    shifted_rejected_mask = rejected_attention_mask[:, 1:]

    # 2. Multiply the probabilities by the mask (padding becomes 0.0)
    masked_rejected_probs = chain_rejected_probs * shifted_rejected_mask

    # 3. Sum only the real tokens
    rejected_sentence_probs = masked_rejected_probs.sum(dim=-1)

    masked_f_rejected_probs = f_chain_rejected_probs * shifted_rejected_mask

    f_rejected_sentence_probs = masked_f_rejected_probs.sum(dim=-1)
    
    rejected_loss = beta * (rejected_sentence_probs - f_rejected_sentence_probs)


    optmizer.zero_grad()

    
    loss = -nn.functional.logsigmoid(chosen_loss - rejected_loss).mean()
    loss.backward()

    optmizer.step()

    loss_history.append(loss.item())

    if step % 1000 == 0:
      clear_output(wait=True)

      plt.figure(figsize=(10, 5))
      plt.plot(loss_history, color='blue', label='Margin Loss')
      plt.title('Reward Model Training Loss')
      plt.xlabel('Training Steps')
      plt.ylabel('Loss')
      plt.grid(True, linestyle='--', alpha=0.7)
      plt.legend()

      plt.show()

