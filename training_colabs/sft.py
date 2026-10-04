# !pip install tiktoken

import torch 
from torch.utils.data import DataLoader,Dataset
import torch.nn as nn
from torch.optim import AdamW

import tiktoken

from transformers import GPT2Config, GPT2LMHeadModel

class ChatDataset(Dataset):
  def __init__(self, textArr):
    self.textArr = textArr
    self.enc = tiktoken.encoding_for_model("gpt-4o")

    self.max = max(len(self.enc.encode(a + b)) for a, b in self.textArr)

    self.pad_token_id = 0

  def __len__(self):
    return len(self.textArr)

  def __getitem__(self, idx):
    prompt_tokens = self.enc.encode(self.textArr[idx][0])
    answer_tokens = self.enc.encode(self.textArr[idx][1])

    actual_len = len(prompt_tokens) + len(answer_tokens)
    pad_len = self.max - actual_len


    input_ids = prompt_tokens + answer_tokens + [self.pad_token_id] * pad_len

    mask = [-100] * len(prompt_tokens)

    labels = mask + answer_tokens + [-100] * pad_len

    attention_mask = [1] * actual_len + [0] * pad_len

    return {
      "input_ids": torch.tensor(input_ids, dtype=torch.long),
      "labels": torch.tensor(labels, dtype=torch.long),
      "attention_mask": torch.tensor(attention_mask, dtype=torch.long)
    }


config = GPT2Config(
    vocab_size=200000,
    n_positions=128,
    n_embd=64,
    n_layer=2,
    n_head=4
)

model= GPT2LMHeadModel(config)

loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

# there are two things 
  # 1. do you either have two chatDataset one will go ahead 
    # a. return getitem for normal inputs and then another dataset class for masked labels
    # b. make the dataset class be one chatDataset and then make it return either or via a different function or something ?

  # 2. how should we store the prompt and then answer ? 
    # a. have the dataset split into prompt and then answer making the dataset class have a pair of arrays ? 
    # b. split via some searching algorithm ?
dataset = [
    ["User: What is the capital of France?\nAssistant", "The capital of France is Paris."],
    ["User: Can you write a quick Python script to reverse a string?\nAssistant", "def reverse_string(s):\n    return s[::-1]"],
    ["User: What is 2 + 2?\nAssistant", "2 + 2 equals 4."],
    ["User: Remind me to buy groceries tomorrow at 5 PM.\nAssistant", "I've noted that down for tomorrow at 5 PM."],
    ["User: Explain quantum computing in one sentence.\nAssistant", "Quantum computing uses the principles of quantum mechanics to process complex data simultaneously using qubits instead of traditional bits."],
    ["User: How do I center a div in CSS?\nAssistant", "You can use flexbox: display: flex; justify-content: center; align-items: center;"],
    ['User: Translate "Hello, how are you?" into Spanish.\nAssistant', '"Hola, ¿cómo estás?"'],
    ["User: What is the tallest mountain in the world?\nAssistant", "Mount Everest is the tallest mountain above sea level."],
    ["User: Give me a random fun fact about space.\nAssistant", "A day on Venus is longer than its year."],
    ["User: Debug this SQL query: SELECT * FROM users WHERE;\nAssistant", "It looks like your query is cut off. You need to specify a condition after the WHERE clause, such as SELECT * FROM users WHERE id = 1;"]
]

chat_dataset = ChatDataset(dataset)

dataLoader = DataLoader(chat_dataset, batch_size=5, shuffle=False)

optmizer = AdamW(model.parameters())

for batch in dataLoader:
  inputs = batch['input_ids']
  labels = batch['labels']
  mask = batch['attention_mask']

  # print(inputs)
  logits = model(inputs, attention_mask=mask).logits

  optmizer.zero_grad()

  shift_logits = logits[..., :-1, :].contiguous()
  # Take all batches, all sequences except the first one
  shift_labels = labels[..., 1:].contiguous()
  flat_logits = shift_logits.view(-1, shift_logits.size(-1))
  flat_labels = shift_labels.view(-1)

  output = loss_fn(flat_logits, flat_labels)
  output.backward()

  optmizer.step()



