# Alignment Engineering: Llama 3.2 3B SFT, DPO & LLM-as-a-Judge Pipeline

An end-to-end alignment and preference optimization pipeline for **Llama 3.2 3B** using Supervised Fine-Tuning (SFT), Direct Preference Optimization (DPO), and automated LLM-as-a-Judge evaluation via Google Gemini. Designed for execution on multi-GPU cloud instances with automated fault tolerance, state persistence, and memory optimization.

---

## Architecture Overview

```text
[ Raw Dataset: Anthropic hh-rlhf ]
             │
             ▼
[ Phase 1: SFT (LoRA) ] ────────► Saved Adapter: ./sft_llama_adapter
             │
             ▼
[ Phase 2: DPO Alignment ] ─────► Saved Adapter: ./dpo_llama_adapter
    ├── Active Model (GPU 0)
    └── Frozen Reference (GPU 1)
             │
             ▼
[ Phase 3: Gemini Evaluation ] ──► Final Dataset: phase_4_4_evaluation.csv
```

1. **Supervised Fine-Tuning (SFT):** LoRA adapter training ($r=16, \alpha=32$) on 10,000 prompt-response pairs to establish baseline conversational capability.
2. **Direct Preference Optimization (DPO):** Dual-GPU setup routing the active policy model to GPU 0 and the frozen SFT reference model to GPU 1. Optimizes model likelihoods on 5,000 chosen vs. rejected response pairs.
3. **Automated Evaluation:** Pairwise comparison between SFT baseline and DPO-aligned outputs evaluated by `gemini-3.8-flash` returning structured JSON win/loss judgments.

---

## Benchmark Results

Evaluation was conducted over 100 test prompts from the Anthropic `hh-rlhf` test split. Each model generated up to 150 new tokens per prompt using temperature sampling. Due to severe API rate limits encountered on the Gemini Free Tier (HTTP 429), a secondary programmatic judge script was utilized to repair the evaluation dataset by filtering incoherent and repeating text loops.

| Model / Outcome | Win Count | Win Rate (%) | Performance Notes |
| :--- | :--- | :--- | :--- |
| **DPO Aligned** | 0 | 0% | Complete model collapse; output consisted of broken punctuation and gibberish. |
| **SFT Baseline** | 28 | 28% | Maintained English coherency but frequently succumbed to repetitive generative loops (e.g., repeatedly wishing the user a "nice day"). |
| **Tie / Neither** | 72 | 72% | Both models failed to produce a high-quality, structured response. |

---

## Technical Shortcomings & Analysis

The DPO phase failed to improve upon the SFT baseline, resulting in severe degradation of language modeling capabilities. This was driven by several identifiable engineering constraints during the prototyping phase:

1. **The 1-Epoch Shortcut:** To minimize cloud compute costs during testing, the DPO training loop was restricted to a single epoch on a 5,000-row subset. DPO requires the model to carefully learn the margin between chosen and rejected responses; a single epoch severely underfits and destabilizes the weights before the model can safely converge.
2. **KL Penalty Drift (`beta` mismatch):** The pipeline utilized a standard `beta=0.1`. DPO relies on this parameter to prevent the active policy model from drifting too far from the frozen reference model. Combined with a highly constrained batch size (reduced to `4` to avoid A100 OOM errors), the active model over-optimized and forgot basic sentence structures.
3. **Missing Inference Constraints:** Evaluation inference was run with `do_sample=True` but lacked specific boundaries for `temperature`, `top_p`, or a `repetition_penalty`. Because DPO pushes output probabilities to extremes, sampling without a repetition penalty made the models highly susceptible to hallucination loops.
4. **LoRA Checkpoint Stacking:** The pipeline trained a LoRA for SFT, and then trained *another* LoRA layer on top of it for DPO. Applying unmerged gradients sequentially across separate adapters can cause high gradient instability. 

---

## Next Steps for Pipeline Iteration

To achieve proper preference alignment and defeat the SFT baseline, the following architecture upgrades must be implemented in the next iteration:

* **Merge Weights Before DPO:** Explicitly merge the `sft_llama_adapter` weights into the base Llama 3.2 parameters and save the resulting model to disk before initializing the DPO sequence.
* **Increase DPO Training Volume:** Scale the DPO training phase to 3–5 epochs over the full `hh-rlhf` dataset to allow proper gradient convergence. 
* **Implement Inference Guardrails:** Update the `model.generate()` function in Phase 3 to include `temperature=0.7`, `top_p=0.9`, and `repetition_penalty=1.15` to prevent text loops.
* **Tune Beta and Learning Rate:** Run a hyperparameter sweep testing `beta` values of `0.01` and `0.05` alongside a reduced learning rate to prevent policy collapse.
* **API Rate Limit Management:** Upgrade to a paid tier for the evaluation LLM or implement an exponential backoff function in the evaluation loop to handle HTTP 429 quota errors gracefully.

---

## Hardware Requirements & VRAM Calculations

To compute the minimum VRAM required to run this pipeline, use the following estimation model:

$$	ext{VRAM}_{	ext{total}} = 	ext{VRAM}_{	ext{weights}} + 	ext{VRAM}_{	ext{gradients}} + 	ext{VRAM}_{	ext{optimizer}} + 	ext{VRAM}_{	ext{activations}}$$

For a **3 Billion Parameter Model** (`Llama-3.2-3B`) using **bfloat16** precision:

1. **Base Weights:**
   $$	ext{Weights} = 3.2 \times 10^9 \times 2 \text{ bytes} \approx 6.4 \text{ GB}$$
2. **DPO Dual Model Requirement:**
   * Active Policy Model (GPU 0): $6.4\text{ GB (Weights)} + \text{Gradients} + \text{AdamW Fused States} \approx 18 - 24\text{ GB}$
   * Frozen Reference Model (GPU 1): $6.4\text{ GB (Weights only)} \approx 7\text{ GB}$
3. **Activations (Batch Size = 4, Sequence Length = 1024, FlashAttention-2):**
   $$\text{Activations} \approx 8 - 12 \text{ GB}$$

**Recommended Hardware Setup:**
* **Minimum:** Dual GPUs with $\ge 24\text{ GB}$ VRAM each (e.g., $2\times\text{NVIDIA RTX 4090}$ or $2\times\text{A10G}$).
* **Optimal:** Dual $80\text{ GB}$ GPUs (e.g., $2\times\text{NVIDIA A100}$) for maximum throughput and larger batch sizes.

---

## Quickstart & Reproduction Guide

### 1. Prerequisites
* **Hugging Face Account:** Accept the license terms for `meta-llama/Llama-3.2-3B`.
* **API Keys:** Hugging Face Access Token (`HF_TOKEN`) and Google Gemini API Key (`GEMINI_API_KEY`).

### 2. Environment Setup

Clone this repository and create a `.env` file in the root directory:

```bash
cat <<EOT> .env
HF_TOKEN="your_huggingface_token_here"
GEMINI_API_KEY="your_gemini_api_key_here"
EOT
```

Install system dependencies and Python packages:

```bash
pip install -U transformers huggingface_hub accelerate peft google-generativeai datasets python-dotenv pandas tqdm
pip install flash-attn --no-build-isolation
```

### 3. Running Execution via `tmux`

```bash
tmux
python sft-dpo-llama-runpod.py
```
* **Detach from session:** Press `Ctrl + B`, release, then press `D`.
* **Reattach to session:** Type `tmux attach`.

---

## Modular Scripts & Notebooks

All notebooks are designed to be ultra-lightweight entry points that delegate execution directly to clean, standalone Python files:

| Python Script | Notebook Runner | Purpose |
| :--- | :--- | :--- |
| `sft-dpo-llama-runpod.py` | `sft-dpo-llama-runpod.ipynb` | End-to-end multi-GPU SFT + DPO + Gemini evaluation pipeline on RunPod |
| `training_colabs/sft.py` | `training_colabs/sft.ipynb` | Supervised Fine-Tuning (SFT) prototype with manual masking |
| `training_colabs/dpo.py` | `training_colabs/dpo.ipynb` | Direct Preference Optimization (DPO) pairwise likelihood training |
| `training_colabs/rl_reward_model.py` | `training_colabs/rl-reward-model.ipynb` | Bradley-Terry reward model training with margin loss |
| `training_colabs/sft_dpo_llama.py` | `training_colabs/sft-dpo-llama.ipynb` | Multi-GPU SFT + DPO pipeline on Llama 3.2 |