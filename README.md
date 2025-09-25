# SQuADGPT2CPT

Concise experiments fine-tuning GPT-2 for extractive QA on SQuAD v1.1 with LoRA, dynamic quantization, and cyclic precision training. The repo is notebook-driven and aims to be easy to run and reproduce.

## Contents
- `gpt2-lora-question-answering.ipynb` — Baseline GPT‑2 + LoRA fine‑tuning on SQuAD v1.1
- `gpt2_dynamic_quantization_lora_clean_ranset.ipynb` — LoRA fine‑tuning with dynamic quantization (clean, seeded)
- `gpt2_cyclic_precision_training.ipynb` — Cyclic precision training experiments
- `gpt2_cyclic_precision_training_ranset.ipynb` — Cyclic precision training (seeded)
- `train-v1.1.json`, `dev-v1.1.json` — SQuAD v1.1 dataset files
- `Question.pdf` — Problem/experiment description (reference)
- `LICENSE` — MIT License

## Highlights
- LoRA adapters to reduce trainable parameters
- PyTorch dynamic quantization for faster/lighter inference on CPU
- Cyclic precision training experiments (e.g., alternating FP32/FP16 or quantization phases)
- Reproducible runs with fixed random seeds in the "ranset" notebooks

## Environment
- Python 3.9–3.11
- Recommended: virtual environment

Install core dependencies (minimal set):

```bash
pip install torch transformers datasets peft accelerate jupyter scikit-learn evaluate
```

Notes:
- GPU is optional; quantized inference is CPU‑friendly. If you have CUDA, install a CUDA-enabled `torch` per PyTorch docs.
- `bitsandbytes` is not required for these notebooks.

## Dataset
The repo includes SQuAD v1.1 JSON files:
- `train-v1.1.json`
- `dev-v1.1.json`

Most notebooks load these directly. You can also load SQuAD via the `datasets` library if preferred.

## How to Run
1. Create and activate a virtual environment (optional but recommended).
2. Install dependencies (see above).
3. Start Jupyter:
   ```bash
   jupyter notebook
   ```
4. Open a notebook based on your goal:
   - For a baseline: open `gpt2-lora-question-answering.ipynb`.
   - For quantized training/inference: open `gpt2_dynamic_quantization_lora_clean_ranset.ipynb`.
   - For precision cycling experiments: open `gpt2_cyclic_precision_training*.ipynb`.

Each notebook contains step‑by‑step cells: data prep, model setup, training, evaluation, and (where relevant) quantized inference.

## Reproducibility
- The `*ranset` notebooks explicitly set seeds for Python, NumPy, and PyTorch.
- For full determinism on GPU, you may also enable cuDNN deterministic mode inside the notebooks (at a performance cost).

## Expected Results
Metrics depend on hardware and hyperparameters. As a reference target for quick runs:
- Baseline GPT‑2 + LoRA on SQuAD v1.1: EM/F1 should improve over a frozen GPT‑2 baseline within a few epochs.
- Dynamic quantization: similar EM/F1 with reduced CPU latency and model size for inference.

You can log exact EM/F1 from the evaluation cells in each notebook.

## Project Status
This is a compact research/teaching setup, focused on clarity over engineering. Feel free to extend with scripts if you prefer CLI workflows.

## Citation
If you use this work, please cite the underlying datasets and methods:
- Rajpurkar et al., SQuAD: `https://arxiv.org/abs/1606.05250`
- Radford et al., GPT‑2: `https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf`
- Hu et al., LoRA: `https://arxiv.org/abs/2106.09685`
- Hugging Face Transformers: `https://github.com/huggingface/transformers`

## License
MIT — see `LICENSE`.
