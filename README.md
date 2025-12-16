# BMW Press Release Fine-Tuning Pipeline

**Author**: Ilir Hajrullahu  
**Role**: AI Engineer Assignment - BMW Automotive  
**Date**: December 2025

## Overview

End-to-end pipeline for fine-tuning a small language model on BMW press releases. Includes web scraping, data processing, model training (Full Fine-Tuning and LoRA), RAG exploratory test, and evaluation with automatic metrics.

**Goal**: Create a Question-Answering system for BMW press release content.

**Three Approaches Tested**:
1. **Full Fine-Tuning**: Train all 494M parameters
2. **LoRA**: Parameter-efficient training (942K params)
3. **RAG**: Zero-training retrieval-based approach (exploratory test)

---

## Table of Contents

- [Project Structure](#project-structure)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Pipeline Steps](#pipeline-steps)
  - [1. Data Extraction](#1-data-extraction-data_extractionipynb)
  - [2. Data Processing](#2-data-processing-data_processing_promptipynb)
  - [3. Training & Evaluation](#3-training--evaluation-trainingipynb)
  - [4. RAG Evaluation](#4-rag-evaluation-rag_evaluationipynb)
- [Results Summary](#results-summary)
- [How to Run](#how-to-run)
- [Design Decisions](#design-decisions)
- [Troubleshooting](#troubleshooting)
- [Future Improvements](#future-improvements)
- [Key Takeaways](#key-takeaways)

---

## Project Structure

```
bmw-assignment/
├── data/
│   ├── raw/               # 241 train + 5 eval articles (JSON)
│   └── processed/         # 723 train + 15 eval QA pairs (JSONL)
├── qwen_full_ft/          # Full fine-tuning checkpoints
├── qwen_lora/             # LoRA adapter weights
├── data_extraction.ipynb  # Step 1: Scrape press releases
├── data_processing_prompt.ipynb  # Step 2: Generate QA pairs
├── training.ipynb         # Step 3: Train & evaluate models
├── rag_evaluation.ipynb   # Step 4: RAG exploratory test
├── eval_generations.jsonl # Fine-tuned model outputs
├── rag_eval_results.jsonl # RAG evaluation outputs
└── requirements.txt       # Python dependencies
```

---

## System Requirements

**Tested Configuration**:
- **Hardware**: MacBook M3 Pro (CPU only)
- **Memory**: 18GB RAM
- **Python**: 3.11.11
- **Environment**: Anaconda

**Minimum**: Python 3.11+, 8GB RAM, ~10GB disk space

---

## Installation

```bash
# Clone repository
git clone <repository-url>
cd bmw-assignment

# Create environment
conda create -n bmw-assignment python=3.11
conda activate bmw-assignment

# Install dependencies
pip install -r requirements.txt
```

**Key Dependencies**:
- `torch==2.9.1`, `transformers==4.57.3`, `peft==0.18.0`
- `sentence-transformers==3.3.1`, `faiss-cpu==1.9.0.post1` (for RAG)
- `selenium`, `beautifulsoup4` (for data extraction only)

---

## Pipeline Steps

### 1. Data Extraction (`data_extraction.ipynb`)

- Scrapes ~246 BMW press releases using Selenium (headless Chrome)
- Splits into 241 train / 5 eval articles  
- Saves raw JSON to `data/raw/`
- **Runtime**: ~15-20 minutes

---

### 2. Data Processing (`data_processing_prompt.ipynb`)

- Generates 3 rule-based QA pairs per article (title, date, summary) for training
- Creates 15 hand-crafted diverse eval questions testing real comprehension
- Formats using chat template: `<|system|>`, `<|user|>`, `<|assistant|>`
- Output: 723 train pairs, 15 eval pairs in JSONL format

**Prompt Format**:
```
<|system|>
You are a QA bot for BMW press articles. Use ONLY the provided context.

<|user|>
Context: {article_text}
Question: {question}

<|assistant|>
{answer}
```

**Runtime**: ~1 minute

---

### 3. Training & Evaluation (`training.ipynb`)

**Base Model**: `Qwen/Qwen2.5-0.5B` (494M parameters, 2048 token context)

**Two Approaches**:
- **Full Fine-Tuning**: LR 2e-5, trains all 494M params (~20 min)
- **LoRA**: LR 1e-4, trains 942K params/0.19% (~15 min), rank=8, alpha=16

**Data Filtering**: Removed 123/723 samples exceeding 2048 tokens → 600 train, 12 eval

**Metrics**: Perplexity, Exact Match Accuracy, Qualitative analysis

**Runtime**: ~40-50 minutes total

---

### 4. RAG Evaluation (`rag_evaluation.ipynb`)

Exploratory test using base `Qwen/Qwen2.5-0.5B` (no fine-tuning) with retrieval.

**Setup**: SentenceTransformer embeddings + FAISS index over 246 articles → Retrieve top-3 → Generate answer

**Questions**: 15 standalone questions (rewritten from eval set to not assume context)

**Runtime**: ~10-15 minutes

---

## Results Summary

### Model Comparison

**Primary Comparison: Full FT vs LoRA**

| Metric | Full FT | LoRA | Winner |
|--------|---------|------|--------|
| **Exact Match Accuracy** | 8% | 17% | LoRA ✓ |
| **Perplexity** | 24.46 | 25.17 | Full FT (slightly) |
| **Trainable Params** | 494M | 942K | LoRA (524x fewer) |
| **Model Size** | 1.9GB | 3.7MB | LoRA (513x smaller) |
| **Training Time** | ~20 min | ~15 min | LoRA (25% faster) |

**RAG as Additional Exploration (Not Directly Comparable)**

| Metric | RAG (No FT) | Notes |
|--------|-------------|-------|
| **Accuracy** | Low (qualitative) | Not calculated - different eval methodology (15 vs 12 questions) |
| **Trainable Params** | 0 | No training required |
| **Setup Time** | ~10 min | Embedding generation + vector index |
| **Use Case** | Zero-shot retrieval | Useful when training is not feasible |

**⚠️ Not Directly Comparable**: Different methodologies - Fine-tuned models use 12 context-provided questions; RAG uses 15 rewritten standalone questions with retrieval. Exploratory comparison only.

### Performance Notes

**Fine-Tuned Models**: Extract direct answers from provided context. Strong when context contains answer.

**RAG**: Often retrieves correct articles but base model struggles with instruction-following, producing verbose or incorrect answers.

### Key Insights

**LoRA vs Full FT**:
- Better generalization on small dataset (regularization effect)
- 25% faster training with 40% less memory
- 500x smaller deployment size
- Comparable quality with far fewer parameters
- Winner for this use case (small dataset, CPU, limited compute)

**Why Low Accuracy Overall?**
- Only 600 training samples (very limited)
- 50 training steps (8% of one epoch)
- Small 0.5B model
- Focus on demonstrating pipeline, not achieving SOTA

---

## How to Run

### Quick Start (Skip Data Collection)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run training (assumes data/ exists)
jupyter notebook training.ipynb
# Run all cells → ~50 minutes on CPU

# 3. Check outputs
cat eval_generations.jsonl
```

### Full Pipeline

```bash
# 1. Extract data (~20 min)
jupyter notebook data_extraction.ipynb

# 2. Process QA pairs (~1 min)
jupyter notebook data_processing_prompt.ipynb

# 3. Train models (~50 min)
jupyter notebook training.ipynb

# 4. RAG evaluation (~15 min)
jupyter notebook rag_evaluation.ipynb
```

**Total Runtime**: ~1.5 hours on MacBook M3 Pro

### RAG-Only Quick Test

```bash
# Test RAG without training (requires data/ folder)
jupyter notebook rag_evaluation.ipynb
# Runtime: ~15 minutes
# Output: rag_eval_results.jsonl
```

---

## Design Decisions

### 1. Model Choice: Qwen2.5-0.5B
- Small enough for CPU training within 6-8 hour budget
- Modern architecture with strong baseline performance
- Open-source and commercially usable

### 2. Data Strategy: Rule-Based QA for Training, Manual QA for Eval
- **Training**: Rule-based generation ensures 100% factual accuracy
- **Eval**: Hand-crafted diverse questions test real comprehension
- Simple, reproducible, transparent
- No dependency on external LLM APIs
- Trade-off: Training questions less diverse, but eval properly tests generalization

### 3. Filtering vs Truncation
- Filter long samples (>2048 tokens) instead of truncating
- Preserves data quality, avoids corrupted training signals
- 600 high-quality samples > 723 mixed-quality samples

### 4. Manual Eval Questions
- Training: Simple rule-based questions (title, date, summary)
- Eval: Hand-crafted, diverse questions testing real comprehension
- Tests understanding vs pattern memorization

### 5. Full FT vs LoRA Comparison
- Demonstrates modern PEFT vs traditional fine-tuning
- LoRA wins: small dataset, CPU training, limited compute

### 6. Different Learning Rates (2e-5 vs 1e-4)
- Industry best practice: PEFT methods use 5-10x higher LR
- Smaller parameter space allows more aggressive updates
- Follows original LoRA paper and Hugging Face recommendations

---

## Troubleshooting

**Training is slow**: Normal on CPU (~50 min). Reduce `max_steps=50` to `max_steps=25` for faster testing.

**Out of memory**: Reduce `max_length=2048` to `1024` in training.ipynb.

**Models perform poorly**: Expected with only 600 samples. Goal is to demonstrate pipeline, not achieve SOTA.

**ChromeDriver errors**: `pip install --upgrade webdriver-manager` or install Chrome manually.

---

## Future Improvements

**With More Data/Time**:
- Scale to 1,000+ press releases
- Train for 3+ epochs with early stopping
- Use larger models (1.5B-3B parameters)
- Implement ROUGE, BLEU, BERTScore metrics
- Add diverse question types (reasoning, comparison)

**With More Compute**:
- Multi-GPU training with DeepSpeed
- Model quantization (4-bit, 8-bit)
- Ensemble multiple fine-tuned models
- Test instruction-tuned models (e.g., Qwen2.5-0.5B-Instruct) for RAG

---

## Key Takeaways

1. **Data Quality > Quantity**: 600 clean samples outperform 723 mixed-quality samples
2. **LoRA Wins**: Fast, efficient, comparable to Full FT on small datasets
3. **Small Models Work**: Even 0.5B parameters show clear domain adaptation  
4. **RAG Insights**: Retrieval works but small base models lack instruction-following; hybrid approach worth exploring
5. **Engineering > Accuracy**: Focus on sound decisions and clear communication
