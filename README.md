# Aspect-Based Sentiment Analysis: DeBERTa-v3 vs. Llama 3.1 8B

A benchmark comparing a task-specific fine-tuned Transformer against a local zero-shot LLM (Large Language Model) for Aspect Category Sentiment Classification (ACSC) on real-world e-commerce product reviews.

## Overview

This project evaluates sentiment classification performance across fine-grained product aspects (such as cleansing performance, moisturizing effect, skin reaction, and product value) using an Oracle setup where ground truth aspect categories are provided directly to isolate sentiment classification from aspect extraction.

```
Raw Review Text ──> spaCy Sentencizer ──> Gold Aspect Pairs ──┬──> DeBERTa-v3 Large ABSA (Batched PyTorch)
                                                              └──> Llama 3.1 8B Instruct (llama-cpp GGUF)
                                                                                     │
                                                                                     v
                                                                  Comparative Metrics & Error Analysis
```

## Models Evaluated

| Model | Architecture | Paradigm | Deployment |
| :--- | :--- | :--- | :--- |
| **DeBERTa-v3 Large ABSA** (`yangheng/deberta-v3-large-absa-v1.1`) | Encoder (435M params) | Fine-tuned specialist | Hugging Face Transformers / PyTorch |
| **Llama 3.1 8B Instruct** (Q4_K_M GGUF) | Decoder (8B params) | Zero-shot generalist | Local runtime via `llama-cpp-python` (CUDA) |

## Benchmark Results

Evaluated on 162 annotated (sentence, aspect, sentiment) triplets across 198 review sentences.

| Model | Accuracy | Macro F1 | Negative F1 | Neutral F1 | Positive F1 | Throughput |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **DeBERTa-v3 Large** | 75.3% | 0.610 | 0.74 | 0.26 | 0.83 | ~107 inf/sec |
| **Llama 3.1 8B** | **92.6%** | **0.840** | **0.95** | **0.61** | **0.95** | ~1-3 inf/sec |

### Key Findings
- **Generalist contextual reasoning**: Zero-shot Llama 3.1 8B resolved subtle aspect-specific negation, mixed sentiments within the same sentence, and neutral sentiments with significantly higher accuracy (+0.23 Macro F1 over DeBERTa).
- **Specialist throughput advantage**: DeBERTa achieved over 35x higher inference throughput on local GPU (Graphics Processing Unit, RTX 3070), making it superior in high-volume pipelines where lower latency and compute cost outweigh accuracy.

## Project Structure

```
├── code/
│   ├── absa_acsc_benchmark.ipynb   # Complete data loading, inference, and evaluation pipeline
│   ├── oracle_deberta_results.csv  # Detailed DeBERTa predictions and confidence scores
│   ├── oracle_llm_results.csv      # Detailed Llama 3.1 predictions
│   └── oracle_llm_results.jsonl    # Raw LLM inference logs
├── data/
│   ├── aspectsv2.jsonl             # Aspect definitions and taxonomy
│   ├── cleanser_sentences.jsonl    # Segmented review sentences
│   ├── gold_set_cleanser_version_2.jsonl # Ground truth aspect-sentiment annotations
│   ├── raw_reviews_cleanser.jsonl  # Raw cleanser review text
│   └── raw_reviews_moisturizer.jsonl# Raw moisturizer review text
└── requirements/
    ├── requirements.txt            # Python dependencies
    └── Labrequirements.txt         # Course/lab baseline dependencies
```

## Quickstart

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/joehu131/TextMiningProject.git
cd TextMiningProject

# Create and activate virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements/requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Run the Benchmark

Open and run `code/absa_acsc_benchmark.ipynb` in Jupyter Notebook or VS Code. Ensure a GPU runtime is selected if running local GGUF (GPT-Generated Unified Format) inference for Llama 3.1.
