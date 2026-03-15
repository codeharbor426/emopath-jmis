# EmoPath: Emotion-Aware Customer Complaint Response Framework

EmoPath is a modular framework for detecting emotional signals in customer complaints and generating response strategies based on emotion-aware planning.

The system combines lexicon-based emotion detection, optional emotion composition, strategy planning, and LLM-based response generation.  
It is designed for research experiments evaluating how emotional signals influence response strategies in customer service contexts.

---

# Project Overview

EmoPath processes a complaint through a multi-stage pipeline:

1. **Emotion Detection**  
   Detect emotional signals using emotion dictionaries.

2. **Emotion Composition**  
   Identify dominant and secondary emotions.

3. **Strategy Planning**  
   Select response strategies based on emotional context.

4. **Response Generation**  
   Generate responses using a language model.

The pipeline can operate under multiple experimental conditions that selectively disable components.

---

# Repository Structure

```
emopath-jmis/
├── data/
│   ├── samples/
│   │   ├── inputs_3a.jsonl
│   │   ├── gold_labels_3a.csv
│   │   ├── cv_folds.json
│   │   └── inputs_3b.jsonl
│   └── policies/
│       └── policy_docs_3b_structured.jsonl
│
├── emopath/
│   ├── detection/detector.py
│   ├── composition/composer.py
│   ├── planner/strategy.py
│   ├── generation/prompt_builder.py
│   ├── generation/generator.py
│   ├── generation/pvi_checker.py
│   ├── audit/logger.py
│   ├── pipeline.py
│   └── cli.py
│
├── scripts/                  # Experiment scripts
│
├── outputs/
│   ├── benchmark/
│   └── ablation/
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

### Directory Explanation

**data/**  
Contains datasets and policy documents used in experiments.

- `inputs_3a.jsonl` – complaint texts used in Study A  
- `gold_labels_3a.csv` – ground truth emotion labels  
- `cv_folds.json` – cross-validation splits  
- `inputs_3b.jsonl` – complaint texts used in Study B  
- `policy_docs_3b_structured.jsonl` – policy documents used in response generation

**emopath/**  
Core EmoPath framework implementation.

- `detection/` – emotion detection modules  
- `composition/` – emotion composition logic  
- `planner/` – strategy planning module  
- `generation/` – response generation and guardrail checks  
- `audit/` – optional logging utilities  
- `pipeline.py` – main orchestration logic  
- `cli.py` – command line interface

**scripts/**  
Scripts used to reproduce experiments and evaluation results.

**outputs/**  
Example outputs generated from benchmark and ablation experiments.

---

# Installation

Clone the repository:

```bash
git clone https://github.com/codeharbor426/emopath-jmis.git
cd emopath-jmis
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Create an environment file:

```bash
cp .env.example .env
```

Add your API key in `.env` if response generation requires an LLM.

---

# Docker Deployment

The EmoPath framework can also be executed using Docker for a fully reproducible environment.

## Build the Docker image

```bash
docker compose build
```

## Run the container

```bash
docker compose run emopath
```

This will start a container with all dependencies installed and the repository mounted inside the container.

You can then execute any EmoPath commands, for example:

```bash
python emopath/cli.py --text "I was charged twice for my purchase."
```

## Pulling the repository and running with Docker

```bash
git clone https://github.com/codeharbor426/emopath-jmis.git
cd emopath-jmis

docker compose build
docker compose run emopath
```

This ensures the environment matches the configuration used for the experiments.

---

# Running EmoPath via CLI

The EmoPath pipeline can be executed through the command line interface.

Example:

```bash
python emopath/cli.py --text "I was charged twice for my purchase and customer service has not helped."
```

Optional parameters:

```
--text        Customer complaint text (required)
--condition   Experimental condition (C1–C4)
--threshold   Emotion detection threshold (default: 0.5)
--show_prompt Include the rendered system/user prompts in the output JSON
```

Example with condition:

```bash
python emopath/cli.py \
  --text "I was charged twice for my purchase and nobody helped me." \
  --condition C1 \
  --threshold 0.5 \
  --show_prompt
```

### Notes on CLI usage

- `cli.py` supports **C1–C4** only.
- CLI mode is intended for **single-complaint inference and debugging**.
- `--show_prompt` is useful for inspecting the exact prompt sent to the language model.

---

# Example Output

Example output returned by the pipeline:

```json
{
  "response": "...",
  "emotion_vector": {
    "1_Anger": 9,
    "2_Frustration": 1,
    "3_Disappointment": 0,
    "4_Helplessness": 7,
    "5_Anxiety": 0
  },
  "dominant_emotion": "1_Anger",
  "secondary_emotions": [
    "4_Helplessness"
  ],
  "strategy_plan": {
    "stage1": "active listening and acknowledge injustice",
    "stage2": "offer compensation within policy limits"
  },
  "model_version": "gpt-4-0125-preview",
  "random_seed": 42,
  "emotion_markers": {
    "1_Anger": [
      "..."
    ],
    "2_Frustration": [
      "..."
    ],
    "4_Helplessness": [
      "..."
    ]
  },
  "rationale": "1_Anger dominant due to perceived injustice and external blame",
  "timestamp": "2026-03-14T12:35:49.587095",
  "prompt": {
    "system": "...",
    "user": "..."
  }
}
```

---

# Experimental Conditions

The EmoPath framework supports multiple experimental conditions used in ablation studies.

| Condition | Description |
|----------|-------------|
| C1 | Full EmoPath pipeline |
| C2 | Emotion composition disabled |
| C3 | Strategy planning disabled |
| C4 | Direct response generation |
| C5 | Fixed template baseline |

These configurations allow controlled evaluation of individual components.

### Condition Notes

- **C1–C4** are supported in `cli.py`.
- **C5** is a fixed template baseline designed for the formal experiment setting.
- **C5 is not exposed through `cli.py`**, because it requires pre-filled structured information such as company-specific fields and compensation-related policy details.
- C5 is intended for controlled Study B execution rather than ad hoc single-text CLI inference.

---

# Reproducing Study A

Study A evaluates the performance of different emotion detection approaches on customer complaint texts.

The benchmark compares lexicon-based methods, fine-tuned transformer models, and prompt-based LLM classification.

## Command

Run the benchmark using:

```bash
python -m scripts.run_benchmark \
 --data data/samples/inputs_3a.jsonl \
 --labels data/samples/gold_labels_3a.csv \
 --folds data/samples/cv_folds.json \
 --models lexicon bert roberta gpt4_few gpt4_zero \
 --out outputs/benchmark/
```

## Parameters

| Parameter | Description |
|----------|-------------|
| `--data` | Input complaint dataset used for emotion detection experiments |
| `--labels` | Ground truth emotion labels for evaluation |
| `--folds` | Cross-validation fold definitions |
| `--models` | List of models to evaluate |
| `--out` | Output directory for benchmark results |

## Model Definitions

Study A compares five emotion detection approaches:

| Model | Description |
|------|-------------|
| **EmoPath Lexicon** | Lexicon-based emotion detection using LIWC, NRC, and a custom-built emotion dictionary |
| **BERT multi-label** | `bert-base-uncased` fine-tuned with **BCEWithLogitsLoss** for multi-label emotion classification |
| **RoBERTa multi-label** | `roberta-base` fine-tuned with **BCEWithLogitsLoss** for multi-label emotion classification |
| **GPT-4 few-shot** | GPT-4 classification using prompts that include **5 labeled examples** |
| **GPT-4 zero-shot** | GPT-4 classification using **task instructions only** without examples |

## Outputs

Benchmark outputs will be written to:

```
outputs/benchmark/
```

However, **pre-generated benchmark outputs are already included in this repository** for reference and reproducibility.

These include:

- `benchmark_results.csv`
- `per_label_metrics.csv`
- `predictions_B*.csv`

Users may re-run the benchmark to regenerate results if desired.

---

# Reproducing Study B

Study B evaluates the full EmoPath response-generation pipeline under multiple experimental conditions.

The experiment compares different configurations of the pipeline to understand the contribution of emotion composition, sequential strategy planning, and policy-constrained response generation.

## Commands

Run the pipeline under each experimental condition:

```bash
python -m emopath.pipeline \
  --input data/samples/inputs_3b.jsonl \
  --out outputs/ablation \
  --condition C1 \
  --policy_doc data/policies/policy_docs_3b_structured.jsonl
```

```bash
python -m emopath.pipeline \
  --input data/samples/inputs_3b.jsonl \
  --out outputs/ablation \
  --condition C2 \
  --policy_doc data/policies/policy_docs_3b_structured.jsonl
```

```bash
python -m emopath.pipeline \
  --input data/samples/inputs_3b.jsonl \
  --out outputs/ablation \
  --condition C3 \
  --policy_doc data/policies/policy_docs_3b_structured.jsonl
```

```bash
python -m emopath.pipeline \
  --input data/samples/inputs_3b.jsonl \
  --out outputs/ablation \
  --condition C4 \
  --policy_doc data/policies/policy_docs_3b_structured.jsonl
```

```bash
python -m emopath.pipeline \
  --input data/samples/inputs_3b.jsonl \
  --out outputs/ablation \
  --condition C5 \
  --policy_doc data/policies/policy_docs_3b_structured.jsonl
```

## Parameters

| Parameter | Description |
|----------|-------------|
| `--input` | Input complaint dataset used in Study B |
| `--out` | Directory where ablation experiment outputs will be saved |
| `--condition` | Experimental configuration (C1–C5) |
| `--policy_doc` | Structured policy documents used for guardrail validation |

## Experimental Conditions

| Condition | Description |
|----------|-------------|
| **C1** | Full EmoPath pipeline |
| **C2** | Emotion composition disabled |
| **C3** | Sequential strategy planning disabled |
| **C4** | Unconstrained response generation |
| **C5** | Fixed template baseline |

## Outputs

Generated outputs will be written to:

```
outputs/ablation/
```

However, **the outputs used in the paper are already included in this repository** for reference and reproducibility.

These outputs contain the generated responses, emotion vectors, strategy plans, and guardrail evaluation results for each complaint instance.

---

# Outputs

Generated experiment outputs are stored in:

```
outputs/
├── benchmark/
└── ablation/
```

These include evaluation results, benchmark metrics, and ablation outputs produced during experiments.

---

# Notes

- `.env` files containing API keys should **never be committed**.
- Only `.env.example` is included to illustrate required environment variables.

---

# Acknowledgment

This repository provides the research artifact implementation for the EmoPath framework.
