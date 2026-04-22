# LLMs Project

## Overview
This project focuses on leveraging Large Language Models (LLMs) for Knowledge Graph Question Answering (KGQA) on a local server. It includes scripts and utilities for preprocessing data, running experiments, and analyzing results across different datasets and LLM models.

In addition to standard prompting pipelines, the repository includes **offline graph-native retrieval methods** inspired by SG-RAG and PathRAG. These methods operate directly over a local symbolic Knowledge Graph (KG) and are designed for fully offline use.

Rather than retrieving external documents, they use:

- encoded KG triplets `(head, relation, tail)`
- entity metadata (label, description, aliases)
- relation metadata (label, description, aliases)

The metadata acts as the **language bridge** between natural-language questions and symbolic graph IDs, while the KG remains the primary retrieval substrate.

## Project Structure

- **configs/**: Contains configuration files for the project.
- **data/**: Placeholder folder for KGQA datasets.
- **model/**: Contains the main implementation of the LLM KGQA model.
- **results/**: Stores the results of experiments in JSON format.
- **utils/**: Utility scripts for API interactions, graph processing, and more.
- **scripts/**: Bash scripts for running experiments and sanity checks.

## Key Files

- `llm_kgqa.py`: Main script for running KGQA experiments.
- `preprocess.py`: Script for preprocessing datasets.
- `utils/sg_rag.py`: Offline SG-RAG-inspired retriever.
- `utils/path_rag.py`: Offline PathRAG-inspired retriever.

## Installation

This project requires Python 3.12 or higher.

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd LLMs
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

## Oracle (Ground Truth) Subgraph Prompting

To run an experiment, use the `llm_kgqa.py` script. For example:
```bash
python ./llm_kgqa.py \
   --dataset mquake_single \
   --hops n \
   --llm-model qwen2.5 \
   --use-instruct \
   -e
```

## Retrieval Methods

### Offline SG-RAG

```bash
python llm_kgqa.py \
  --dataset mquake_single \
  --hops n \
  --sampling-method sg_rag \
  --rag-max-hop 4 \
  --rag-top-contexts 10 \
  --sg-top-query-patterns 5 \
  --llm-model qwen2.5 \
  --use-instruct
```

### Offline PathRAG

```bash
python llm_kgqa.py \
  --dataset mquake_single \
  --hops n \
  --sampling-method path_rag \
  --rag-max-hop 4 \
  --rag-top-contexts 8 \
  --path-top-nodes 8 \
  --path-alpha 0.8 \
  --path-threshold 0.3 \
  --llm-model qwen2.5 \
  --use-instruct
```

## Important Clarification

These implementations are **offline adaptations** of SG-RAG and PathRAG for symbolic local KGs. They are not exact replications of the original papers.

## Results
Results are stored in the `results/` directory. Each result file is named based on the dataset, model, and experiment parameters.

## License
This project is licensed under the Academic License.

## TODO:
- [ ] !Double Check all LLMs are using the same sub graph in the prompt (randomness must be deterministic)
- [x] Set Seed for LLM via backend.
- [ ] Add an option for bulk/batch chat functionality.
- [x] Enhance chat timeout and retry mechanisms to handle longer subgraphs effectively.
- [ ] Add a functionality that converts the triplets of the subgraph into a sentence instead.
- [x] Add a feature to calculate the number of tokens in the prompt.
- [x] Add a feature to increment the context window of the llm (num_ctx).
- [ ] Add an option to just evalute the LLM on the question without any context (zero-shot), just the question and the answer choices (if any).
