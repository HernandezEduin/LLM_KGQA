# LLMs Project

## Overview
This project focuses on leveraging Large Language Models (LLMs) for Knowledge Graph Question Answering (KGQA). It includes scripts and utilities for preprocessing data, running experiments, and analyzing results across different datasets and LLM models.

## Project Structure

- **configs/**: Contains configuration files for the project.
- **data/**: Placeholder folder for KGQA datasets.
- **model/**: Contains the main implementation of the LLM KGQA model.
- **results/**: Stores the results of experiments in JSON format.
- **utils/**: Utility scripts for API interactions, graph processing, and more.
- **scripts**: Bash scripts for running specific experiments and sanity checks.

## Key Files

- `llm_kgqa.py`: Main script for running KGQA experiments.
- `preprocess.py`: Script for preprocessing datasets.

## Installation

This project requires Python 3.12 or higher. Ensure you have the correct version installed before proceeding.

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

### Running Experiments

To run an experiment, use the `llm_kgqa.py` script. For example:
```bash
python ./llm_kgqa.py --dataset kinship --hops n --llm-model gpt-oss -e
```

### Running Offline SG-RAG

The repo now includes an offline SG-RAG path for multi-hop KGQA. It works directly from:

- `triplets.txt` for the encoded KG edges (`QID<TAB>PID<TAB>QID`)
- `node_data.csv` for entity labels/descriptions
- `relation_data.csv` for relation labels/descriptions

The retriever does not use oracle `Path`, `Path-Key`, or per-question hop annotations at inference time.
Instead, it:

- infers candidate graph query patterns from the question and KG metadata
- executes those patterns on the offline KG
- returns the matched records as grouped subgraphs for the LLM

This is closer to SG-RAG's original "query -> matched subgraphs -> grouped triplets" flow, without the later MOT/MOB merge-and-order extension.

Example with the reference dataset in `data/mquake_single`:

```bash
conda run -n llms python llm_kgqa.py \
  --dataset mquake_single \
  --hops n \
  --sampling-method sg_rag \
  --sg-max-hop 5 \
  --sg-top-query-patterns 1 \
  --sg-top-subgraphs 5 \
  --sg-include-descriptions \
  --llm-model qwen2.5 \
  --use-instruct
```

Key SG-RAG flags:

- `--sampling-method sg_rag`: Enables the offline SG-RAG retriever.
- `--sg-max-hop <int>`: Sets the largest hop count SG-RAG will consider while inferring a path length without oracle hop access.
- `--sg-top-query-patterns <int>`: Limits how many candidate query patterns are considered.
- `--sg-top-subgraphs <int>`: Limits how many matched subgraph records are sent to the LLM.
- `--sg-beam-width <int>` and `--sg-max-branching <int>`: Control graph traversal width.
- `--sg-include-descriptions`: Appends entity/relation descriptions to the prompt as compact hints.

### Running Offline PathRAG

The repo also includes an offline PathRAG-style retriever for multi-hop KGQA. It follows the core PathRAG idea:

- retrieve query-relevant nodes from KG metadata
- find key relational paths between those nodes
- prune and rank paths with a flow-based reliability score
- place retrieved paths in the prompt from lower to higher reliability, with the best paths last

Because this repo uses an offline symbolic KG rather than a text-indexed graph with embeddings, the node retrieval stage is approximated with overlap over entity and relation labels/descriptions.

Example:

```bash
conda run -n llms python llm_kgqa.py \
  --dataset mquake_single \
  --hops n \
  --sampling-method path_rag \
  --path-max-hop 4 \
  --path-top-nodes 8 \
  --path-top-paths 8 \
  --path-alpha 0.8 \
  --path-threshold 0.3 \
  --path-include-descriptions \
  --llm-model qwen2.5 \
  --use-instruct
```

Key PathRAG flags:

- `--sampling-method path_rag`: Enables the offline PathRAG retriever.
- `--path-top-nodes <int>`: Limits how many query-relevant nodes are retained before path retrieval.
- `--path-top-paths <int>`: Limits how many relational paths are sent to the LLM.
- `--path-max-hop <int>`: Sets the maximum path length explored.
- `--path-alpha <float>` and `--path-threshold <float>`: Control the flow-based pruning stage.
- `--path-max-paths-per-pair <int>` and `--path-max-branching <int>`: Bound candidate path enumeration for offline use.
- `--path-include-descriptions`: Appends entity/relation descriptions to the prompt as compact hints.

## Results
Results are stored in the `results/` directory. Each result file is named based on the dataset, model, and experiment parameters.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request.

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
