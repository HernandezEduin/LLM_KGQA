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
- [ ] Manually revise the Path-Fidelity Metrics.
- [ ] Ensure the PED and F1_SG accept multiple answers for a single question. Reconstruct the path on-the-fly to calculate the metrics.
- [ ] Allow compatibility with other datasets like Kinship, MetaQA, PathQuestion, and more. This includes adding support for different graph structures and question formats.
- [ ] Create a parent class for LLM_KGQA and child classes for each task: subgraph and iterative navigation.
