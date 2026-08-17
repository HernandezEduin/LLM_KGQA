# LLMs KGQA Project

## Overview

This project runs Large Language Model (LLM) experiments for Knowledge Graph Question Answering (KGQA). It supports two task styles:

- **Subgraph QA**: provide the LLM with a sampled subgraph and ask for the final answer.
- **Iterative navigation QA**: let the LLM choose graph actions step by step while the controller validates and executes only legal KG edges.

The current runners are `kgqa_subgraph.py` and `kgqa_navigation.py`.

## Project Structure

- `configs/`: API/backend configuration files.
- `data/`: KGQA datasets.
- `model/`: shared base client and task-specific LLM clients.
- `results/`: JSON outputs from experiment runs.
- `utils/`: shared typing, graph, metric, KGQA, and API utilities.
- `scripts/`: helper scripts for experiments and sanity checks.

## Key Files

- `kgqa_navigation.py`: iterative graph-navigation KGQA runner.
- `kgqa_subgraph.py`: subgraph-at-once KGQA runner.
- `model/base_llm_client.py`: shared LLM client logic.
- `model/navigation_llm_client.py`: navigation-specific prompt, parsing, and control logic.
- `model/subgraph_llm_client.py`: subgraph-specific prompt and prediction logic.
- `utils/kgqa_utils.py`: shared KGQA helpers, including optional title-map loading.
- `utils/kgqa_types.py`: shared KGQA type aliases.
- `utils/kgqa_navigation_metrics.py`: navigation answer/path metrics.

## Installation

This project requires Python 3.12 or higher.

```bash
pip install -r requirements.txt
```

The scripts expect an API configuration in `configs/openwebui_config.json`.

## Data Layout

Each dataset should live under `data/<dataset_name>/` and include:

- `triplets.txt`
- `qa_<hops>hop.csv`, for example `qa_nhop.csv`

Encoded datasets such as MQuAKE may also include:

- `node_data.csv`
- `relation_data.csv`

These mapping files are optional. If they are missing, as in unencoded datasets such as `kinship_v2`, the runners assume entity and relation strings are already readable and omit title mappings from prompts.

## Usage

### Iterative Navigation QA

```bash
python ./kgqa_navigation.py \
  --dataset mquake_single \
  --hops n \
  --llm-model qwen2.5 \
  --use-instruct \
  --navigation-approach tuple \
  --memory-approach full \
  --prompting-approach zero-shot \
  --max-navigation-steps 4 \
  --max-actions 200 \
  --result-dir ./results
```

Navigation modes:

- `tuple`: the LLM chooses directly from full outgoing edge actions.
- `factorized`: the LLM chooses a relation first, then a destination entity.
- `hybrid`: uses tuple mode for small neighborhoods and factorized mode for larger ones.

`--max-actions` caps the number of options shown in each prompt. If a node has more than `N` sorted options, only the first `N` are shown to the LLM. This does not terminate the episode by itself. Result JSON records this with `max_actions_truncated` and `max_actions_truncations`.

The prompt is still checked against `--context-window`; if the truncated prompt is too large, the episode terminates with `context_window_exceeded` before that LLM call.

Use `--show-navigation` or `--show-actions` to print each prompt, model response, validated move, and termination reason.

### Subgraph QA

```bash
python ./kgqa_subgraph.py \
  --dataset mquake_single \
  --hops n \
  --llm-model qwen2.5 \
  --use-instruct \
  --sampling-method neighborhood \
  --subgraph-size 50 \
  --max-depth 3 \
  --result-dir ./results
```

Subgraph sampling modes:

- `neighborhood`: expand around evidence or source nodes.
- `random`: sample graph triplets while preserving required evidence seeds.
- `evidence`: use only evidence paths.

Use `-r` / `--retrieve` for non-oracle retrieval from the source node.

## Results

Results are saved under `results/<dataset>/`.

Navigation outputs include:

- run configuration
- aggregate statistics
- per-question episode records
- selected actions and readable executed paths
- path-fidelity and final-entity metrics
- title-mapping status
- truncation/context-window metadata

Subgraph outputs include:

- run configuration
- aggregate statistics
- title-mapping status

## Notes

- The graph controller only executes legal KG edges listed in the prompt.
- Navigation uses the terminal graph entity as the prediction.
- `--max-actions` is deterministic because actions are sorted before truncation.
- No answer-type hints are added to prompts.

## License

This project is licensed under the Academic License.

## TODO

- [ ] Double-check all LLMs use deterministic subgraph prompts when seeded.
- [x] Set seed for LLM via backend.
- [ ] Add an option for bulk/batch chat functionality.
- [x] Enhance chat timeout and retry mechanisms to handle longer subgraphs.
- [ ] Add functionality that converts triplets into sentences.
- [x] Add prompt token estimation.
- [x] Add context-window configuration.
- [ ] Add zero-context QA evaluation without graph context.
- [ ] Manually revise path-fidelity metrics.
- [ ] Ensure PED and F1_SG accept multiple answers for a single question.
- [x] Support more general datasets with optional entity/relation title mappings.
- [x] Create a parent class and child clients for subgraph and iterative navigation.
- [ ] Include a human navigation GUI for manually navigating the knowledge graph.
- [ ] Double-check that LLM cancellation still works.
