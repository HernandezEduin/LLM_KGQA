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
  --n-shots 0 \
  --demo-history-mode full \
  --demo-max-actions 10 \
  --max-navigation-steps 4 \
  --max-actions 200 \
  --result-dir ./results
```

Navigation modes:

- `tuple`: the LLM chooses directly from full outgoing edge actions.
- `factorized`: the LLM chooses a relation first, then receives the normal action prompt limited to only that relation's edges.
- `hybrid`: uses tuple mode for small neighborhoods and the same two-stage factorized flow for larger ones.

`--n-shots` prepends complete solved train trajectories to action-selection prompts. Demonstration sampling is seed-reproducible and prefers longer train trajectories so later hops show gold path history. One shot is one complete trajectory: the question and start entity are shown once, then each hop shows the current entity, selected history view, available actions, and gold JSON selected action. `--n-shots 0` preserves the existing zero-shot behavior. `--prompting-approach one-shot` is a convenience alias for `--n-shots 1` when no explicit shot count is provided.

`--demo-history-mode` controls demonstration history independently of test-time navigation memory. Use `full` for all previous gold hops, `last` for only the immediately previous hop, or `random` for one seeded random previous hop. Hops with no previous edge show `(none)`.

`--demo-max-actions` caps the number of legal actions shown in each demonstrated hop. The demonstrated action list always contains the gold next edge, with other options filled from the remaining sorted neighborhood, and the gold `{"action": ...}` ID is recomputed after truncation. This cap is independent from `--max-actions` and does not change test-time navigation logic.

`--max-actions` caps the number of options shown in each inference prompt. If a node has more than `N` sorted options, only the first `N` are shown to the LLM. This does not terminate the episode by itself. Result JSON records this with `max_actions_truncated` and `max_actions_truncations`.

The prompt is still checked against `--context-window`; if the prompt, including demonstrations, is too large, the episode terminates with `context_window_exceeded` before that LLM call.

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

### Navigation

- [ ] Implement `--prompting-approach io` and keep the prompt-template interface extensible for future prompt families.
- [ ] Add graph directionality options (`outgoing`, `incoming`, `bidirectional`) and propagate the setting through action indexing, path validation, metrics, and result config.
- [ ] Add configurable `--max-actions` selection policies, such as `first`, seeded random sampling, and question-aware ranking.
- [ ] Record both prompt-local option IDs and original sorted graph-action IDs in episode records, especially for truncated tuple prompts and factorized relation-action prompts.
- [ ] Add adaptive context-window handling before termination, such as reducing shown actions or switching from tuple to factorized prompts when possible.
- [ ] Build a human navigation GUI for manually stepping through the graph and answering questions.
- [x] Reuse the standard action prompt for factorized second-stage navigation over the selected relation's edges.
- [x] Add n-shot navigation demonstrations from complete train-set gold trajectories.

### Prompting And LLM Calls

- [ ] Add shared one-shot and few-shot prompt templates for subgraph QA.
- [ ] Add zero-context QA evaluation without graph context.
- [ ] Add an optional triplet-to-sentence prompt format for subgraph evidence.
- [ ] Add bulk/batch chat execution where supported by the backend.
- [ ] Verify LLM cancellation and model unload behavior across normal completion, timeout, and interruption.
- [x] Set LLM seed through the backend.
- [x] Add timeout and retry handling for long LLM calls.
- [x] Add prompt token estimation and context-window configuration.

### Metrics And Evaluation

- [ ] Revisit path-fidelity metrics and document the intended PED, RED, F1_SG, and F1_REL behavior.
- [ ] Ensure PED and F1_SG correctly support multiple gold answers and multiple valid evidence paths.
- [ ] Add regression tests for navigation termination reasons, parse retries, max-action truncation, and context-window failures.
- [ ] Double-check deterministic subgraph prompts across models when seeds are fixed.

### Data And Architecture

- [ ] Add dataset compatibility checks for required columns and optional fields before launching long runs.
- [ ] Add support notes or adapters for additional KGQA datasets such as MetaQA and PathQuestion.
- [x] Support datasets with optional entity/relation title mappings.
- [x] Split shared LLM client logic from subgraph and navigation task clients.
