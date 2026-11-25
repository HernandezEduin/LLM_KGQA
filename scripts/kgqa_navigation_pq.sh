#!/usr/bin/env bash

set -euo pipefail

# ============================================================
# Model profiles
# ============================================================

model_configs=(
    "configs/models/qwen3.json"
    "configs/models/gemma4.json"
    "configs/models/ministral-3-instruct-q4.json"
)

# ============================================================
# Datasets
# ============================================================

datasets=(
    "pq"
    "pql"
)

# ============================================================
# Graph / QA variants
# ============================================================

variants=(
    "filtered"
    "source"
)

declare -A variant_triplets
declare -A variant_qa_prefix

variant_triplets["filtered"]="filtered_triplets.txt"
variant_qa_prefix["filtered"]="filtered"

variant_triplets["source"]="source_triplets.txt"
variant_qa_prefix["source"]="source"

# ============================================================
# Dataset settings
# ============================================================

declare -A dataset_max_steps
declare -A dataset_max_actions
declare -A dataset_context_window

dataset_max_steps["pq"]=3
dataset_max_actions["pq"]=200
dataset_context_window["pq"]=$((32 * 1024))

dataset_max_steps["pql"]=3
dataset_max_actions["pql"]=200
dataset_context_window["pql"]=$((32 * 1024))

# ============================================================
# Common experiment settings
# ============================================================

temperature=0
seed=42
timeout=15
max_parse_retries=0

run_navigation() {
    local dataset="$1"
    local variant="$2"
    local model_config="$3"

    echo
    echo "============================================================"
    echo "Dataset:          $dataset"
    echo "Variant:          $variant"
    echo "Triplets:         ${variant_triplets[$variant]}"
    echo "QA prefix:        ${variant_qa_prefix[$variant]}"
    echo "Model profile:    $model_config"
    echo "Prompting:        one-shot"
    echo "Structured:       true"
    echo "Max steps:        ${dataset_max_steps[$dataset]}"
    echo "Max actions:      ${dataset_max_actions[$dataset]}"
    echo "Context window:   ${dataset_context_window[$dataset]}"
    echo "Temperature:      $temperature"
    echo "Seed:             $seed"
    echo "============================================================"
    echo

    python ./kgqa_navigation.py \
        --model-config "$model_config" \
        --dataset "$dataset" \
        --hops n \
        --triplets-file "${variant_triplets[$variant]}" \
        --qa-file-prefix "${variant_qa_prefix[$variant]}" \
        --entity-id-col EID \
        --relation-id-col RID \
        --max-navigation-steps "${dataset_max_steps[$dataset]}" \
        --prompting-approach one-shot \
        --n-shots 1 \
        --demo-history-mode full \
        --demo-max-actions 5 \
        --max-actions "${dataset_max_actions[$dataset]}" \
        --context-window "${dataset_context_window[$dataset]}" \
        --navigation-approach tuple \
        --memory-approach full \
        --result-dir "./results/navigation/${variant}" \
        --seed "$seed" \
        --temperature "$temperature" \
        --timeout-cooldown 0 \
        --max-parse-retries "$max_parse_retries" \
        --structured-output \
        --timeout "$timeout"
}

echo "Running PQ/PQL KGQA Navigation Experiments"
echo
echo "### One-shot / Full Demo / Full Memory / Tuple / Structured ###"

for dataset in "${datasets[@]}"; do
    for variant in "${variants[@]}"; do
        for model_config in "${model_configs[@]}"; do
            run_navigation "$dataset" "$variant" "$model_config"
        done
    done
done
