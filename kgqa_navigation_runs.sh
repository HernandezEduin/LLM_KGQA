#!/usr/bin/env bash

set -euo pipefail

# ============================================================
# Models
# ============================================================

models=(
    "qwen3"
    "gemma4"
    "qwen2.5"
    "llama3.1"
    "granite3.3"
    "ministral-3"
    # "deepseek-r1"
    "olmo-3"
    "phi4-mini"
)

# These models should use their explicit instruct variants.
requires_instruct=(
    "qwen2.5"
    "llama3.1"
    "ministral-3"
    "olmo-3"
)

# These models should use the Q4 variant.
requires_quantized=(
    "qwen2.5"
    "llama3.1"
    "ministral-3"
    "olmo-3"
)

# ============================================================
# Datasets
# ============================================================

datasets=(
    # "kinship_v2"
    # "mquake_single"
    "mquake_multi"
    # "metaqa"
)

declare -A dataset_hops
declare -A dataset_max_steps
declare -A dataset_max_actions
declare -A dataset_context_window

dataset_hops["kinship_v2"]="n"
dataset_max_steps["kinship_v2"]=3
dataset_max_actions["kinship_v2"]=100
dataset_context_window["kinship_v2"]=$((8 * 1024))

dataset_hops["mquake_single"]="n"
dataset_max_steps["mquake_single"]=4
dataset_max_actions["mquake_single"]=200
dataset_context_window["mquake_single"]=$((32 * 1024))

dataset_hops["mquake_multi"]="n"
dataset_max_steps["mquake_multi"]=4
dataset_max_actions["mquake_multi"]=200
dataset_context_window["mquake_multi"]=$((32 * 1024))

dataset_hops["metaqa"]="n"
dataset_max_steps["metaqa"]=3
dataset_max_actions["metaqa"]=200
dataset_context_window["metaqa"]=$((32 * 1024))

# ============================================================
# Common experiment settings
# ============================================================

temperature=0
seed=42
timeout=15

# Use 0 for the structured-response benchmark so one logical
# decision always corresponds to one LLM generation.
max_parse_retries=0

# ============================================================
# Helpers
# ============================================================

contains_model() {
    local target="$1"
    shift

    local item
    for item in "$@"; do
        if [[ "$item" == "$target" ]]; then
            return 0
        fi
    done

    return 1
}


run_navigation() {
    local dataset="$1"
    local model="$2"
    local prompting="$3"
    local structured="$4"

    local model_flags=()
    local prompt_flags=()
    local output_flags=()

    # --------------------
    # Model variant
    # --------------------

    if contains_model "$model" "${requires_instruct[@]}"; then
        model_flags+=(--use-instruct)
    fi

    if contains_model "$model" "${requires_quantized[@]}"; then
        model_flags+=(--use-quantized --quantization-bits 4)
    fi

    # --------------------
    # Prompting
    # --------------------

    case "$prompting" in
        zero-shot)
            prompt_flags+=(
                --prompting-approach zero-shot
                --n-shots 0
            )
            ;;

        one-shot)
            prompt_flags+=(
                --prompting-approach one-shot
                --n-shots 1
                --demo-history-mode full
                --demo-max-actions 5
            )
            ;;

        *)
            echo "Unknown prompting mode: $prompting"
            return 1
            ;;
    esac

    # --------------------
    # Output constraint
    # --------------------

    if [[ "$structured" == "true" ]]; then
        output_flags+=(--structured-output)
    fi

    echo
    echo "============================================================"
    echo "Dataset:          $dataset"
    echo "Model:            $model"
    echo "Prompting:        $prompting"
    echo "Structured:       $structured"
    echo "Max steps:        ${dataset_max_steps[$dataset]}"
    echo "Max actions:      ${dataset_max_actions[$dataset]}"
    echo "Context window:   ${dataset_context_window[$dataset]}"
    echo "Temperature:      $temperature"
    echo "Seed:             $seed"
    echo "============================================================"
    echo

    python ./kgqa_navigation.py \
        --dataset "$dataset" \
        --hops "${dataset_hops[$dataset]}" \
        --max-navigation-steps "${dataset_max_steps[$dataset]}" \
        --max-actions "${dataset_max_actions[$dataset]}" \
        --context-window "${dataset_context_window[$dataset]}" \
        --llm-model "$model" \
        "${model_flags[@]}" \
        --navigation-approach tuple \
        --memory-approach full \
        "${prompt_flags[@]}" \
        "${output_flags[@]}" \
        --temperature "$temperature" \
        --seed "$seed" \
        --timeout "$timeout" \
        --timeout-cooldown 0 \
        --max-parse-retries "$max_parse_retries"
}


# ============================================================
# Experiments
# ============================================================

echo "Running KGQA Navigation Experiments"

# ------------------------------------------------------------
# 1. Zero-shot, original tuple navigation
# ------------------------------------------------------------

echo
echo "### Zero-shot / Full Memory / Tuple / Structured ###"

for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        run_navigation \
            "$dataset" \
            "$model" \
            "zero-shot" \
            "true"
    done
done


# ------------------------------------------------------------
# 2. One-shot, full demonstration
# ------------------------------------------------------------

echo
echo "### One-shot / Full Demo / Full Memory / Tuple / Structured ###"

for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        run_navigation \
            "$dataset" \
            "$model" \
            "one-shot" \
            "true"
    done
done