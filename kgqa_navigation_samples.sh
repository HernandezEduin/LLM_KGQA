# Evaluation on KGQA Iterative Navigation on a small subset of questions

# Evaluation on MQuAKE (zero-shot, original navigation)
python ./kgqa_navigation.py \
    --dataset mquake_single \
    --hops n \
    --max-navigation-steps 4 \
    --max-actions 200 \
    --context-window 32768 \
    --llm-model qwen2.5 \
    --use-instruct \
    --navigation-approach tuple \
    --memory-approach full \
    --prompting-approach zero-shot \
    --timeout 15

# Evaluation on MQuAKE (one-shot, original navigation)
python ./kgqa_navigation.py \
    --dataset mquake_single \
    --hops n \
    --max-navigation-steps 4 \
    --n-shots 1 \
    --demo-history-mode full \
    --demo-max-actions 5 \
    --max-actions 200 \
    --context-window 32768 \
    --llm-model qwen2.5 \
    --use-instruct \
    --navigation-approach tuple \
    --memory-approach full \
    --prompting-approach one-shot \
    --timeout 15

# Evaluation on MQuAKE (zero-shot, hybrid navigation)
python ./kgqa_navigation.py \
    --dataset mquake_single \
    --hops n \
    --max-navigation-steps 4 \
    --max-actions 200 \
    --context-window 32768 \
    --llm-model qwen2.5 \
    --use-instruct \
    --navigation-approach hybrid \
    --memory-approach full \
    --prompting-approach zero-shot \
    --timeout 15

# Evaluation on Kinship (zero-shot, original navigation)
python ./kgqa_navigation.py \
    --dataset kinship_v2 \
    --hops n \
    --max-navigation-steps 3 \
    --llm-model qwen2.5 \
    --use-instruct \
    --navigation-approach tuple \
    --memory-approach full \
    --prompting-approach zero-shot \
    --timeout 15
