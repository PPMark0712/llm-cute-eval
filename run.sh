export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

declare -A models=(
    # ["Sheared-LLaMA-1.3B"]="/data1/yyz/downloads/models/princeton-nlp/Sheared-LLaMA-1.3B"
    ["Qwen2-7B"]="/data1/yyz/downloads/models/Qwen/Qwen2-7B"
)

for model_name in "${!models[@]}"; do
    model_path=${models[$model_name]}
    python main.py \
        --model_path "$model_path" \
        --model_type vllm \
        --format_type default \
        --tasks all \
        --save_name "$model_name" \
        --save_infer_texts \
        --save_infer_results \
        --config_path "config_debug.json" \
        --output_path output \
        --max_new_tokens 180 \
        --temperature 0.0 \
        --top_p 0.1 \
        --top_k 10 \

done