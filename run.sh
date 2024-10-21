export CUDA_VISIBLE_DEVICES=2
export TOKENIZERS_PARALLELISM=false

declare -A models=(
    # ["Sheared-LLaMA-1.3B"]="/data1/yyz/downloads/models/princeton-nlp/Sheared-LLaMA-1.3B"
    # ["Qwen2-7B"]="/data1/yyz/downloads/models/Qwen/Qwen2-7B"
    ["Qwen2.5-7B-Instruct"]="/data1/yyz/downloads/models/Qwen/Qwen2.5-7B-Instruct"
)

for model_name in "${!models[@]}"; do
    model_path=${models[$model_name]}
    python main.py \
        --model_path "$model_path" \
        --model_type vllm \
        --format_type default \
        --tasks iclbench \
        --save_name "$model_name" \
        --save_infer_texts \
        --save_infer_results \
        --config_path "" \
        --output_path output \
        --max_new_tokens 1200 \
        --temperature 0.0 \
        --top_p 0.1 \
        --top_k 10 \
        --no_timestamp

done