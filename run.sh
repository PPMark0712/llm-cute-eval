export CUDA_VISIBLE_DEVICES=7
export TOKENIZERS_PARALLELISM=false

declare -A models=(
    ["Qwen-7B"]="/data1/yyz/downloads/models/Qwen/Qwen2-7B"
)

for model_name in "${!models[@]}"; do
    model_path=${models[$model_name]}
    python main.py \
        --model_path "$model_path" \
        --model_type default \
        --sampling_params '{"max_tokens": 40, "stop": ["Input"]}' \
        --tasks all \
        --save_name "$model_name" \
        --save_infer_results \
        
done
