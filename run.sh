export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

declare -A models=(
    # ["Llama-2-7b-hf"]="/data1/yyz/downloads/models/NousResearch/Llama-2-7b-hf"
    # ["Xwin-Math-7B-V1.1"]="/data1/yyz/downloads/models/Xwin-LM/Xwin-Math-7B-V1.1"
    # ["Qwen1.5-14B"]="/data1/yyz/downloads/models/Qwen/Qwen1.5-14B"
    # ["Qwen1.5-14B-FT"]="/data1/yyz/downloads/models/Qwen/Qwen1.5-14B-FT"
    ["Qwen2-7B"]="/data1/yyz/downloads/models/Qwen/Qwen2-7B"
    # ["Qwen2-7B-refine"]="/data1/dcy/projects/fine-tune/fine-tune-yyz/train/output/8-30_23:59_qwen2_all_refine"
    # ["Qwen2-7B-sft"]="/data1/dcy/projects/fine-tune/fine-tune-yyz/train/output/8-30_14:57_qwen2_all_sft"
)

for model_name in "${!models[@]}"; do
    model_path=${models[$model_name]}
    python main.py \
        --model_path "$model_path" \
        --model_type hf \
        --format_type default \
        --tasks rgb icleval \
        --save_name "$model_name" \
        --save_infer_texts \
        --config_path config_debug.json \
        --output_path output/debug \
        --use_cpu
done
