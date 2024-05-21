export CUDA_VISIBLE_DEVICES=4
export TOKENIZERS_PARALLELISM=false

# model_path=/data1/dcy/projects/fine-tune/fine-tune-yyz/train/output/5-16_11:21_llama2_13b/ckpt/llama2_13b_8000
# model_path=/data1/dcy/projects/fine-tune/fine-tune-yyz/train/output/5-1_00:12_Meta-Llama-3-8B-Instruct/ckpt/Meta-Llama-3-8B-Instruct_10000
model_path=/data1/dcy/downloads/model/meta-llama/Meta-Llama-3-8B-Instruct
model_path=/data1/dcy/downloads/model/meta-llama/Meta-Llama-3-8B
# model_path=/data1/yyz/downloads/models/Xwin-LM/Xwin-Math-7B-V1.0
# model_path=/data1/yyz/downloads/models/NousResearch/Llama-2-13b-hf
# model_path=/data1/dcy/projects/fine-tune/fine-tune-yyz/train/output/5-21_00:39_llama3_shuffle/ckpt/llama3_shuffle_36000

python main.py \
    --model_path $model_path \
    --model_type default \
    --tasks mmlu \
    --save_name yyz_debug \
    --round 1 \
    --save_infer_results \
    --config_path config_debug.json

