export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false

# model_path=/data1/dcy/projects/fine-tune/fine-tune-yyz/train/output/5-16_11:21_llama2_13b/ckpt/llama2_13b_8000
# model_path=/data1/dcy/projects/fine-tune/fine-tune-yyz/train/output/5-1_00:12_Meta-Llama-3-8B-Instruct/ckpt/Meta-Llama-3-8B-Instruct_10000
# model_path=/data1/dcy/downloads/model/meta-llama/Meta-Llama-3-8B-Instruct
# model_path=/data1/dcy/downloads/model/meta-llama/Meta-Llama-3-70B-Instruct
# model_path=/data1/dcy/downloads/model/meta-llama/Meta-Llama-3-8B
# model_path=/data1/yyz/downloads/models/Xwin-LM/Xwin-Math-7B-V1.0
model_path=/data1/yyz/downloads/models/NousResearch/Llama-2-13b-hf
model_path=/data1/dcy/projects/fine-tune/fine-tune-yyz/train/output/5-25_08:01_llama3_shuffle/ckpt/llama3_shuffle_1000
model_path=/data1/dcy/projects/fine-tune/fine-tune-yyz/train/output/5-25_13:20_llama3_shuffle/ckpt/llama3_shuffle_8000
# model_path=/data1/dcy/projects/fine-tune/fine-tune-yyz/train/output/5-21_00:39_llama3_shuffle/ckpt/llama3_shuffle_6000
model_path=/data1/dcy/projects/fine-tune/fine-tune-yyz/train/output/5-29_17:20_llama3_shuffle/ckpt/llama3_shuffle_12000
model_path=/data1/dcy/projects/fine-tune/fine-tune-yyz/train/output/6-3_13:07_llama3_shuffle_lm/ckpt/llama3_shuffle_lm_6000
model_path=/data1/dcy/projects/fine-tune/fine-tune-yyz/train/output/6-3_12:58_llama3_shuffle_dpo/ckpt/llama3_shuffle_dpo_8000
model_path=/data1/dcy/projects/fine-tune/fine-tune-yyz/train/output/6-3_12:57_llama3_shuffle_dpo/ckpt/llama3_shuffle_dpo_4000
model_path=/data1/dcy/projects/fine-tune/fine-tune-yyz/train/output/6-3_12:57_llama3_shuffle_dpo/ckpt/llama3_shuffle_dpo_10000
# model_path=/data1/dcy/projects/fine-tune/fine-tune-yyz/train/output/6-3_12:58_llama3_shuffle_dpo/ckpt/llama3_shuffle_dpo_10000
model_path=/data1/dcy/projects/fine-tune/fine-tune-yyz/train/output/6-4_15:35_llama3_shuffle_dpo_base_14000/ckpt/llama3_shuffle_dpo_base_14000_28000

model_path=/data1/dcy/downloads/model/meta-llama/Meta-Llama-3-70B-Instruct
model_path=/data1/yyz/downloads/models/NousResearch/Llama-2-13b-chat-hf
model_path=/data1/dcy/projects/fine-tune/fine-tune-yyz/train/output/6-9_19:39_llama3_shuffle_dpo/ckpt/llama3_shuffle_dpo_100000
model_path=/data1/dcy/projects/fine-tune/LLaMA-Factory-main/saves/LLaMA3-8B-Chat/full/train_2024-07-20-23-25-59/checkpoint-1544
model_path=/data1/dcy/projects/fine-tune/LLaMA-Factory-main/saves/Qwen2-7B-Chat/full/train_2024-07-24-15-03-40/checkpoint-525
model_path=/data1/dcy/projects/fine-tune/LLaMA-Factory-main/saves/LLaMA3-8B-Chat/full/train_2024-07-20-23-25-59
model_path=/data1/dcy/projects/fine-tune/LLaMA-Factory-main/saves/Qwen2-7B-Chat/full/train_2024-07-24-15-03-40/checkpoint-525
python main.py \
    --model_path $model_path \
    --model_type qwen \
    --tasks all\
    --save_name qwen-sft_0808 \
    --round 4 \
    --save_infer_results \
    --config_path config_zero.json
