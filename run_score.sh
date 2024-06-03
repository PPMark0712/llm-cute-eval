export CUDA_VISIBLE_DEVICES=3
export TOKENIZERS_PARALLELISM=false
export  CUDA_LAUNCH_BLOCKING=1
# model_path=/data1/dcy/projects/fine-tune/fine-tune-yyz/train/output/5-16_11:21_llama2_13b/ckpt/llama2_13b_8000
# model_path=/data1/dcy/projects/fine-tune/fine-tune-yyz/train/output/5-1_00:12_Meta-Llama-3-8B-Instruct/ckpt/Meta-Llama-3-8B-Instruct_10000
# model_path=/data1/dcy/downloads/model/meta-llama/Meta-Llama-3-8B-Instruct
# model_path=/data1/dcy/downloads/model/meta-llama/Meta-Llama-3-70B-Instruct
# model_path=/data1/dcy/downloads/model/meta-llama/Meta-Llama-3-8B
# model_path=/data1/yyz/downloads/models/Xwin-LM/Xwin-Math-7B-V1.0
model_path=/data1/yyz/downloads/models/NousResearch/Llama-2-13b-hf
# model_path=/data1/dcy/projects/fine-tune/fine-tune-yyz/train/output/5-21_00:39_llama3_shuffle/ckpt/llama3_shuffle_6000

python main_score.py \
    --model_path $model_path \
    --model_type default \
    --tasks all\
    --save_name llama3_gen \
    --round 2 \
    --save_infer_results \
    --config_path config_dcy.json

