from vllm import LLM, SamplingParams
import torch

def init_vllm_model(args):
    round = args.rounds
    round = round-1
    temperatures = [0.9, 0.6, 0.3, 0.1]
    vllm_model = LLM(
        args.model_path,
        trust_remote_code=True,
        tensor_parallel_size=torch.cuda.device_count(),
    )
    if args.sampling_params:
        sampling_params = SamplingParams(**args.sampling_params)
    else:
        sampling_params = SamplingParams(
            max_tokens=512,  # 根据需要生成的内容长度来调整
            temperature=temperatures[0],
            top_p=0.8,
            top_k=50,
            frequency_penalty =0.5,  # 适度惩罚重复的词汇
            presence_penalty=0.5,  # 适度惩罚已经出现的词汇
            repetition_penalty=1,  # 适度鼓励新词汇的使用
            stop=["<|eot_id|>", "<|end_of_text|>", "Question:", "Human:"]
        )
    return vllm_model, sampling_params