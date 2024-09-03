from vllm import LLM, SamplingParams
import torch

def init_vllm_model(args):
    vllm_model = LLM(
        args.model_path,
        trust_remote_code=True,
        tensor_parallel_size=torch.cuda.device_count(),
    )
    if args.sampling_params:
        sampling_params = SamplingParams(**args.sampling_params)
    else:
        sampling_params = SamplingParams(
            max_tokens=128,  # 根据需要生成的内容长度来调整
            temperature=0,
            top_p=0.1,
            top_k=10,
            stop=[
                "Question:",
                "</s>",
                "<|eot_id|>",
                "Human:", "Q:",
                "Text:",
                "<|end_of_text|>",
                "<|start_header_id|>",
                "<|end_header_id|>",
                "Input"
            ]
        )
    return vllm_model, sampling_params