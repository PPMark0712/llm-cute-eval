from vllm import LLM, SamplingParams
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


class VllmModel:
    def __init__(self, args) -> None:
        print("loading model...")
        self.model = LLM(
            model=args.model_path, 
            trust_remote_code=True,
            tensor_parallel_size=torch.cuda.device_count(),
        )
        sampling_kwargs = {
            "top_p": args.top_p,
            "top_k": args.top_k,
            "temperature": args.temperature,
            "max_tokens": args.max_new_tokens,
            "stop": [
                "</s>",
                "<|eot_id|>",
                # "Question:",
                "Human:",
                "Q:",
                "<|end_of_text|>",
                "<|start_header_id|>",
                "<|end_header_id|>",
                "Input",
                "问题",
            ]
        }
        self.sampling_kwargs = {k: v for k, v in sampling_kwargs.items() if v is not None}
    
    def generate(self, prompts, new_sampling_kwargs=None):
        sampling_kwargs = self.sampling_kwargs
        if new_sampling_kwargs:
            for k, v in new_sampling_kwargs.items():
                if k == "max_new_tokens":
                    sampling_kwargs["max_tokens"] = v
                elif k == "stop":
                    sampling_kwargs["stop"].extend(v)
                else:
                    sampling_kwargs[k] = v
        sampling_params = SamplingParams(**sampling_kwargs)
        outputs = self.model.generate(prompts, sampling_params)
        generated_texts = [output.outputs[0].text.strip() for output in outputs]
        return generated_texts
    
    
class HfModel:
    def __init__(self, args) -> None:
        print("loading model...")
        self.device = torch.device("cpu" if args.use_cpu else "cuda")
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        if args.temperature is not None and args.temperature == 0:
            self.sampling_kwargs = {
                "max_new_tokens": args.max_new_tokens,
                "pad_token_id": self.tokenizer.pad_token_id,
                "do_sample": False
            }
        else:
            self.sampling_kwargs = {
                "max_new_tokens": args.max_new_tokens,
                "pad_token_id": self.tokenizer.pad_token_id,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "do_sample": True,            
            }
            self.sampling_kwargs = {k: v for k, v in self.sampling_kwargs.items() if v is not None}
        self.model = AutoModelForCausalLM.from_pretrained(args.model_path).to(self.device)

    def generate(self, prompts, new_sampling_kwargs=None):
        sampling_kwargs = self.sampling_kwargs
        if new_sampling_kwargs:
            for k, v in new_sampling_kwargs.items():
                if k in ["top_p", "top_k", "temperature", "do_sample", "max_new_tokens"]:
                    sampling_kwargs[k] = v
        generated_texts = []
        for prompt in tqdm(prompts, desc="infering"):
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            generate_ids = self.model.generate(
                **inputs,
                **sampling_kwargs
            )
            output = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            generated_texts.append(output[len(prompt):].strip())
        return generated_texts


def initialize_model(args):
    model_type = {
        "hf": HfModel,
        "vllm": VllmModel
    }
    model = model_type[args.model_type](args)
    return model
