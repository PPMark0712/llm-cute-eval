from .default_format import format_prompt_default
from .gemma_format import format_prompt_gemma
from .llama2_format import format_prompt_llama2
from .llama3_format import format_prompt_llama3
from .phi_format import format_prompt_phi
from .qwen_format import format_prompt_qwen
from .vicuna_format import format_vicuna_prompt


MODEL_FORMAT = {
    "default": format_prompt_default,
    "gemma": format_prompt_gemma,
    "llama2": format_prompt_llama2,
    "llama3": format_prompt_llama3,
    "phi": format_prompt_phi,
    "qwen": format_prompt_qwen,
    "vicuna": format_vicuna_prompt,
}