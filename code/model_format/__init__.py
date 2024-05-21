from .default_format import format_prompt_default
from .llama2_format import format_prompt_llama2
from .llama3_format import format_prompt_llama3
from .vicuna_format import format_vicuna_prompt

MODEL_FORMAT = {
    "default": format_prompt_default,
    "llama2": format_prompt_llama2,
    "llama3": format_prompt_llama3,
    "vicuna": format_vicuna_prompt,
}