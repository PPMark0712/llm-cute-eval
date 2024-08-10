TASK_LIST = [
    "arc",
    "commonsenseqa",
    "gsm8k",
    "hellaswag",
    "humaneval",
    "icleval",
    "mmlu",
    "winogrande",
]

from .tasks import LOAD_TASK_DATA, MATCH_TASK_ANSWER
from .model_format import MODEL_FORMAT
