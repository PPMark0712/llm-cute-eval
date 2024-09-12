TASK_LIST = [
    "arc",
    "commonsenseqa",
    "drop",
    "gsm8k",
    "hellaswag",
    "humaneval",
    "icleval",
    "mmlu",
    "rgb",
    "winogrande",
    "xsum",
]

from .tasks import LOAD_TASK_DATA, MATCH_TASK_ANSWER
from .model_format import MODEL_FORMAT
