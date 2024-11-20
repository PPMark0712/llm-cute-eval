TASK_LIST = [
    "arc",
    "cfinbench",
    "cmmlu",
    "commonsenseqa",
    "drop",
    "gsm8k",
    "hellaswag",
    "humaneval",
    "iclformat",
    "icleval",
    "mmlu",
    "mmluproplus",
    "rgb",
    "winogrande",
    "xiezhi",
    "xsum",
]

from .tasks import LOAD_TASK_DATA, MATCH_TASK_ANSWER
from .model_format import MODEL_FORMAT
