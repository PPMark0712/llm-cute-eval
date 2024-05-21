from .arc.load_data_arc import load_data_arc
from .commonsenseqa.load_data_commonsenseqa import load_data_commonsenseqa
from .gsm8k.load_data_gsm8k import load_data_gsm8k
from .hellaswag.load_data_hellaswag import load_data_hellaswag
from .humaneval.load_data_humaneval import load_data_humaneval
from .mmlu.load_data_mmlu import load_data_mmlu
from .winogrande.load_data_winogrande import load_data_winogrande

LOAD_TASK_DATA = {
    "arc": load_data_arc, 
    "commonsenseqa": load_data_commonsenseqa,
    "gsm8k": load_data_gsm8k,
    "hellaswag": load_data_hellaswag,
    "humaneval": load_data_humaneval,
    "mmlu": load_data_mmlu,
    "winogrande": load_data_winogrande,
}

from .arc.match_answer_arc import match_answer_arc
from .commonsenseqa.match_answer_commonsenseqa import match_answer_commonsenseqa
from .gsm8k.match_answer_gsm8k import match_answer_gsm8k
from .hellaswag.match_answer_hellaswag import match_answer_hellaswag
from .humaneval.match_answer_humaneval import match_answer_humaneval
from .mmlu.match_answer_mmlu import match_answer_mmlu
from .winogrande.match_answer_winogrande import match_answer_winogrande

MATCH_TASK_ANSWER = {
    "arc": match_answer_arc,
    "commonsenseqa": match_answer_commonsenseqa,
    "gsm8k": match_answer_gsm8k,
    "hellaswag": match_answer_hellaswag,
    "humaneval": match_answer_humaneval,
    "mmlu": match_answer_mmlu,
    "winogrande": match_answer_winogrande,
}