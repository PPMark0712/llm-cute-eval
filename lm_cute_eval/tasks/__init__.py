from .arc.load_data_arc import load_data_arc
from .commonsenseqa.load_data_commonsenseqa import load_data_commonsenseqa
from .drop.load_data_drop import load_data_drop
from .gsm8k.load_data_gsm8k import load_data_gsm8k
from .hellaswag.load_data_hellaswag import load_data_hellaswag
from .humaneval.load_data_humaneval import load_data_humaneval
from .icleval.load_data_icleval import load_data_icleval
from .mmlu.load_data_mmlu import load_data_mmlu
from .rgb.load_data_rgb import load_data_rgb
from .winogrande.load_data_winogrande import load_data_winogrande
from .xsum.load_data_xsum import load_data_xsum
LOAD_TASK_DATA = {
    "arc": load_data_arc, 
    "commonsenseqa": load_data_commonsenseqa,
    "drop": load_data_drop,
    "gsm8k": load_data_gsm8k,
    "hellaswag": load_data_hellaswag,
    "humaneval": load_data_humaneval,
    "icleval": load_data_icleval,
    "mmlu": load_data_mmlu,
    "rgb": load_data_rgb,
    "winogrande": load_data_winogrande,
    "xsum": load_data_xsum,
}

from .arc.match_answer_arc import match_answer_arc
from .commonsenseqa.match_answer_commonsenseqa import match_answer_commonsenseqa
from .drop.match_answer_drop import match_answer_drop
from .gsm8k.match_answer_gsm8k import match_answer_gsm8k
from .hellaswag.match_answer_hellaswag import match_answer_hellaswag
from .humaneval.match_answer_humaneval import match_answer_humaneval
from .icleval.match_answer_icleval import match_answer_icleval
from .mmlu.match_answer_mmlu import match_answer_mmlu
from .rgb.match_answer_rgb import match_answer_rgb
from .winogrande.match_answer_winogrande import match_answer_winogrande
from .xsum.match_answer_xsum import match_answer_xsum
MATCH_TASK_ANSWER = {
    "arc": match_answer_arc,
    "commonsenseqa": match_answer_commonsenseqa,
    "drop": match_answer_drop,
    "gsm8k": match_answer_gsm8k,
    "hellaswag": match_answer_hellaswag,
    "humaneval": match_answer_humaneval,
    "icleval": match_answer_icleval,
    "mmlu": match_answer_mmlu,
    "rgb": match_answer_rgb,
    "winogrande": match_answer_winogrande,
    "xsum": match_answer_xsum
}