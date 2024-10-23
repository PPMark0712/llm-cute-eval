from .arc.load_data_arc import load_data_arc
from .cfinbench.load_data_cfinbench import load_data_cfinbench
from .cmmlu.load_data_cmmlu import load_data_cmmlu
from .commonsenseqa.load_data_commonsenseqa import load_data_commonsenseqa
from .drop.load_data_drop import load_data_drop
from .gsm8k.load_data_gsm8k import load_data_gsm8k
from .hellaswag.load_data_hellaswag import load_data_hellaswag
from .humaneval.load_data_humaneval import load_data_humaneval
from .iclformat.load_data_iclformat import load_data_iclformat
from .icleval.load_data_icleval import load_data_icleval
from .mmlu.load_data_mmlu import load_data_mmlu
from .rgb.load_data_rgb import load_data_rgb
from .winogrande.load_data_winogrande import load_data_winogrande
from .xiezhi.load_data_xiezhi import load_data_xiezhi
from .xsum.load_data_xsum import load_data_xsum


LOAD_TASK_DATA = {
    "arc": load_data_arc, 
    "cfinbench": load_data_cfinbench,
    "cmmlu": load_data_cmmlu,
    "commonsenseqa": load_data_commonsenseqa,
    "drop": load_data_drop,
    "gsm8k": load_data_gsm8k,
    "hellaswag": load_data_hellaswag,
    "humaneval": load_data_humaneval,
    "iclformat": load_data_iclformat,
    "icleval": load_data_icleval,
    "mmlu": load_data_mmlu,
    "rgb": load_data_rgb,
    "winogrande": load_data_winogrande,
    "xiezhi": load_data_xiezhi,
    "xsum": load_data_xsum,
}


from .arc.match_answer_arc import match_answer_arc
from .cfinbench.match_answer_cfinbench import match_answer_cfinbench
from .cmmlu.match_answer_cmmlu import match_answer_cmmlu
from .commonsenseqa.match_answer_commonsenseqa import match_answer_commonsenseqa
from .drop.match_answer_drop import match_answer_drop
from .gsm8k.match_answer_gsm8k import match_answer_gsm8k
from .hellaswag.match_answer_hellaswag import match_answer_hellaswag
from .humaneval.match_answer_humaneval import match_answer_humaneval
from .iclformat.match_answer_iclformat import match_answer_iclformat
from .icleval.match_answer_icleval import match_answer_icleval
from .mmlu.match_answer_mmlu import match_answer_mmlu
from .rgb.match_answer_rgb import match_answer_rgb
from .winogrande.match_answer_winogrande import match_answer_winogrande
from .xiezhi.match_answer_xiezhi import match_answer_xiezhi
from .xsum.match_answer_xsum import match_answer_xsum


MATCH_TASK_ANSWER = {
    "arc": match_answer_arc,
    "cfinbench": match_answer_cfinbench,
    "cmmlu": match_answer_cmmlu,
    "commonsenseqa": match_answer_commonsenseqa,
    "drop": match_answer_drop,
    "gsm8k": match_answer_gsm8k,
    "hellaswag": match_answer_hellaswag,
    "humaneval": match_answer_humaneval,
    "iclformat": match_answer_iclformat,
    "icleval": match_answer_icleval,
    "mmlu": match_answer_mmlu,
    "rgb": match_answer_rgb,
    "winogrande": match_answer_winogrande,
    "xiezhi": match_answer_xiezhi,
    "xsum": match_answer_xsum
}