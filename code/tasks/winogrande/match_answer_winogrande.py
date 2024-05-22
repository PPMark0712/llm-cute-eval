import re
from ..match_answer import find_first_selection

pattern = r"answer\s+is\s+[\(\[\{]*([AB])[\)\]\}]*\.?"
def match_answer_winogrande(infer_result:dict, round_idx:int, args):
    result = {}
    correct_cnt = 0
    for item in infer_result["winogrande"]:
        matched_answer = re.findall(pattern, item[f"infer_round{round_idx}"])
        if len(matched_answer) > 0 and matched_answer[0] in "AB":
            model_answer = matched_answer[0]
        else:
            model_answer = find_first_selection(item[f"infer_round{round_idx}"])
        item[f"extract_answer_round{round_idx}"] = model_answer
        if model_answer == item["ans"]:
            correct_cnt += 1
    result["winogrande"] = {
        "acc": correct_cnt / len(infer_result["winogrande"]),
    }
    return result