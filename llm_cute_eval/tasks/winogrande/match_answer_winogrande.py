import re
from ..match_answer import find_first_selection

pattern = r"answer\s+is\s+[\(\[\{]*([AB])[\)\]\}]*\.?"
def match_answer_winogrande(infer_result:dict, round_idx:int, args):
    result = {}
    correct_cnt = 0
    total_cnt = 0
    for item in infer_result["winogrande"]:
        matched_answer = re.findall(pattern, item[f"infer_round{round_idx}"])
        if len(matched_answer) > 0 and matched_answer[0] in "AB":
            model_answer = matched_answer[0]
        else:
            model_answer = find_first_selection(item[f"infer_round{round_idx}"])
        item[f"extracted_answer_round{round_idx}"] = model_answer
        item[f"judge{round_idx}"] = False
        total_cnt += 1
        if model_answer == item["ans"]:
            correct_cnt += 1
            item[f"judge{round_idx}"] = True
    result["winogrande"] = {
        "acc": correct_cnt / total_cnt,
    }
    return result
