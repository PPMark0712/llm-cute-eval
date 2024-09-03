from ..match_answer import find_first_selection

def match_answer_hellaswag(infer_result:dict, round_idx:int, args):
    result = {}
    correct_cnt = 0
    total_cnt = 0
    for item in infer_result["hellaswag"]:
        model_answer = find_first_selection(item[f"infer_round{round_idx}"])
        item[f"extract_answer_round{round_idx}"] = model_answer
        item[f"judge{round_idx}"] = False
        if not item[f"extract_answer_round{round_idx}"]:
            continue
        total_cnt+=1
        if model_answer == item["ans"]:
            correct_cnt += 1
            item[f"judge{round_idx}"] = True
    result["hellaswag"] = {
        "acc": correct_cnt / total_cnt,
    }
    return result