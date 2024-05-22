from ..match_answer import find_first_selection

def match_answer_hellaswag(infer_result:dict, round_idx:int, args):
    result = {}
    correct_cnt = 0
    for item in infer_result["hellaswag"]:
        model_answer = find_first_selection(item[f"infer_round{round_idx}"])
        item[f"extract_answer_round{round_idx}"] = model_answer
        if model_answer == item["ans"]:
            correct_cnt += 1
    result["hellaswag"] = {
        "acc": correct_cnt / len(infer_result["hellaswag"]),
    }
    return result