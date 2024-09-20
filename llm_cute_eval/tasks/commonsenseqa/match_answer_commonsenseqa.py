from ..match_answer import find_first_selection

def match_answer_commonsenseqa(infer_result:dict, round_idx:int, args):
    result = {}
    correct_cnt = 0
    for item in infer_result["commonsenseqa"]:
        model_answer = find_first_selection(item[f"infer_round{round_idx}"])
        item[f"extract_answer_round{round_idx}"] = model_answer
        item[f"judge_round{round_idx}"] = False
        if model_answer == item["ans"]:
            correct_cnt += 1
            item[f"judge_round{round_idx}"] = True
        
    result["commonsenseqa"] = {
        "acc": correct_cnt / len(infer_result["commonsenseqa"]),
    }
    return result