from ..match_answer import find_first_selection

def match_answer_hellaswag(infer_result:dict, round_idx:int, args):
    result = {}
    correct_cnt = 0
    for item in infer_result["hellaswag"]:
        model_answer = find_first_selection(item[f"infer_round{round_idx}"])
        item[f"extract_answer_round{round_idx}"] = model_answer
        item[f"judge{round_idx}"] = False
        problem_answer = f'{item["ans"]} {item[item["ans"]]}'
        problem_answer_list = problem_answer.replace("(", " ").replace(")", " ").replace("\n", " ").strip().upper().split(" ")
        if model_answer is None:
            model_answer = item[f"infer_round{round_idx}"]
            model_answer_list = model_answer.upper().split(" ")
        else:
            model_answer_list = [model_answer.upper()]
        for problem_answer in problem_answer_list:
            if problem_answer in model_answer_list:
                correct_cnt += 1
                item[f"judge{round_idx}"] = True
                break
        
    result["hellaswag"] = {
        "acc": correct_cnt / len(infer_result["hellaswag"]),
    }
    return result