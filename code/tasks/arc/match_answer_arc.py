import json
from ..match_answer import find_first_selection

def match_answer_arc(infer_result:dict, round_idx:int, args):
    task_config = args.tasks_config["arc"]
    result = {}
    for subject in task_config["subjects"]:
        correct_cnt = 0
        for item in infer_result[subject]:
            model_answer = find_first_selection(item[f"infer_round{round_idx}"])
            item[f"judge{round_idx}"] = False
            if item["ans"] == "E":
                correct_cnt += 1
                item[f"judge{round_idx}"] = True
                break
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
            
            item[f"extract_answer_round{round_idx}"] = model_answer
            
        result[subject] = {
            "acc": correct_cnt / len(infer_result[subject]),
        }

    result["arc"] = {
        subject: result[subject]["acc"] for subject in task_config["subjects"]
    }
    return result