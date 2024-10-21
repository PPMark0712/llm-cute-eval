import re
from ..match_answer import find_first_selection
from collections import defaultdict


def match_answer_iclformat(infer_result:dict, round_idx, args):
    task_config = args.tasks_config["iclformat"]
    result = {}
    acc_sum = 0
    for subject in task_config["subjects"]:
        if subject in ["rubustness"]:
            bucket = {}
            for item in infer_result[subject]:
                model_response = item[f"infer_round{round_idx}"].strip()
                model_answer = find_first_selection(model_response)
                id = item["id"].split("-")[0]
                if id not in bucket:
                    bucket[id] = {}
                if model_answer not in bucket[id]:
                    bucket[id][model_answer] = 0
                bucket[id][model_answer] += 1
            correct_cnt = 0
            for id, choice_dict in bucket.items():
                if len(choice_dict.keys()) == 1:
                    correct_cnt += 1
            acc = correct_cnt / len(bucket)
            acc_sum += acc
            result[subject] = {
                "acc": acc,
                "correct_cnt": correct_cnt,
                "total_cnt": len(bucket)
                # "details": bucket,
            }
        elif subject in ["learn"]:
            correct_cnt = 0
            for item in infer_result[subject]:
                item[f"judge_round{round_idx}"] = False
                model_response = item[f"infer_round{round_idx}"]
                model_answer = find_first_selection(model_response)
                if model_answer == item["answer"]:
                    item[f"judge_round{round_idx}"] = True
                    correct_cnt += 1
            acc = correct_cnt / len(infer_result[subject])
            acc_sum += acc
            result[subject] = {
                "acc": acc,
                "correct_cnt": correct_cnt,
                "total_cnt": len(infer_result[subject])
            }
        else:
            correct_cnt = 0
            for item in infer_result[subject]:
                item[f"judge_round{round_idx}"] = False
                model_response = item[f"infer_round{round_idx}"].strip()
                if "match_pattern" in item:
                    pattern = item["match_pattern"]
                    if re.findall(pattern, model_response[:len(pattern) + 5]):
                        correct_cnt += 1
                        item[f"judge_round{round_idx}"] = True
                else:
                    if model_response.startswith(item["answer"]):
                        correct_cnt += 1
                        item[f"judge_round{round_idx}"] = True
        
            acc = correct_cnt / len(infer_result[subject])
            acc_sum += acc
            result[subject] = {
                "acc": acc,
                "correct_cnt": correct_cnt,
                "total_cnt": len(infer_result[subject])
            }

    result["iclformat"] = {
        "acc": acc_sum / len(task_config["subjects"])
    }
    return result
            

