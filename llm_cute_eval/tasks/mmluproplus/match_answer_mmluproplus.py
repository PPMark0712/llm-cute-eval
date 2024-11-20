from ..match_answer import find_first_selection

def match_answer_mmluproplus(infer_result:dict, round_idx, args):
    task_config = args.tasks_config["mmluproplus"]
    result = {}
    acc_sum = 0
    for subject in task_config["subjects"]:
        correct_cnt = 0
        for item in infer_result[subject]:
            model_answer = find_first_selection(item[f"infer_round{round_idx}"], "ABCDEFGHIJKLMN")
            item[f"extracted_answer_round{round_idx}"] = model_answer
            item[f"judge_round{round_idx}"] = False
            if model_answer == item["answer"]:
                correct_cnt += 1
                item[f"judge_round{round_idx}"] = True
     
        acc = correct_cnt / len(infer_result[subject])
        acc_sum += acc
        result[subject] = {
            "acc": acc,
            "correct_cnt": correct_cnt,
            "total_cnt": len(infer_result[subject])
        }

    result["mmluproplus"] = {
        "acc": acc_sum / len(task_config["subjects"])
    }
    return result
            

