from ..match_answer import find_first_selection


def match_answer_cmmlu(infer_result:dict, round_idx, args):
    task_config = args.tasks_config["cmmlu"]
    result = {}
    for subject in task_config["subjects"]:
        correct_cnt = 0
        for item in infer_result[subject]:
            model_answer = find_first_selection(item[f"infer_round{round_idx}"])
            item[f"extracted_answer_round{round_idx}"] = model_answer
            item[f"judge_round{round_idx}"] = False
            if model_answer == item["ans"]:
                correct_cnt += 1
                item[f"judge_round{round_idx}"] = True
                
        subject_result = correct_cnt / len(infer_result[subject])
        result[subject] = {
            "acc": subject_result,
            "correct_cnt": correct_cnt,
            "total_cnt": len(infer_result[subject])
        }

    result["cmmlu"] = {
        "acc": sum([result[subject]["correct_cnt"] for subject in task_config["subjects"]]) / sum([result[subject]["total_cnt"] for subject in task_config["subjects"]])
    }
    return result
            

