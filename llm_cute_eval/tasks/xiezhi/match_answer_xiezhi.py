def match_answer_xiezhi(infer_result, round_idx, args):
    task_config = args.tasks_config["xiezhi"]
    result = {}
    for subject in task_config["subjects"]:
        correct_cnt = 0
        for item in infer_result[subject]:
            item[f"judge_round{round_idx}"] = False
            if item["answer"] in item[f"infer_round{round_idx}"]:
                correct_cnt += 1
                item[f"judge_round{round_idx}"] = True
                
        subject_result = correct_cnt / len(infer_result[subject])
        result[subject] = {
            "acc": subject_result,
            "correct_cnt": correct_cnt,
            "total_cnt": len(infer_result[subject])
        }

    result["xiezhi"] = {
        "acc": sum([result[subject]["correct_cnt"] for subject in task_config["subjects"]]) / sum([result[subject]["total_cnt"] for subject in task_config["subjects"]])
    }
    return result