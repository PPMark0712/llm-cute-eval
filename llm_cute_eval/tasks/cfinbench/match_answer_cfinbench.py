import re
from ..match_answer import find_first_selection


def match_answer_cfinbench(infer_result:dict, round_idx, args):
    task_config = args.tasks_config["cfinbench"]
    result = {}
    pattern = r""
    for subject in task_config["subjects"]:
        correct_cnt = 0
        for item in infer_result[subject]:
            flag = False
            model_response = item[f"infer_round{round_idx}"]
            if item["q_type"] == "judgment":
                if item["Answer"] in model_response:
                    flag = True
            elif item["q_type"] == "multi_choice":
                model_choices = model_response.split("。")[0].strip()
                if item["Answer"] == model_choices:
                    flag = True
            elif item["q_type"] == "single_choice":
                if find_first_selection(model_response) == item["Answer"]:
                    flag = True
            else:
                raise TypeError
            
            item[f"judge_round{round_idx}"] = flag
            if flag:
                correct_cnt += 1
           
        subject_result = correct_cnt / len(infer_result[subject])
        result[subject] = {
            "acc": subject_result,
            "correct_cnt": correct_cnt,
            "total_cnt": len(infer_result[subject])
        }

    result["cfinbench"] = {
        "acc": sum([result[subject]["correct_cnt"] for subject in task_config["subjects"]]) / sum([result[subject]["total_cnt"] for subject in task_config["subjects"]])
    }
    return result
            

