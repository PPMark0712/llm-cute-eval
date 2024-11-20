import re
from ..match_answer import find_first_selection


def match_answer_cfinbench(infer_result:dict, round_idx, args):
    task_config = args.tasks_config["cfinbench"]
    result = {}
    for subject in task_config["subjects"]:
        correct_cnt = 0
        for item in infer_result[subject]:
            flag = False
            model_response = item[f"infer_round{round_idx}"]
            if item["q_type"] == "judgment":
                if item["Answer"] in model_response:
                    flag = True
            elif item["q_type"] == "multi_choice":
                model_answer = model_response.split("。")[0].strip()
                ans_choices = item["Answer"].split(",")
                model_choices = []
                for c in model_answer:
                    if c in "ABCDEFG" and c not in model_choices:
                        model_choices.append(c)
                flag = True
                if len(ans_choices) != len(model_choices):
                    flag = False
                else:
                    model_choices.sort()
                    for c1, c2 in zip(ans_choices, model_choices):
                        if c1 != c2:
                            flag = False
                            break
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
            

