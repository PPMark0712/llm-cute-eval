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
                model_answer = model_response.split("。")[0].strip()  # 假设模型按照'答案是：A,B,C。'的形式输出
                ans_choices = set(item["Answer"].split(","))  # 答案的格式固定，形如A,B,C
                model_choices = set()
                for c in model_answer:
                    if c in "ABCDEFG" and c not in model_choices:
                        model_choices.add(c)
                flag = ans_choices == model_choices
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
            

