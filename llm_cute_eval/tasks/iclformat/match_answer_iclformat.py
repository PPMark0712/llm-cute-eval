import re
from ..match_answer import find_first_selection


def match_answer_iclformat(infer_result:dict, round_idx, args):
    pattern = r"<output>(.*?)</output>"
    task_config = args.tasks_config["iclformat"]
    result = {}
    acc_sum = 0
    for subject in task_config["subjects"]:
        if subject in ["format_tree", "struct_to_struct", "struct_to_text", "text_to_struct", "text_to_text"]:
            correct_cnt = 0
            for item in infer_result[subject]:
                item[f"judge_round{round_idx}"] = False
                match = re.search(pattern, item[f"infer_round{round_idx}"], re.DOTALL)
                if match:
                    model_response = match.group(1).strip()
                    if model_response == item["output"].strip():
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
            

