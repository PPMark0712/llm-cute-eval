import re
import string

def check_sentence_structure(s1, s2):
    def split_by_punctuation(sentence):
        sentence_structure = []
        str_between_puncts = ""
        for c in sentence:
            if c in ".,?":
                sentence_structure.append(len(str_between_puncts.split()))
                sentence_structure.append(c)
                str_between_puncts = ""
            else:
                str_between_puncts += c
        if str_between_puncts:
            sentence_structure.append(len(str_between_puncts.split()))
        return sentence_structure
    structure1 = split_by_punctuation(s1)
    structure2 = split_by_punctuation(s2)
    return structure1 == structure2


def match_answer_iclformat(infer_result:dict, round_idx, args):
    task_config = args.tasks_config["iclformat"]
    result = {}
    acc_sum = 0
    for subject in task_config["subjects"]:
        correct_cnt = 0
        for item in infer_result[subject]:
            flag = False
            item[f"judge_round{round_idx}"] = False
            # match = re.search(pattern, item[f"infer_round{round_idx}"], re.DOTALL)
            # if not match:
            #     continue
            # model_response = match.group(1).strip()
            model_response = item[f"infer_round{round_idx}"].split("<output>", 1)[-1].strip()

            if subject in ["format_tree", "struct_to_struct", "struct_to_text", "text_to_struct", "text_to_text"]:
                if model_response == item["output"].strip():
                    flag = True
            elif subject in ["format_answer", "format_choice"]:
                if re.match(item["pattern"], model_response):
                    flag = True
            elif subject == "bullet_pointed_response":
                flag = True
                for label in item["label_list"][:3]:
                    if label not in model_response:
                        flag = False
                # answer_lines = model_response.split("\n")    
                # for i in range(min(len(answer_lines), len(item["label_list"]))):
                #     if not answer_lines[i].startswith(item["label_list"][i]):
                #         flag = False
            elif subject in ["short_sentence", "long_sentence"]:
                if check_sentence_structure(model_response, item["input"]):
                    flag = True
            
            if flag:
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