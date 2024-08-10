def match_answer_icleval(infer_result:dict, round_idx:int, args):
    result = {}
    total_cnt, correct_cnt = 0, 0
    for subject, subject_result in infer_result.items():
        subject_correct_cnt = 0
        total_cnt += len(subject_result)
        for item in subject_result:
            ans = str(item["label"]).strip()
            if subject == "generate_output_format":
                ans.replace("value", "key")
                if "####" in item["ans_content"]:
                    ans.replace("key", item["ans_content"].split("#### ")[1])
                else:
                    ans.replace("key", item["ans_content"])
            if ans in item[f"infer_round{round_idx}"]:
                subject_correct_cnt += 1
        result[subject] = {
            "acc": subject_correct_cnt / len(subject_result)
        }
        correct_cnt += subject_correct_cnt
    result["icleval"] = {
        "acc": correct_cnt / total_cnt,
    }
    return result