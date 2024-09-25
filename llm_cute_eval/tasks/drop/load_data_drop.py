import os, json


def load_file_drop(fn, limit=0):
    data = []
    with open(fn, "r", encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
            if limit and len(data) >= limit:
                break
    return data


def get_fewshot_cot_prompt_drop(drop_path, num_fewshots):
    assert 0 <= num_fewshots <= 8
    fewshot_cot_fn = os.path.join(drop_path, "fewshot.txt")
    file_str = ""
    with open(fewshot_cot_fn, "r") as f:
        for line in f:
            file_str += line
    fewshot_prompt = ""
    lst = file_str.split("\n\n")
    for text in lst[:num_fewshots]:
        fewshot_prompt += text + "\n\n"
    return fewshot_prompt


def load_data_drop(args):
    drop_path = os.path.join(args.data_path, "tasks", "drop")
    task_config = args.tasks_config["drop"]
    test_data = load_file_drop(os.path.join(drop_path, "test.jsonl"), task_config["limit"])
    task_data = {"drop": []}
    fewshot_prompt = get_fewshot_cot_prompt_drop(drop_path, task_config["num_fewshots"])
    for item in test_data:
        prompt = "Question: " + item["question"] + "\nAnswer: Let's think step by step\n"
        task_data["drop"].append({
            **item, 
            "instruction": task_config["instruction"],
            "fewshot_prompt": fewshot_prompt,
            "prompt_round1": prompt,
        })
    return task_data