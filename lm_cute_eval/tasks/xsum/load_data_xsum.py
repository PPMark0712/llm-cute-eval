import os, json


def load_file_xsum(fn, limit=0):
    data = []
    with open(fn, "r", encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
            if limit and len(data) >= limit:
                break
    return data


def get_fewshot_cot_prompt_xsum(xsum_path, num_fewshot):
    assert 0 <= num_fewshot <= 8
    fewshot_cot_fn = os.path.join(xsum_path, "fewshot.txt")
    file_str = ""
    with open(fewshot_cot_fn, "r") as f:
        for line in f:
            file_str += line
    fewshot_prompt = ""
    lst = file_str.split("\n\n")
    for text in lst[:num_fewshot]:
        fewshot_prompt += text + "\n\n"
    return fewshot_prompt


def load_data_xsum(args):
    xsum_path = os.path.join(args.data_path, "tasks", "xsum")
    task_config = args.tasks_config["xsum"]
    test_data = load_file_xsum(os.path.join(xsum_path, "test.jsonl"), task_config["limit"])
    task_data = {"xsum": []}
    fewshot_prompt = get_fewshot_cot_prompt_xsum(xsum_path, task_config["num_fewshot"])
    for item in test_data:
        prompt = "Question: " + item["question"] + "\nAnswer:"
        task_data["xsum"].append({
            **item, 
            "instruction": task_config["instruction"],
            "fewshot_prompt": fewshot_prompt,
            "prompt_round1": prompt,
        })
    return task_data
