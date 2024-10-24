import os, json


def format_query_winogrande(data, task_config, has_answer=False):
    question_template = task_config["question_template"]
    prompt = question_template.format(
        question=data["Q"],
        A=data["A"],
        B=data["B"],
    )
    if has_answer:
        prompt += f"({data['ans']})\n\n"
    return prompt


def load_file_winogrande(fn, limit=None):
    data = []
    with open(fn, "r") as f:
        for line in f:
            d = json.loads(line)
            question = {
                "Q": d["sentence"],
                "A": d["option1"],
                "B": d["option2"],
                "ans": "A" if d["answer"] == "1" else "B",
            }
            data.append(question)
            if limit and len(data) >= limit:
                break
    return data


def get_fewshot_prompt_winogrande(winogrande_path, task_config):
    fewshot_prompt = ""
    fewshot_fn = os.path.join(winogrande_path, "train_xs.jsonl")
    fewshot_data = load_file_winogrande(fewshot_fn, task_config["num_fewshots"])
    for item in fewshot_data:
        fewshot_prompt += format_query_winogrande(item, task_config, True)
    return fewshot_prompt


def get_fewshot_cot_prompt_winogrande(winogrande_path):
    fewshot_prompt = ""
    fewshot_fn = os.path.join(winogrande_path, "fewshot_cot.txt")
    with open(fewshot_fn, "r") as f:
        for line in f:
            fewshot_prompt += line
    return fewshot_prompt

def load_data_winogrande(args):
    winogrande_path = os.path.join(args.data_path, "tasks", "winogrande")
    task_config = args.tasks_config["winogrande"]
    winogrande_instruction = task_config["instruction"]
    task_data = {}
    test_fn = os.path.join(winogrande_path, "dev.jsonl")
    test_data = load_file_winogrande(test_fn, task_config["limit"])
    # fewshot_prompt = get_fewshot_prompt_winogrande(winogrande_dir, task_config["num_fewshots"])
    fewshot_cot_prompt = get_fewshot_cot_prompt_winogrande(winogrande_path)
    data = []
    for item in test_data:
        prompt = format_query_winogrande(item, task_config, False) + " Let's think step by step.\n"
        data.append({
            **item,
            "instruction": winogrande_instruction,
            "fewshot_prompt": fewshot_cot_prompt,
            "prompt_round1": prompt,
        })
    task_data = {"winogrande": data}
    return task_data
