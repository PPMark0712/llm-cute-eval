import os, json


def format_cqa_query(data, config, has_answer=False):
    prompt = config["question_template"].format(
        question=data["Q"],
        A=data["A"],
        B=data["B"],
        C=data["C"],
        D=data["D"],
        E=data["E"],
    )
    if has_answer:
        prompt += f"({data['ans']})\n\n\n"
    return prompt


def load_file_cqa(fn, limit=None):
    data = []
    with open(fn, "r") as f:
        for line in f:
            d = json.loads(line)
            choices = ["A", "B", "C", "D", "E"]
            question = {"Q": d["question"]["stem"]}
            for i in range(5):
                question[choices[i]] = d["question"]["choices"][i]["text"]
            question["ans"] = d["answerKey"]
            data.append(question)
            if limit and len(data) >= limit:
                break
    return data


def get_fewshot_prompt_commonsenseqa(train_data, begin_idx, config):
    num_fewshots = config["num_fewshots"]
    fewshot_prompt = ""
    for i in range(begin_idx, begin_idx + num_fewshots):
        fewshot_prompt += format_cqa_query(train_data[i % len(train_data)], config, True)
    return fewshot_prompt


def load_data_commonsenseqa(args):
    cqa_path = os.path.join(args.data_path, "tasks", "commonsenseqa")
    task_config = args.tasks_config["commonsenseqa"]
    cqa_instruction = task_config["instruction"]
    task_data = {}
    test_fn = os.path.join(cqa_path, "dev_rand_split.jsonl")
    train_fn = os.path.join(cqa_path, "train_rand_split.jsonl")
    test_data = load_file_cqa(test_fn, task_config["limit"])
    train_data = load_file_cqa(train_fn, task_config["limit"] * task_config["num_fewshots"] if task_config["limit"] else 0)
    data = []
    cur_fewshot_begin_idx = 0
    for item in test_data:
        fewshot_prompt = get_fewshot_prompt_commonsenseqa(train_data, cur_fewshot_begin_idx, task_config)
        cur_fewshot_begin_idx += task_config["num_fewshots"]
        prompt = format_cqa_query(item, task_config, False)
        data.append({
            **item,
            "instruction": cqa_instruction,
            "fewshot_prompt": fewshot_prompt,
            "prompt_round1": prompt,
        })
    task_data = {"commonsenseqa": data}
    return task_data
