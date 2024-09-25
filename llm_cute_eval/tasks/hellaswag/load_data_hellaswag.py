import os, json


def format_query_hellaswag(question_template, data, has_answer):
    prompt = question_template.format(
        question=data["Q"],
        A=data["A"],
        B=data["B"],
        C=data["C"],
        D=data["D"],
    )
    if has_answer:
        prompt += f"({data['ans']})\n\n"
    return prompt


def load_file_hellaswag(fn, limit=None):
    data = []
    with open(fn, "r") as f:
        for line in f:
            d = json.loads(line)
            choices = ["A", "B", "C", "D"]
            question = {"Q": d["query"]}
            for i in range(4):
                question[choices[i]] = f'{d["choices"][i]}'
            question["ans"] = choices[d["gold"]]
            data.append(question)
            if limit and len(data) >= limit:
                break
    return data


def get_fewshot_prompt_hellaswag(hellaswag_path, question_template, num_fewshots):
    assert 0 <= num_fewshots <= 25
    fewshot_prompt = ""
    fewshot_fn = os.path.join(hellaswag_path, "hellaswag_train_sampled25.jsonl")
    fewshot_data = load_file_hellaswag(fewshot_fn, num_fewshots)
    for item in fewshot_data:
        fewshot_prompt += format_query_hellaswag(question_template, item, True)
    return fewshot_prompt


def load_data_hellaswag(args):
    hellaswag_path = os.path.join(args.data_path, "tasks", "hellaswag")
    task_config = args.tasks_config["hellaswag"]
    question_template = task_config["question_template"]    
    task_data = {}
    test_fn = os.path.join(hellaswag_path, "hellaswag.jsonl")
    test_data = load_file_hellaswag(test_fn, task_config["limit"])
    fewshot_prompt = get_fewshot_prompt_hellaswag(hellaswag_path, question_template, task_config["num_fewshots"])
    data = []
    for item in test_data:
        prompt = format_query_hellaswag(question_template, item, False)
        data.append({
            **item,
            "instruction": task_config["instruction"],
            "fewshot_prompt": fewshot_prompt,
            "prompt_round1": prompt,
        })
    task_data = {"hellaswag": data}
    return task_data