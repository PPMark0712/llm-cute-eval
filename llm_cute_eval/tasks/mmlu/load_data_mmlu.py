import os, csv, json



def format_query_mmlu(data, question_template, has_answer=False):
    prompt = question_template.format(
        Q=data["Q"],
        A=data["A"],
        B=data["B"],
        C=data["C"],
        D=data["D"]
    )
    if has_answer:
        prompt += f"{data['ans']}).\n\n"
    return prompt


def load_file_mmlu(fn, limit=0):
    data = []
    with open(fn, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            data.append({
                "Q": row[0],
                "A": row[1],
                "B": row[2],
                "C": row[3],
                "D": row[4],
                "ans": row[5],
            })
            if limit and len(data) >= limit:
                break
    return data


def get_inst_and_fewshot_cot(fewshot_data, subject, num_fewshots):
    text = fewshot_data[subject].strip()
    idx = text.find("Q:")
    instruction = text[:idx]
    fewshot_cot_prompt = text[idx:] + "\n\n\n\n"
    return instruction, fewshot_cot_prompt


def get_fewshot_prompt(fewshot_data, config):
    assert 0 <= config["num_fewshots"] <= 5
    fewshot_prompt = ""
    for item in fewshot_data:
        fewshot_prompt += format_query_mmlu(item, config["question_template"], True)
    return fewshot_prompt


def load_data_mmlu(args):
    mmlu_path = os.path.join(args.data_path, "tasks", "mmlu")
    task_config = args.tasks_config["mmlu"]

    if task_config["use_cot"]:
        fewshot_cot_fn = os.path.join(mmlu_path, "fewshot-cot", "mmlu-cot-claude-multiple.json")
        with open(fewshot_cot_fn, "r") as f:
            fewshot_cot_data = json.load(f)
    
    task_data = {}
    subjects = task_config["subjects"]
    for subject in subjects:
        fn = os.path.join(mmlu_path, "test", f"{subject}_test.csv")
        subject_data = load_file_mmlu(fn, task_config["limit"])
        if task_config["use_cot"]:
            instruction, fewshot_prompt = get_inst_and_fewshot_cot(fewshot_cot_data, subject, task_config["num_fewshots"])
        else:
            instruction = task_config["instruction_template"].format(subject=subject)
            fewshot_fn = os.path.join(mmlu_path, "dev", f"{subject}_dev.csv")
            fewshot_data = load_file_mmlu(fewshot_fn, task_config["num_fewshots"])
            fewshot_prompt = get_fewshot_prompt(fewshot_data, task_config)
        task_data[subject] = []
        for item in subject_data:
            if task_config["use_cot"]:
                prompt = format_query_mmlu(item, task_config["question_template_cot"])
            else:
                prompt = format_query_mmlu(item, task_config["question_template"])
            task_data[subject].append({
                **item,
                "instruction": instruction,
                "fewshot_prompt": fewshot_prompt,
                "prompt_round1": prompt,
            })
    return task_data
