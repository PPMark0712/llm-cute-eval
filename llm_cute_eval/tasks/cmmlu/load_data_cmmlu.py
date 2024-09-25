import os, csv


def format_query_cmmlu(data, config, has_answer=False):
    prompt = config["question_template"].format(
        Q=data["Q"],
        A=data["A"],
        B=data["B"],
        C=data["C"],
        D=data["D"],
    )
    if has_answer:
        prompt += data["ans"] + "\n\n"
    return prompt


def load_file_cmmlu(fn, limit=None):
    data = []
    first_line = True
    with open(fn, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if first_line:
                first_line = False
                continue
            data.append({
                "Q": row[1],
                "A": row[2],
                "B": row[3],
                "C": row[4],
                "D": row[5],
                "ans": row[6],
            })
            if limit and len(data) >= limit:
                break
    return data


def get_inst_and_fewshot_cot(fewshot_data, subject):
    text = fewshot_data[subject].strip()
    idx = text.find("Q:")
    instruction = text[:idx]
    fewshot_cot_prompt = text[idx:]
    return instruction, fewshot_cot_prompt


def get_fewshot_prompt(data, idx, config):
    n = config["num_fewshots"]
    fewshot_data = data[idx:idx + n] if idx + n <= len(data) else data[idx:] + data[:idx + n - len(data)]
    fewshot_prompt = ""
    for item in fewshot_data:
        fewshot_prompt += format_query_cmmlu(item, config, True)
    return fewshot_prompt


def load_data_cmmlu(args):
    cmmlu_path = os.path.join(args.data_path, "tasks", "cmmlu")
    task_config = args.tasks_config["cmmlu"]
    task_data = {}
    subject_dict = task_config["subjects"]
    for subject_en, subject_zh in subject_dict.items():
        test_fn = os.path.join(cmmlu_path, "test", f"{subject_en}.csv")
        test_data = load_file_cmmlu(test_fn, task_config["limit"])

        dev_fn = os.path.join(cmmlu_path, "dev", f"{subject_en}.csv")
        dev_data = load_file_cmmlu(dev_fn)
        dev_idx = 0

        instruction = task_config["instruction_template"].format(subject=subject_zh)

        task_data[subject_en] = []
        for item in test_data:
            prompt = format_query_cmmlu(item, task_config)
            fewshot_prompt = get_fewshot_prompt(dev_data, dev_idx, task_config)
            dev_idx = (dev_idx + task_config["num_fewshots"]) % len(dev_data)
            task_data[subject_en].append({
                **item,
                "instruction": instruction,
                "fewshot_prompt": fewshot_prompt,
                "prompt_round1": prompt,
            })
    return task_data
