import os, json


def read_file(fn, limit=0):
    data = []
    with open(fn, "r", encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
            if limit and len(data) >= limit:
                break
    return data


def format_question(item, question_template, has_answer=False):
    choices = ""
    for i, choice in enumerate(item["options"].split("\n")):
        if i > 0:
            choices += "  "
        choices += f"({i + 1}) {choice}"
        if choice == item["answer"]:
            answer_idx = i + 1
    prompt = question_template.format(
        question=item["question"],
        choices=choices
    )
    if has_answer:
        prompt += f"({answer_idx}) {item['answer']}。\n\n"
    return prompt

def load_fewshot_prompt(data, config, language):
    fewshot_prompt = ""
    for item in data[:config["num_fewshots"]]:
        fewshot_prompt += format_question(item, config[f"question_template_{language}"], True)
    return fewshot_prompt


def load_fewshot_data(train_data_path):
    chn_fewshot_fn = os.path.join(train_data_path, os.listdir(train_data_path)[0])
    data = read_file(chn_fewshot_fn)
    return data


def load_data_xiezhi(args):
    xiezhi_path = os.path.join(args.data_path, "tasks", "xiezhi")
    task_config = args.tasks_config["xiezhi"]
    task_data = {}

    chn_fewshot_path = os.path.join(xiezhi_path, "train", "xiezhi_train_chn")
    chn_fewshot_data = load_fewshot_data(chn_fewshot_path)
    eng_fewshot_path = os.path.join(xiezhi_path, "train", "xiezhi_train_eng")
    eng_fewshot_data = load_fewshot_data(eng_fewshot_path)

    for subject in task_config["subjects"]:
        task_data[subject] = []
        subject_path = os.path.join(xiezhi_path, "test", subject)
        subject_fn = os.path.join(subject_path, os.listdir(subject_path)[0])
        subject_data = read_file(subject_fn, task_config["limit"] if "limit" in task_config else None)
        language = subject[-3:]
        if language == "chn":
            fewshot_prompt = load_fewshot_prompt(chn_fewshot_data, task_config, language)
        else:
            fewshot_prompt = load_fewshot_prompt(eng_fewshot_data, task_config, language)
        for item in subject_data:
            task_data[subject].append({
                **item,
                "instruction": task_config[f"instruction_template_{language}"],
                "fewshot_prompt": fewshot_prompt,
                "prompt_round1": format_question(item, task_config[f"question_template_{language}"])
            })
    
    return task_data