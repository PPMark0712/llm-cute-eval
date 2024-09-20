import os, csv, json



def format_query_mmlu(data, question_template):
#     question_template = """
# Human:
# Q: {Q}
# Which one of the four choices is correct, (A), (B), (C) or (D)?
# Choices: 
# (A) {A}
# (B) {B}
# (C) {C}
# (D) {D}

# Assistant:
# Let's think step by step. 
# A: """
    prompt = question_template.format(
        Q=data["Q"],
        A=data["A"],
        B=data["B"],
        C=data["C"],
        D=data["D"]
    )
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


def get_inst_and_fewshot_cot(fewshot_data, subject):
    text = fewshot_data[subject].strip()
    idx = text.find("Q:")
    instruction = text[:idx]
    fewshot_cot_prompt = text[idx:]
    return instruction, fewshot_cot_prompt


def load_data_mmlu(args):
    mmlu_dir = os.path.join(args.data_path, "tasks", "mmlu")
    task_config = args.tasks_config["mmlu"]
    fewshot_fn = os.path.join(mmlu_dir, "fewshot-cot", "mmlu-cot-claude-multiple.json")
    with open(fewshot_fn, "r") as f:
        fewshot_data = json.load(f)
    task_data = {}
    subjects = task_config["subjects"]
    for subject in subjects:
        fn = os.path.join(mmlu_dir, "test", f"{subject}_test.csv")
        subject_data = load_file_mmlu(fn, task_config["limit"])
        instruction, fewshot_cot_prompt = get_inst_and_fewshot_cot(fewshot_data, subject)
        task_data[subject] = []
        for item in subject_data:
            prompt = format_query_mmlu(item, task_config["question_template"])
            task_data[subject].append({
                **item,
                "instruction": instruction,
                "fewshot_prompt": fewshot_cot_prompt,
                "prompt_round1": prompt,
            })
    return task_data
