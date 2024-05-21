import os, csv, json
from ...utils import TASKS_SUBJECTS

mmlu_dir = os.path.join("data", "tasks", "mmlu")

question_template = """
Human:
Q: {Q}
Which one of the four choices is correct, (A), (B), (C) or (D)?
Choices: 
(A) {A}
(B) {B}
(C) {C}
(D) {D}

Assistant:
Let's think step by step. 
A:"""

def format_query_mmlu(data):
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


def load_data_mmlu(args):
    task_config = args.tasks_config["mmlu"]
    fewshot_fn = os.path.join(mmlu_dir, "fewshot-cot", "mmlu-cot-claude-multiple.json")
    with open(fewshot_fn, "r") as f:
        fewshot_data = json.load(f)
    task_data = {}
    if "subjects" not in task_config:
        task_config["subjects"] = TASKS_SUBJECTS["mmlu"]
    subjects = task_config["subjects"]
    for subject in subjects:
        fn = os.path.join(mmlu_dir, "test", f"{subject}_test.csv")
        subject_data = load_file_mmlu(fn, task_config["limit"])
        fewshot_prompt = fewshot_data[subject].strip()
        task_data[subject] = []
        for item in subject_data:
            prompt = fewshot_prompt + "\n\n\n" + format_query_mmlu(item)
            task_data[subject].append({
                **item,
                "prompt_round1": prompt,
            })
    return task_data
