import os, json


def format_query_winogrande(data, has_answer):
    question_template = "Question: {question}\nOptions:\n(A) {A}\n(B) {B}\nAnswer:"
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


def get_fewshot_prompt_winogrande(winogrande_dir, num_fewshots):
    fewshot_prompt = ""
    fewshot_fn = os.path.join(winogrande_dir, "train_xs.jsonl")
    fewshot_data = load_file_winogrande(fewshot_fn, num_fewshots)
    for item in fewshot_data:
        fewshot_prompt += format_query_winogrande(item, True)
    return fewshot_prompt


def get_fewshot_cot_prompt_winogrande(winogrande_dir):
    fewshot_prompt = ""
    fewshot_fn = os.path.join(winogrande_dir, "fewshot_cot.txt")
    with open(fewshot_fn, "r") as f:
        for line in f:
            fewshot_prompt += line
    return fewshot_prompt

def load_data_winogrande(args):
    winogrande_dir = os.path.join(args.data_path, "tasks", "winogrande")
    winogrande_instruction = "Below are some cloze questions on general knowledge. Choose the most appropriate option from A or B to fill in the blank (_) in the sentence.\n\n\n"

    task_config = args.tasks_config["winogrande"]
    task_data = {}
    test_fn = os.path.join(winogrande_dir, "dev.jsonl")
    test_data = load_file_winogrande(test_fn, task_config["limit"])
    # fewshot_prompt = get_fewshot_prompt_winogrande(winogrande_dir, task_config["num_fewshots"])
    fewshot_cot_prompt = get_fewshot_cot_prompt_winogrande(winogrande_dir)
    data = []
    for item in test_data:
        prompt = format_query_winogrande(item, False) + " Let's think step by step.\n"
        data.append({
            **item,
            "instruction": winogrande_instruction,
            "fewshot_prompt": fewshot_cot_prompt,
            "prompt_round1": prompt,
        })
    task_data = {"winogrande": data}
    return task_data
