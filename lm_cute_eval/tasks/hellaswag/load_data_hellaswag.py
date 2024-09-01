import os, json


def format_query_hellaswag(data, has_answer):
    question_template = "Question: {question}\nOptions:\n(A) {A}\n(B) {B}\n(C) {C}\n(D) {D}\nAnswer: The most appropriate continuation is "

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


def get_fewshot_prompt_hellaswag(hellaswag_dir, num_fewshot):
    assert 0 <= num_fewshot <= 25
    fewshot_prompt = ""
    fewshot_fn = os.path.join(hellaswag_dir, "hellaswag_train_sampled25.jsonl")
    fewshot_data = load_file_hellaswag(fewshot_fn, num_fewshot)
    for item in fewshot_data:
        fewshot_prompt += format_query_hellaswag(item, True)
        
    return fewshot_prompt


def load_data_hellaswag(args):
    hellaswag_dir = os.path.join(args.data_path, "tasks", "hellaswag")
    hellaswag_instruction = "Here are some multiple-choice questions about continuation writing. Each question contains a paragraph and four options for possible continuations. Choose the most appropriate continuation from options A, B, C, and D.\n\n\n"

    task_config = args.tasks_config["hellaswag"]
    task_data = {}
    test_fn = os.path.join(hellaswag_dir, "hellaswag.jsonl")
    test_data = load_file_hellaswag(test_fn, task_config["limit"])
    fewshot_prompt = get_fewshot_prompt_hellaswag(hellaswag_dir, task_config["num_fewshot"])
    data = []
    for item in test_data:
        prompt = format_query_hellaswag(item, False)
        data.append({
            **item,
            "instruction": hellaswag_instruction,
            "fewshot_prompt": fewshot_prompt,
            "prompt_round1": prompt,
        })
    task_data = {"hellaswag": data}
    return task_data
