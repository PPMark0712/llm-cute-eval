import os, json


def format_query_math(question_template, data):
    prompt = question_template.format(
        question=data["problem"],
    )
    return prompt


def load_file_math(fn, limit=None):
    data = []
    with open(fn, "r") as f:
        for line in f:
            d = json.loads(line)
            data.append(d)
            if limit and len(data) >= limit:
                break
    return data


def get_fewshot_prompt_math(math_path, num_fewshots):
    assert 0 <= num_fewshots <= 5
    fewshot_prompt = ""
    fewshot_fn = os.path.join(math_path, "fewshot_prompt_math.txt")
    with open(fewshot_fn, "r") as f:
        text = "".join(f.readlines())
    data = text.split("\n\n")
    for item in data[:num_fewshots]:
        fewshot_prompt += item + "\n\n"
    return fewshot_prompt


def load_data_math(args):
    math_path = os.path.join(args.data_path, "tasks", "math")
    task_config = args.tasks_config["math"]
    question_template = task_config["question_template"]
    task_data = {}
    fn = os.path.join(math_path, "test.jsonl")
    test_data = load_file_math(fn, task_config["limit"])
    fewshot_prompt = get_fewshot_prompt_math(math_path, task_config["num_fewshots"])
    data = []
    for item in test_data:
        prompt = format_query_math(question_template, item)
        data.append({
            **item,
            "instruction": task_config["instruction"],
            "fewshot_prompt": fewshot_prompt,
            "prompt_round1": prompt,
        })
    task_data = {"math": data}
    return task_data