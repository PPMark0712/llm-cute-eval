import os, json

def read_file(fn, limit=None):
    with open(fn, "r") as f:
        data = json.load(f)
    if limit and len(data) > limit:
        data = data[:limit]
    return data


def get_fewshot_prompt(examples):
    # fewshot_prompt = "<examples>\n"
    # for i, example in enumerate(examples):
    #     fewshot_prompt += f"<example_{i + 1}>\n"
    #     fewshot_prompt += f"Input:\n<input>\n\n{example['input']}\n\n</input>\n\nOutput:\n<output>\n\n{example['output']}\n\n</output>\n\n"
    #     fewshot_prompt += f"</example_{i + 1}>\n"
    # fewshot_prompt += "</examples>\n\nHere comes the official input, please follow the format above to answer me.\n\n"
    
    fewshot_prompt = "Examples:\n"
    for i, example in enumerate(examples):
        fewshot_prompt += f"example {i + 1}:\n"
        fewshot_prompt += f"Input:\n{example['input']}\n\nOutput:\n<output>\n\n{example['output']}\n\n</output>\n\n"
    fewshot_prompt += "Here comes the official input, please follow the format above to answer me.\n\n"
    return fewshot_prompt

def load_data_iclformat(args):
    iclbench_path = os.path.join(args.data_path, "tasks", "iclformat")
    task_config = args.tasks_config["iclformat"]
    task_data = {}
    overall_inst = task_config["instruction"]["default"]
    for subject in task_config["subjects"]:
        inst = overall_inst + "\n\n\n" + task_config["instruction"][subject] + "\n\n\n"
        task_data[subject] = []
        fn = os.path.join(iclbench_path, f"{subject}.json")
        data = read_file(fn, task_config["limit"])
        for item in data:
            examples = item["examples"]
            fewshot_prompt = get_fewshot_prompt(examples)
            task_data[subject].append({
                **item,
                "instruction": inst,
                "fewshot_prompt": fewshot_prompt,
                # "prompt_round1": "Input:\n<input>\n\n" + item["input"] + "\n\n</input>\n\n",
                "prompt_round1": "Input:\n" + item["input"] + "\n\n",
            })

    return task_data