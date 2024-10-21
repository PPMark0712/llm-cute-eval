import os, json


def format_query_cfinbench(item, config, q_type, has_answer=False):
    question_template = config[f"question_template_{q_type}"]
    if q_type == "judgment":
        prompt = question_template.format(
            Q=item["text"],
        )
    elif q_type in ["multi_choice", "single_choice"]:
        choices = ""
        for choice in item["OptionList"]:
            choices += choice + "  "
        prompt = question_template.format(
            Q=item["text"],
            choices=choices
        )
    if has_answer:
        prompt += f"{item['Answer']}。\n\n"
    return prompt


def load_file_cfinbench(fn, limit=None):
    data = []
    with open(fn, "r") as f:
        for line in f:
            data.append(json.loads(line))
            if limit and len(data) >= limit:
                break
    return data


def get_fewshot_prompt(data, config, q_type):
    fewshot_prompt = ""
    for item in data:
        fewshot_prompt += format_query_cfinbench(item, config, q_type, True)
    return fewshot_prompt


def load_data_cfinbench(args):
    cfinbench_path = os.path.join(args.data_path, "tasks", "cfinbench")
    task_config = args.tasks_config["cfinbench"]
    task_data = {}
    subject_dict = task_config["subjects"]
    q_type_list = ["judgment", "multi_choice", "single_choice"]

    for subject, subject_file_list in subject_dict.items():
        task_data[subject] = []
        for q_type in q_type_list:
            subject_val_path = os.path.join(cfinbench_path, "val", q_type)
            subject_dev_path = os.path.join(cfinbench_path, "dev", q_type)

            for file in subject_file_list:
                fewshot_fn = os.path.join(subject_dev_path, file)
                fewshot_data = load_file_cfinbench(fewshot_fn, task_config["num_fewshots"])
                fewshot_prompt = get_fewshot_prompt(fewshot_data, task_config, q_type)

                fn = os.path.join(subject_val_path, file)
                subject_data = load_file_cfinbench(fn, task_config["limit"])

                for item in subject_data:
                    prompt = format_query_cfinbench(item, task_config, q_type)
                    task_data[subject].append({
                        **item,
                        "q_type": q_type,
                        "instruction": task_config[f"instruction_{q_type}"],
                        "fewshot_prompt": fewshot_prompt,
                        "prompt_round1": prompt,
                    })

    return task_data