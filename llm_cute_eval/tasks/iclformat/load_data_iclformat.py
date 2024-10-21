import os, json

def read_file(fn, limit=None):
    with open(fn, "r") as f:
        data = json.load(f)
    if limit and len(data) > limit:
        data = data[:limit]
    return data


def load_data_iclformat(args):
    iclbench_path = os.path.join(args.data_path, "tasks", "iclformat")
    task_config = args.tasks_config["iclformat"]
    task_data = {}
    for subject in task_config["subjects"]:
        task_data[subject] = []
        fn = os.path.join(iclbench_path, f"{subject}.json")
        data = read_file(fn, task_config["limit"])
        for item in data:
            fewshot_prompt = ""
            for example in item["examples"]:
                fewshot_prompt += example
            task_data[subject].append({
                **item,
                "instruction": task_config["instruction"],
                "fewshot_prompt": fewshot_prompt,
                "prompt_round1": item["input"],
            })

    return task_data