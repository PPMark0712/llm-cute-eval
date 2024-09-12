import os, json


def load_data(fn, limit):
    with open(fn, "r") as f:
        data = json.load(f)
    if limit and limit < len(data):
        data = data[:limit]
    return data


def load_data_icleval(args):
    icleval_dir = os.path.join(args.data_path, "tasks", "icleval")
    task_config = args.tasks_config["icleval"]
    task_data = {}
    for subject in task_config["subjects"]:
        subject_fn = os.path.join(icleval_dir, f"{subject}.json")
        subject_data = load_data(subject_fn, task_config["limit"])
        data = []
        for item in subject_data:
            instruction = ""
            fewshot_prompt = ""
            prompt = item["prompt"]
            if subject == "copy_dict_search_string":
                for k, v in item["dict"].items():
                    fewshot_prompt += f"\"{k}\": \"{v}\"\n"
            elif subject == "copy_natural_language_string":
                fewshot_prompt = item["content"]
            else:
                fewshot_prompt = item["examples"]
            data.append({
                **item,
                "instruction": instruction,
                "fewshot_prompt": fewshot_prompt,
                "prompt_round1": prompt,
            })
        
        task_data.update({subject: data})
    
    return task_data
