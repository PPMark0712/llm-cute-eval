import os, json

icleval_dir = os.path.join("data", "tasks", "icleval")


def load_data(fn, limit):
    with open(fn, "r") as f:
        data = json.load(f)
    if limit and limit < len(data):
        data = data[:limit]
    return data


def load_data_icleval(args):
    task_config = args.tasks_config["icleval"]
    task_data = {}
    for subject in task_config["subjects"]:
        subject_fn = os.path.join(icleval_dir, f"{subject}.json")
        subject_data = load_data(subject_fn, task_config["limit"])
        data = []
        # if subject == "copy_dict_search_string":
        #     for item in subject_data:
        #         fewshot_prompt = ""
        #         for k, v in item["dict"].items():
        #             fewshot_prompt += f"\"{k}\": \"{v}\"\n"
        #         data.append({
        #             **item,
        #             "instruction": "",
        #             "fewshot_prompt": fewshot_prompt,
        #             "prompt_round1": f"\"{item["prompt"]}\": ",
        #         })
        # else:
        for item in subject_data:
            data.append({
                **item,
                "instruction": "",
                "fewshot_prompt": item["examples"],
                "prompt_round1": item["prompt"],
            })
        task_data.update({subject: data})
    
    return task_data
