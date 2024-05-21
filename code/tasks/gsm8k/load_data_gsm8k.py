import os, json

gsm8k_dir = os.path.join("data", "tasks", "gsm8k")
gsm8k_inst = "Solve the following math questions. Please think step by step and finally give the answer.\n\n"

def load_file_gsm8k(fn, limit=0):
    data = []
    with open(fn, "r") as f:
        for line in f:
            data.append(json.loads(line))
            if limit and len(data) >= limit:
                break
    return data


# def get_fewshot_prompt_gsm8k(train_data, begin_idx, num_fewshot):
#     if num_fewshot == 0:
#         return ""
#     fewshot_prompt = ""
#     for i in range(begin_idx, begin_idx + num_fewshot):
#         item = train_data[i % len(train_data)]
#         fewshot_prompt += "Question: " + item["question"] + "\nAnswer: " + item["answer"] + "\n\n"
#     return fewshot_prompt


def get_fewshot_cot_prompt_gsm8k(num_fewshot):
    assert 0 <= num_fewshot <= 8
    fewshot_cot_fn = os.path.join(gsm8k_dir, "fewshot_cot.txt")
    file_str = ""
    with open(fewshot_cot_fn, "r") as f:
        for line in f:
            file_str += line
    fewshot_prompt = ""
    lst = file_str.split("\n\n")
    for text in lst[:num_fewshot]:
        fewshot_prompt += text + "\n\n"
    return fewshot_prompt



def load_data_gsm8k(args):
    task_config = args.tasks_config["gsm8k"]
    test_data = load_file_gsm8k(os.path.join(gsm8k_dir, "test.jsonl"), task_config["limit"])
    task_data = {"gsm8k": []}
    fewshot_prompt = get_fewshot_cot_prompt_gsm8k(task_config["num_fewshot"])
    for item in test_data:
        prompt = gsm8k_inst + fewshot_prompt + "Question: " + item["question"] + "\nAnswer: Let's think step by step\n"
        task_data["gsm8k"].append({**item, "prompt_round1": prompt})
    return task_data
