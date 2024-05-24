import os, json

drop_dir = os.path.join("data", "tasks", "drop")
drop_instruction = "You will be asked to read a passage and answer a question. Some examples of passages and Q&A are provided below.Think step by step, then write a line of the form 'Answer: $ANSWER' at the end of your response."


def load_file_drop(fn, limit=0):
    data = []
    with open(fn, "r", encoding='utf-8') as f:
        
        for line in f:
            try:
                # 尝试解析JSON
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                # 如果解析失败，打印错误信息并跳过当前行
                print(f"Skipping line with JSONDecodeError: {e}")
                continue  # 继续读取下一行
            if limit and len(data) >= limit:
                break
    return data


def get_fewshot_cot_prompt_drop(num_fewshot):
    assert 0 <= num_fewshot <= 8
    fewshot_cot_fn = os.path.join(drop_dir, "fewshot.txt")
    file_str = ""
    with open(fewshot_cot_fn, "r") as f:
        for line in f:
            file_str += line
    fewshot_prompt = ""
    lst = file_str.split("\n\n")
    for text in lst[:num_fewshot]:
        fewshot_prompt += text + "\n\n"
    return fewshot_prompt


def load_data_drop(args):
    task_config = args.tasks_config["drop"]
    test_data = load_file_drop(os.path.join(drop_dir, "test.jsonl"), task_config["limit"])
    task_data = {"drop": []}
    fewshot_prompt = get_fewshot_cot_prompt_drop(task_config["num_fewshot"])
    for item in test_data:
        prompt = "Question: " + item["question"] + "\nAnswer: Let's think step by step\n"
        task_data["drop"].append({
            **item, 
            "instruction": drop_instruction,
            "fewshot_prompt": fewshot_prompt,
            "prompt_round1": prompt,
        })
    return task_data
