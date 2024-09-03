import os, json

<<<<<<< HEAD:lm_cute_eval/tasks/hellaswag/load_data_hellaswag.py

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
=======
drop_dir = os.path.join("data", "tasks", "drop")
drop_instruction = "You will be asked to read a passage and answer a question. Some examples of passages and Q&A are provided below.Think step by step, then write a line of the form 'Answer: $ANSWER' at the end of your response."
>>>>>>> 3be6db8e48b08ec47f89e426baf512f8c5dc06a0:code/tasks/drop/load_data_drop.py


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


<<<<<<< HEAD:lm_cute_eval/tasks/hellaswag/load_data_hellaswag.py
def get_fewshot_prompt_hellaswag(hellaswag_dir, num_fewshot):
    assert 0 <= num_fewshot <= 25
=======
def get_fewshot_cot_prompt_drop(num_fewshot):
    assert 0 <= num_fewshot <= 8
    fewshot_cot_fn = os.path.join(drop_dir, "fewshot.txt")
    file_str = ""
    with open(fewshot_cot_fn, "r") as f:
        for line in f:
            file_str += line
>>>>>>> 3be6db8e48b08ec47f89e426baf512f8c5dc06a0:code/tasks/drop/load_data_drop.py
    fewshot_prompt = ""
    lst = file_str.split("\n\n")
    for text in lst[:num_fewshot]:
        fewshot_prompt += text + "\n\n"
    return fewshot_prompt


<<<<<<< HEAD:lm_cute_eval/tasks/hellaswag/load_data_hellaswag.py
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
=======
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
>>>>>>> 3be6db8e48b08ec47f89e426baf512f8c5dc06a0:code/tasks/drop/load_data_drop.py
            "fewshot_prompt": fewshot_prompt,
            "prompt_round1": prompt,
        })
    return task_data
