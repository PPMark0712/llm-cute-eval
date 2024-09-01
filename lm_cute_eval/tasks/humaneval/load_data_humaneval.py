import os
from .human_eval.data import read_problems


def get_fewshot_prompt(humaneval_dir):
    fewshot_fn = os.path.join(humaneval_dir, "fewshot_prompt.txt")
    fewshot_prompt = ""
    with open(fewshot_fn, "r") as f:
        for line in f:
            fewshot_prompt += line
    return fewshot_prompt

def format_humaneval_prompt(question:str):
    prompt = "Question:\n" + "```python\n" + question.strip() + "```\n\nAnswer:\n"
    return prompt


def load_data_humaneval(args):
    humaneval_instruction = "Please complete the following python functions and output the entire function within a python code block, without any explainations.\n\n\n"
    humaneval_dir = os.path.join(args.data_path, "tasks", "humaneval")
    task_config = args.tasks_config["humaneval"]
    data = read_problems(os.path.join(humaneval_dir, "HumanEval.jsonl.gz"))
    task_data = {"humaneval": []}
    fewshot_prompt = get_fewshot_prompt(humaneval_dir)
    for humaneval_id, item in data.items():
        task_data["humaneval"].append({
            **item, 
            "instruction": humaneval_instruction,
            "fewshot_prompt": fewshot_prompt,
            "prompt_round1": format_humaneval_prompt(item["prompt"])
        })
        if task_config["limit"] and len(task_data["humaneval"]) >= task_config["limit"]:
            break
    return task_data