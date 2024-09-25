import os, json

def format_arc_query(data, has_answer):
    question_template = "{question}\n{A}{B}{C}{D}\n"
    prompt = "Question: " + question_template.format(
        question=data["Q"],
        A=data["A"],
        B=data["B"],
        C=data["C"],
        D=data["D"]
    ) + "Answer: The answer is "
    if has_answer:
        prompt += f"({data['ans']})\n\n\n"
    return prompt


def load_file_arc(fn, limit=None):
    data = []
    with open(fn, "r") as f:
        for line in f:
            d = json.loads(line)
            choices = ["A", "B", "C", "D"]
            question = {"Q": d["question"]["stem"]}
            for i in range(4):
                if i < len(d["question"]["choices"]):
                    question[choices[i]] = f'({choices[i]}) {d["question"]["choices"][i]["text"]}\n'
                else:
                    question[choices[i]] = ""
            question["ans"] = d["answerKey"] if d["answerKey"].isalpha() else "ABCDEFG"[int(d["answerKey"]) - 1]  # sometimes the choices are 1,2,3,4 instead of A,B,C,D
            data.append(question)
            if limit and len(data) >= limit:
                break
    return data


def load_fewshot_data(dev_data, begin_idx, num_fewshots):
    fewshot_prompt = ""
    for i in range(begin_idx, begin_idx + num_fewshots):
        fewshot_prompt += format_arc_query(dev_data[i % len(dev_data)], True)
    return fewshot_prompt


def load_data_arc(args):
    arc_dir = os.path.join(args.data_path, "tasks", "arc")
    arc_instruction = "The following are multiple choice questions about reasoning. Choose A, B, C, or D to answer the questions.\n\n\n"

    task_config = args.tasks_config["arc"]
    task_data = {}
    dev_data = {
        "arc_e": load_file_arc(os.path.join(arc_dir, "ARC-e", "ARC-Easy-Dev.jsonl"), (task_config["arc_e"]["limit"] if task_config["arc_e"]["limit"] else 0) * task_config["arc_e"]["num_fewshots"]),
        "arc_c": load_file_arc(os.path.join(arc_dir, "ARC-c", "ARC-Challenge-Dev.jsonl"), (task_config["arc_c"]["limit"] if task_config["arc_c"]["limit"] else 0) * task_config["arc_c"]["num_fewshots"]),
    }
    
    for subject in task_config["subjects"]:
        if subject == "arc_e":
            fn = os.path.join(arc_dir, "ARC-e", "ARC-Easy-Test.jsonl")
        elif subject == "arc_c":
            fn = os.path.join(arc_dir, "ARC-c", "ARC-Challenge-Test.jsonl")
        subject_data = load_file_arc(fn, task_config[subject]["limit"])
        cur_fewshot_begin_idx = 0
        
        task_data[subject] = []
        for item in subject_data:
            fewshot_prompt = load_fewshot_data(dev_data[subject], cur_fewshot_begin_idx, task_config[subject]["num_fewshots"])
            cur_fewshot_begin_idx += task_config[subject]["num_fewshots"]
            prompt = format_arc_query(item, False)
            task_data[subject].append({
                **item,
                "instruction": arc_instruction,
                "fewshot_prompt": fewshot_prompt,
                "prompt_round1": prompt,
            })
    return task_data
