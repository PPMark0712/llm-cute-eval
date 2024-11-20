import os, json
from datasets import load_dataset


def format_query_mmluproplus(item):
    prompt = f"Question: {item['question']}\nOptions:\n"
    for c, choice in zip("ABCDEFGHIJKLMN", item["options"]):
        prompt += f"{c}: {choice}\n"
    prompt += "Answer: "
    return prompt


def load_mmlu_pro_p(data_path):
    dataset_orig = load_dataset("parquet", data_files={"validation": os.path.join(data_path, "mmlupro_validation.parquet")})
    dataset_modified = load_dataset("parquet", data_files={"test": os.path.join(data_path, "mmluproplus.parquet")})
    test_df, val_df = dataset_modified["test"], dataset_orig["validation"]

    # Add logging for modification types
    modification_counts = {
        "two_wrong": 0,
        "correct_and_wrong": 0,
        "llm_modified": 0,
        "none": 0
    }

    for item in test_df:
        if item.get('is_modified') == True:
            modification_counts["llm_modified"] += 1
        elif item.get('is_modified_non_llm') == True:
            mod_type = item.get('modification_type_non_llm')
            if mod_type in modification_counts:
                modification_counts[mod_type] += 1
            else:
                print(f"Unexpected modification type: {mod_type}")
        else:
            modification_counts["none"] += 1

    # print("Modification type counts:")
    # print(json.dumps(modification_counts, indent=2))

    test_df = preprocess(test_df)
    val_df = preprocess(val_df)
    return test_df, val_df


def preprocess(test_df):
    res_df = []
    for each in test_df:
        options = []
        for opt in each["options"]:
            if opt == "N/A":
                continue
            options.append(opt)
        each["options"] = options
        # Preserve modification information
        if 'is_modified' in each:
            each['is_modified'] = each['is_modified']
        if 'is_modified_non_llm' in each:
            each['is_modified_non_llm'] = each['is_modified_non_llm']
        if 'modification_type_non_llm' in each:
            each['modification_type_non_llm'] = each['modification_type_non_llm']
        res_df.append(each)
    res = {}
    for each in res_df:
        if each["category"] not in res:
            res[each["category"]] = []
        res[each["category"]].append(each)
    return res


def load_data_mmluproplus(args):
    mmluproplus_path = os.path.join(args.data_path, "tasks", "mmluproplus")
    task_config = args.tasks_config["mmluproplus"]
    test_df, val_df = load_mmlu_pro_p(mmluproplus_path)
    task_data = {}
    for subject in test_df.keys():
        task_data[subject] = []
        if task_config.get("limit", 0):
            test_df[subject] = test_df[subject][:task_config["limit"]]
        for item in test_df[subject]:
            task_data[subject].append({
                **item,
                "instruction": task_config["instruction_template"].format(subject=subject),
                "fewshot_prompt": "",
                "prompt_round1": format_query_mmluproplus(item),
            })
    return task_data


if __name__ == "__main__":
    data_path = "/data1/yyz/projects/evaluate/llm-cute-eval/data/tasks/mmluproplus"
    test_df, val_df = load_mmlu_pro_p(data_path)
    print(test_df.keys())
    # print(json.dumps(test_df["law"][0], indent=4))