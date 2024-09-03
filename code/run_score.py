import datetime
import json
import argparse
import os
from collections import defaultdict
import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams
import os
import json
from argparse import Namespace
from FlagEmbedding import BGEM3FlagModel
from .model import init_vllm_model
from .utils import TASK_LIST, MODEL_FORMAT, LOAD_TASK_DATA, MATCH_TASK_ANSWER,TASKS_SUBJECTS



def load_args_from_config(config_path: str):
    """
    从配置文件加载参数。

    :param config_path: 配置文件的路径
    :return: 包含参数的 Namespace 对象
    """
    # 读取配置文件
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    args = argparse.Namespace()

    # 遍历字典，将值赋给Namespace对象的属性
    for key, value in config_dict.items():
        setattr(args, key, value)
        
    # init task config
    if "all" in args.tasks:
        args.tasks = TASK_LIST
    
    with open(args.config_path, "r") as f:
        args.tasks_config = json.load(f)
    for task in args.tasks:
        try:
            with open(f"code/tasks/{task}/config_{task}.json", "r") as f:
                default_task_config =  json.load(f)
            if task not in args.tasks_config:
                args.tasks_config[task] = default_task_config
            for k, v in default_task_config.items():
                if k not in args.tasks_config[task]:
                    args.tasks_config[task][k] = v
        except FileNotFoundError:
            pass
    if "all" in args.tasks or "mmlu"in args.tasks:
        if "subjects" not in args.tasks_config["mmlu"]:
            args.tasks_config["mmlu"]["subjects"] = TASKS_SUBJECTS["mmlu"]
    return args

def get_tasks_data(args):
    """
    return:
        tasks_data: Dict[task(str), Dict[subject(str), List[item(dict)]]]
    """
    tasks_data = defaultdict(list)
    for task in tqdm(args.tasks, desc="load task data"):
        tasks_data[task] = LOAD_TASK_DATA[task](args)
    return tasks_data


def run_infer(tasks_data:dict, model:LLM, sampling_params:SamplingParams, args):
    """
    params:
        tasks_data: Dict[task(str), Dict[subject(str), List[item(dict)]]]

    returns:
        infer_result: dict[task(str), dict[subject(str), item(dict)]]
    """
    infer_result = dict(tasks_data)
    for round_idx in range(1, args.rounds + 1):
        print(f"running infer round {round_idx}")
        # get all prompts
        prompts = []
        for task in tasks_data:
            for subject in tasks_data[task]:
                for item in tasks_data[task][subject]:
                    if round_idx == 1:
                        prompt = item["instruction"] + item["fewshot_prompt"] + item["prompt_round1"]
                    else:                    
                        history = []
                        history.append((item[f"prompt_round{1}"], item[f"infer_round{round_idx-1}"]))
                        query = item[f"prompt_round{round_idx}"]    
                        prompt = MODEL_FORMAT[args.model_type](query, history)
                    prompts.append(prompt)

        outputs = model.generate(prompts, sampling_params)
        generated_texts = [output.outputs[0].text for output in outputs]

        if args.save_infer_results:
            with open(f"{args.save_path}/infer_round{round_idx}.txt", "w") as f:
                for x, y in zip(prompts, generated_texts):
                    print("="*20, file=f)
                    print(x, file=f)
                    print("-"*20, file=f)
                    print(y, file=f)
        
        # save infer result in this round
        cur_infer_idx = 0
        for task in tasks_data:
            for subject in tasks_data[task]:
                for item in tasks_data[task][subject]:
                    item[f"infer_round{round_idx}"] = generated_texts[cur_infer_idx]
                    cur_infer_idx += 1

        # prepare prompt for next round
        if round_idx == args.rounds:
            break
        for task in tasks_data:
            for subject in tasks_data[task]:
                for item in tasks_data[task][subject]:
                    item[f"prompt_round{round_idx + 1}"] = args.refine_prompt
    
    return infer_result


def run_eval(infer_results, args):
    result = defaultdict(dict)
    for round_idx in range(1, args.rounds + 1):
        result[f"round{round_idx}"] = {}
        print(args.tasks)
        if "xsum" in args.tasks:
            torch.cuda.empty_cache()
            model = BGEM3FlagModel('/data1/dcy/downloads/model/BAAI/bge-m3', use_fp16=True)
            args.model = model
        for task in args.tasks:
            print(task)
            result[f"round{round_idx}"][task] = MATCH_TASK_ANSWER[task](infer_results[task], round_idx, args)
    return result

def save_result_inference(infer_result:dict, args):
    """
        infer_result: dict[task(str), dict[subject(str), item(dict)]]
        score: dict[round{i}(str), dict[task(str), dict[subject(str), item(dict)]]]
    """
    # save infer results in file
    if args.save_infer_results:
        infer_result_path = os.path.join(args.save_path, "infer_results_withoutscore")
        os.makedirs(infer_result_path, exist_ok=True)
        for task in infer_result:
            task_path = os.path.join(infer_result_path, task)
            os.makedirs(task_path, exist_ok=True)
            for subject in infer_result[task]:
                subject_filename = os.path.join(task_path, f"{subject}.json")
                with open(subject_filename, "w") as f:
                    json.dump(infer_result[task][subject], f, ensure_ascii=False, indent=4)



def save_result(infer_result:dict, score:dict, args):
    """
        infer_result: dict[task(str), dict[subject(str), item(dict)]]
        score: dict[round{i}(str), dict[task(str), dict[subject(str), item(dict)]]]
    """
    # save infer results in file
    print("save infer results in file")
    if args.save_infer_results:
        infer_result_path = os.path.join(args.save_path, "infer_results")
        os.makedirs(infer_result_path, exist_ok=True)
        for task in infer_result:
            print("task")
            task_path = os.path.join(infer_result_path, task)
            os.makedirs(task_path, exist_ok=True)
            for subject in infer_result[task]:
                subject_filename = os.path.join(task_path, f"{subject}.json")
                with open(subject_filename, "w") as f:
                    json.dump(infer_result[task][subject], f, ensure_ascii=False, indent=4)

    # save evaluation result
    summary_score = {}
    for task in tqdm(args.tasks, desc="save evaluation result"):
        for subject in score["round1"][task].keys():
            subject_result_path = os.path.join(args.save_path, "eval_result", task)
            subject_result = {}
            for round_idx in range(1, args.rounds + 1):
                subject_result[f"round{round_idx}"] = score[f"round{round_idx}"][task][subject]
            os.makedirs(subject_result_path, exist_ok=True)
            fn = os.path.join(subject_result_path, f"{subject}.json")
            with open(fn, "w") as f:
                json.dump(subject_result, f, indent=4)
        if args.rounds == 1:
            task_result = score[f"round1"][task][task]
        else:
            task_result = {f"round{round_idx}": score[f"round{round_idx}"][task][task] for round_idx in range(1, args.rounds + 1)}
        summary_score[task] = task_result

    with open(os.path.join(args.save_path, "summary.json"), "w") as f:
        json.dump(summary_score, f, indent=4)
    print(json.dumps(summary_score, indent=4))

def load_inference_results(infer_result_path: str):
    """
    从指定目录加载推断结果到字典中。

    :param infer_result_path: 包含推断结果文件的目录路径
    :return: 包含推断结果的字典
    """
    infer_result = {}
    # 遍历目录中的所有文件和文件夹
    for root, dirs, files in os.walk(infer_result_path):
        for file in files:
            # 构建文件的完整路径
            file_path = os.path.join(root, file)
            # 检查文件是否是JSON文件
            if file.endswith('.json'):
                # 从文件名中提取任务和主题
                task = os.path.basename(root)
                subject = os.path.splitext(file)[0]
                # 打开并读取JSON文件
                with open(file_path, 'r', encoding='utf-8') as f:
                    item = json.load(f)
                    # 将读取的数据添加到字典中
                    if task not in infer_result:
                        infer_result[task] = {}
                    infer_result[task][subject] = item
    return infer_result


def main():
    load_path = "/data1/dcy/projects/evaluate/lm-cute-eval/output/7-25_13:09_Llama-3_dpo_1"
    # load_path = "/data1/dcy/projects/evaluate/lm-cute-eval/output/5-25_02:21_llama3_gen"
    # load_path = "/data1/dcy/projects/evaluate/lm-cute-eval/output/5-25_02:38_llama3_gen"
    # load_path = "/data1/dcy/projects/evaluate/lm-cute-eval/output/5-25_02:39_llama3_gen"
    config_path = os.path.join(load_path, "config.json")
    result_path = os.path.join(load_path, "infer_results_withoutscore")
    args = load_args_from_config(config_path)
    print(args)
    inference_result = load_inference_results(result_path)
    score = run_eval(inference_result, args)
    save_result(inference_result, score, args)
    

if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()
