import datetime
import json
import argparse
import os
from collections import defaultdict
import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams
import os

from .model import init_vllm_model
from .utils import TASK_LIST, MODEL_FORMAT, LOAD_TASK_DATA, MATCH_TASK_ANSWER


def initialize(args):
    # init args
    if args.sampling_params:
        args.sampling_params = json.loads(args.sampling_params)
    
    # init task config
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
    
    t = datetime.datetime.now()
    args.save_path = os.path.join(args.output_path, f"{t.month}-{t.day}_{t.hour:02d}:{t.minute:02d}_{args.save_name}")
    
    os.makedirs(os.path.join(args.save_path, args.temp_file_path), exist_ok=True)

    # save config
    os.makedirs(args.save_path, exist_ok=True)
    with open(f"{args.save_path}/config.json", "w") as f:
        run_config = {**vars(args)}
        json.dump(run_config, f, indent=4)


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
                    history = []
                    for i in range(1, round_idx):
                        history.append((item[f"prompt_round{i}"], item[f"infer_round{i}"]))
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
        for task in args.tasks:
            result[f"round{round_idx}"][task] = MATCH_TASK_ANSWER[task](infer_results[task], round_idx, args)
    return result


def save_result(infer_result:dict, score:dict, args):
    """
        infer_result: dict[task(str), dict[subject(str), item(dict)]]
        score: dict[round{i}(str), dict[task(str), dict[subject(str), item(dict)]]]
    """
    # save infer results in file
    if args.save_infer_results:
        infer_result_path = os.path.join(args.save_path, "infer_results")
        os.makedirs(infer_result_path, exist_ok=True)
        for task in infer_result:
            task_path = os.path.join(infer_result_path, task)
            os.makedirs(task_path, exist_ok=True)
            for subject in infer_result[task]:
                subject_filename = os.path.join(task_path, f"{subject}.json")
                with open(subject_filename, "w") as f:
                    json.dump(infer_result[task][subject], f, ensure_ascii=False, indent=4)

    # save evaluation result
    summary_score = {}
    for task in args.tasks:
        for subject in score[task]:
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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--model_type", type=str, default="default")
    parser.add_argument("--data_path", type=str, default="data")
    parser.add_argument("--config_path", type=str, default="config.json")
    parser.add_argument("--tasks", type=str, nargs="+")
    parser.add_argument("--output_path", type=str, default="output")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--save_name", type=str)
    parser.add_argument("--save_infer_results", action="store_true")
    parser.add_argument("--sampling_params", type=str, default=None)
    parser.add_argument("--refine_prompt", type=str, default="Please further think about and give me a more precise and professional answer.\nThe answer is ")
    parser.add_argument("--temp_file_path", type=str, default="temp_file")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    initialize(args)
    tasks_data = get_tasks_data(args)
    model, sampling_params = init_vllm_model(args)
    inference_result = run_infer(tasks_data, model, sampling_params, args)
    score = run_eval(inference_result, args)
    save_result(inference_result, score, args)
    

if __name__ == "__main__":
    print("222")
    main()
