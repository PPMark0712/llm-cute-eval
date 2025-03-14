import datetime
import json
import argparse
import os
from collections import defaultdict
from tqdm import tqdm
import os
import logging

from .get_multiround_prompt import get_multiround_prompt
from .model import initialize_model
from .utils import TASK_LIST, MODEL_FORMAT, LOAD_TASK_DATA, MATCH_TASK_ANSWER


def initialize(args):
    t = datetime.datetime.now()
    args.start_time = f"{t.year}-{t.month:02d}-{t.day:02d}_{t.hour:02d}:{t.minute:02d}"
    if args.no_timestamp:
        args.save_path = os.path.join(args.output_path, args.save_name)
    else:
        args.save_path = os.path.join(args.output_path, f"{args.start_time}_{args.save_name}")
    os.makedirs(args.save_path, exist_ok=True)

    # initialize log
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(args.save_path, args.log_fn)),
            logging.StreamHandler(),
        ]
    )
    logging.info(f"log initialized")
    
    if args.use_cpu and args.model_type in ["vllm"]:
        logging.error(f"Error: {args.model_type} can't use cpu!")
        exit()
    
    # init task config
    if "all" in args.tasks:
        args.tasks = TASK_LIST
    
    for task in args.tasks:
        assert task in args.tasks, f"{task} not exists!"
    
    if args.config_path:
        with open(args.config_path, "r", encoding="utf-8") as f:
            args.tasks_config = json.load(f)
    else:
        args.tasks_config = {}
    
    def merge_dicts(d1, d2):
        # 递归地将将d2合并进d1
        for key, value in d2.items():
            if key in d1:
                if isinstance(d1[key], dict) and isinstance(value, dict):
                    merge_dicts(d1[key], value)  # 递归合并子字典
                else:
                    continue  # 如果d1已有此键值，跳过，不做更改
            else:
                d1[key] = value  # 如果d1没有此键值，添加
        return d1
    
    # 将默认config合并到当前config
    for task in args.tasks:
        try:
            default_config_fn = os.path.join("llm_cute_eval", "tasks", task, f"config_{task}.json")
            with open(default_config_fn, "r", encoding="utf-8") as f:
                default_task_config =  json.load(f)
            if task not in args.tasks_config:
                args.tasks_config[task] = default_task_config
            args.tasks_config[task] = merge_dicts(args.tasks_config[task], default_task_config)
        except FileNotFoundError:
            logging.error(f"{task} default config not found!")
            exit()
    args.tasks_config = {key: args.tasks_config[key] for key in args.tasks_config if key in args.tasks}  # 删除不评测的任务的config
    
    os.makedirs(os.path.join(args.save_path, args.temp_file_path), exist_ok=True)

    # save config
    with open(f"{args.save_path}/config.json", "w", encoding="utf-8") as f:
        json.dump({**vars(args)}, f, indent=4, ensure_ascii=False)


def finallize(args):
    os.system(f"rm -r {os.path.join(args.save_path, args.temp_file_path)}")


def get_tasks_data(args):
    """
    return:
        tasks_data: Dict[task(str), Dict[subject(str), List[item(dict)]]]
    """
    tasks_data = defaultdict(list)
    for task in tqdm(args.tasks, desc="load task data"):
        tasks_data[task] = LOAD_TASK_DATA[task](args)
    return tasks_data


def run_infer(tasks_data:dict, model, args):
    """
    params:
        tasks_data: Dict[task(str), Dict[subject(str), List[item(dict)]]]

    returns:
        infer_result: dict[task(str), dict[subject(str), item(dict)]]
    """
    infer_result = dict(tasks_data)
    for round_idx in range(1, args.rounds + 1):
        if args.rounds > 1:
            logging.info(f"running infer round {round_idx}")

        # calculate total data count        
        total_data_cnt = 0
        for task in tasks_data:
            for subject in tasks_data[task]:
                total_data_cnt += len(tasks_data[task][subject])

        # run tasks
        generated_texts = []
        processed_data_cnt = 0
        for task_id, task in enumerate(tasks_data):
            logging.info(f"Running task [{task:^15}], task id: {task_id + 1}/{len(tasks_data)}, processed data: {processed_data_cnt}/{total_data_cnt}")
            if args.use_chat:
                conversations = []
                for subject in tasks_data[task]:
                    processed_data_cnt += len(tasks_data[task][subject])
                    for item in tasks_data[task][subject]:
                        conversation = [{"role": "system", "content": "You are a helpful assistant."}]
                        if round_idx == 1:
                            prompt = item["instruction"] + item["fewshot_prompt"] + item["prompt_round1"]
                            conversation.append({"role": "user", "content": prompt})
                        else:
                            for i in range(1, round_idx):
                                if i == 1:
                                    user_prompt = item["instruction"] + item["fewshot_prompt"] + item[f"prompt_round{i}"]
                                else:
                                    user_prompt = item[f"prompt_round{i}"]
                                conversation.extend([
                                    {
                                        "role": "user",
                                        "content": user_prompt
                                    },
                                    {
                                        "role": "assistant",
                                        "content": item[f"infer_round{i}"]
                                    }
                                ])
                            conversation.append(
                                {
                                    "role": "user",
                                    "content": item[f"prompt_round{round_idx}"]
                                }
                            )
                        conversations.append(conversation)
                generated_texts.extend(model.chat(conversations, args.tasks_config[task].get("sampling_kwargs")))
            else:
                prompts = []
                for subject in tasks_data[task]:
                    processed_data_cnt += len(tasks_data[task][subject])
                    for item in tasks_data[task][subject]:
                        if round_idx == 1:
                            prompt = item["instruction"] + item["fewshot_prompt"] + item["prompt_round1"]
                            prompt = MODEL_FORMAT[args.format_type](prompt, history=[])
                        else:
                            history = []
                            for i in range(1, round_idx):
                                if i == 1:
                                    history.append((item["instruction"] + item["fewshot_prompt"] + item[f"prompt_round{i}"], item[f"infer_round{i}"]))
                                else:
                                    history.append((item[f"prompt_round{i}"], item[f"infer_round{i}"]))
                            query = item[f"prompt_round{round_idx}"]
                            prompt = MODEL_FORMAT[args.format_type](query, history)
                        prompts.append(prompt)
                generated_texts.extend(model.generate(prompts, args.tasks_config[task].get("sampling_kwargs")))

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
                    item[f"prompt_round{round_idx + 1}"] = get_multiround_prompt(round_idx + 1, args)
    
    return infer_result


def run_eval(infer_results, args):
    """匹配答案, 对infer_results中不同task的结果调用不同的匹配函数
    Params:
        infer_result: dict[task(str), dict[subject(str), item(dict)]]
    Returns:
        score: dict[round{i}(str), dict[task(str), dict[subject(str), item(dict)]]]
    """
    result = defaultdict(dict)
    for round_idx in range(1, args.rounds + 1):
        result[f"round{round_idx}"] = {}
        for task in args.tasks:
            try:
                result[f"round{round_idx}"][task] = MATCH_TASK_ANSWER[task](infer_results[task], round_idx, args)
            except Exception as e:
                logging.error(f"Error in task {task} round {round_idx}: {e}")
    return result


def save_result(infer_result:dict, score:dict, args):
    """保存infer_result和score
    Params:
        infer_result: dict[task(str), dict[subject(str), item(dict)]]
        score: dict[round{i}(str), dict[task(str), dict[subject(str), item(dict)]]]
    """
    # 保存推理文本的可视化结果，输入和输出用'-'*20分割，每一条数据之间用'='*20分割
    if args.save_infer_texts:
        text_path = os.path.join(args.save_path, "infer_texts")
        for task in args.tasks:
            task_path = os.path.join(text_path, task)
            os.makedirs(task_path, exist_ok=True)
            for round_idx in range(1, args.rounds + 1):
                for subject, subject_data in infer_result[task].items():
                    subject_dialogs = []
                    for item in subject_data:
                        if round_idx == 1:
                            prompt = item["instruction"] + item["fewshot_prompt"] + item["prompt_round1"]
                            prompt = MODEL_FORMAT[args.format_type](prompt, history=[])
                        else:                    
                            history = []
                            for i in range(1, round_idx):
                                if i == 1:
                                    history.append((item["instruction"] + item["fewshot_prompt"] + item[f"prompt_round{i}"], item[f"infer_round{i}"]))
                                else:
                                    history.append((item[f"prompt_round{i}"], item[f"infer_round{i}"]))
                            query = item[f"prompt_round{round_idx}"]
                            prompt = MODEL_FORMAT[args.format_type](query, history)
                        response = item[f"infer_round{round_idx}"]
                        subject_dialogs.append((prompt, response))
                    subject_round_fn = os.path.join(task_path, f"{subject}_round{round_idx}.txt")
                    with open(subject_round_fn, "w", encoding="utf-8") as f:
                        for input, output in subject_dialogs:
                            print("=" * 20, file=f)
                            print(input, file=f)
                            print("-" * 20, file=f)
                            print(output, file=f)

    # 保存推理结果到json文件，每一条数据包含数据集原始信息、推理文本以及匹配答案详情（不同任务有区别）
    if args.save_infer_results:
        infer_result_path = os.path.join(args.save_path, "infer_results")
        os.makedirs(infer_result_path, exist_ok=True)
        for task in infer_result:
            task_path = os.path.join(infer_result_path, task)
            os.makedirs(task_path, exist_ok=True)
            for subject in infer_result[task]:
                subject_filename = os.path.join(task_path, f"{subject}.json")
                with open(subject_filename, "w", encoding="utf-8") as f:
                    json.dump(infer_result[task][subject], f, ensure_ascii=False, indent=4)

    # 保存评估结果
    summary_score = {}
    summary_score_with_subjects = {}
    for task in args.tasks:
        try:
            task_result_with_subjects = {}
            for subject in infer_result[task]:
                subject_result_path = os.path.join(args.save_path, "eval_result", task)
                subject_result = {}
                for round_idx in range(1, args.rounds + 1):
                    subject_result[f"round{round_idx}"] = score[f"round{round_idx}"][task][subject]
                os.makedirs(subject_result_path, exist_ok=True)
                fn = os.path.join(subject_result_path, f"{subject}.json")
                with open(fn, "w", encoding="utf-8") as f:
                    json.dump(subject_result, f, indent=4)
                
                if args.rounds == 1:
                    task_result_with_subjects[subject] = subject_result["round1"]
                else:
                    task_result_with_subjects[subject] = {f"round{round_idx}": subject_result[f"round{round_idx}"] for round_idx in range(1, args.rounds + 1)}

            if args.rounds == 1:
                task_result = score[f"round1"][task][task]
            else:
                task_result = {f"round{round_idx}": score[f"round{round_idx}"][task][task] for round_idx in range(1, args.rounds + 1)}
            summary_score[task] = task_result
            summary_score_with_subjects[task] = task_result_with_subjects
        except Exception as e:
            logging.error(f"Error in task {task}: {e}")

    with open(os.path.join(args.save_path, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_score, f, indent=4)  # 保存分数总览
    with open(os.path.join(args.save_path, "summary_of_subjects.json"), "w", encoding="utf-8") as f:
        json.dump(summary_score_with_subjects, f, indent=4, ensure_ascii=False)  # 保存子学科详情
    print(json.dumps(summary_score, indent=4))  # 同时将分数打印到终端


def parse_args():
    parser = argparse.ArgumentParser()
    # model config
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_type", type=str, default="vllm")
    parser.add_argument("--format_type", type=str, default="default")

    # task config
    parser.add_argument("--tasks", type=str, nargs="+")
    parser.add_argument("--config_path", type=str, default=None)
    parser.add_argument("--data_path", type=str, default="data")

    # save config
    parser.add_argument("--output_path", type=str, default="output")
    parser.add_argument("--save_name", type=str, required=True)
    parser.add_argument("--save_infer_results", action="store_true")
    parser.add_argument("--save_infer_texts", action="store_true")
    parser.add_argument("--temp_file_path", type=str, default="temp_file")
    parser.add_argument("--no_timestamp", action="store_true")
    parser.add_argument("--log_fn", type=str, default="log.log")

    # generate config
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--seed", type=int, default=19260817)
    parser.add_argument("--use_cpu", action="store_true")
    parser.add_argument("--use_chat", action="store_true")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    initialize(args)
    tasks_data = get_tasks_data(args)
    model = initialize_model(args)
    inference_result = run_infer(tasks_data, model, args)
    score = run_eval(inference_result, args)
    save_result(inference_result, score, args)
    finallize(args)
    

if __name__ == "__main__":
    main()
