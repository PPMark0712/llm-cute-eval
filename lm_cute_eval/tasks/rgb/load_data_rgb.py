import os, json
import math
import random


def format_query_rgb(query, docs, subject):
    if "en" in subject:
        instruction = "You are an accurate and reliable AI assistant that can answer questions with the help of external documents. Please note that external documents may contain noisy or factually incorrect information. If the information in the document contains the correct answer, you will give an accurate answer. If the information in the document does not contain the answer, you will generate ’I can not answer the question because of the insufficient information in documents.‘. If there are inconsistencies with the facts in some of the documents, please generate the response 'There are factual errors in the provided documents.' and provide the correct answer.\n"
        prompt_format = "Document:\n{DOCS} \n\nQuestion:\n{QUERY}"
    else:
        instruction = "你是一个准确和可靠的人工智能助手，能够借助外部文档回答问题，请注意外部文档可能存在噪声事实性错误。如果文档中的信息包含了正确答案，你将进行准确的回答。如果文档中的信息不包含答案，你将生成“文档信息不足，因此我无法基于提供的文档回答该问题。”。如果部分文档中存在与事实不一致的错误，请先生成“提供文档的文档存在事实性错误。”，并生成正确答案。\n"
        prompt_format = "文档：\n{DOCS} \n\n问题：\n{QUERY}"
    
    docs = '\n'.join(docs) if len(docs) > 0 else ""
    prompt = prompt_format.format(QUERY=query, DOCS=docs)
    return instruction, prompt


def load_file_rgb(fn, limit=0):
    data = []
    with open(fn, "r") as f:
        for line in f:
            data.append(json.loads(line))
            if limit and len(data) >= limit:
                break
    return data


def process_data(instance, noise_rate, passage_num, subject, correct_rate):
    neg_num = math.floor(passage_num * noise_rate)
    pos_num = passage_num - neg_num

    if '_int' in subject:
        for i in instance['positive']:
            random.shuffle(i)
        docs = [i[0] for i in instance['positive']]
        if len(docs) < pos_num:
            maxnum = max([len(i) for i in instance['positive']])
            for i in range(1, maxnum):
                for j in instance['positive']:
                    if len(j) > i:
                        docs.append(j[i])
                        if len(docs) == pos_num:
                            break
                if len(docs) == pos_num:
                    break
        neg_num = passage_num - len(docs)
        if neg_num > 0:
            negative = instance['negative'][:neg_num]
            docs += negative
    elif '_fact' in subject:
        correct_num = math.ceil(passage_num * correct_rate)
        pos_num = passage_num - neg_num - correct_num
        indexs = list(range(len(instance['positive'])))
        selected = random.sample(indexs, min(len(indexs), pos_num))
        docs = [instance['positive_wrong'][i] for i in selected]
        remain = [i for i in indexs if i not in selected]
        if correct_num > 0 and len(remain) > 0:
            docs += [instance['positive'][i] for i in random.sample(remain, min(len(remain), correct_num))]
        if neg_num > 0:
            docs += instance['negative'][:neg_num]
    else:
        if noise_rate == 1:
            neg_num = passage_num
            pos_num = 0
        else:
            if neg_num > len(instance['negative']):
                neg_num = len(instance['negative'])
                pos_num = passage_num - neg_num
            elif pos_num > len(instance['positive']):
                pos_num = len(instance['positive'])
                neg_num = passage_num - pos_num

        positive = instance['positive'][:pos_num]
        negative = instance['negative'][:neg_num]
        docs = positive + negative
    
    random.shuffle(docs)
    return instance['query'], docs


def load_data_rgb(args):
    rgb_dir = os.path.join(args.data_path, "tasks", "rgb")
    task_config = args.tasks_config["rgb"]
    task_data = {}
    subjects = task_config["subjects"]
    for subject in subjects:
        fn = os.path.join(rgb_dir, f"{subject}.jsonl")
        subject_data = load_file_rgb(fn, task_config["limit"])
        task_data[subject] = []
        for item in subject_data:
            query, docs = process_data(item, task_config["noise_rate"], task_config["passage_num"], subject, task_config["correct_rate"])
            instruction, prompt = format_query_rgb(query, docs, subject)
            task_data[subject].append({
                **item,
                "instruction": instruction,
                "fewshot_prompt": "",
                "prompt_round1": prompt,
            })
    return task_data
