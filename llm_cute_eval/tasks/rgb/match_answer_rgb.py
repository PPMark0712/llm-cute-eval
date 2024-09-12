def get_labels(prediction, ground_truth):
    prediction = prediction.lower()
    if type(ground_truth) is not list:
        ground_truth = [ground_truth]
    labels = []
    for instance in ground_truth:
        if type(instance) == list:
            # 该问题的回答有多种格式（例如日期），只要其中一种正确即为正确
            flag = False
            instance = [i.lower() for i in instance]
            for i in instance:
                if i in prediction:
                    flag = True
                    break
        else:
            instance = instance.lower()
            flag = instance in prediction
        
        labels.append(int(flag))
    return labels


def match_answer_rgb(infer_result:dict, round_idx, args):
    task_config = args.tasks_config["rgb"]
    result = {}
    for subject in task_config["subjects"]:
        correct_cnt = 0
        subject_results = []
        for item in infer_result[subject]:
            prediction = item[f"infer_round{round_idx}"]

            if 'zh' in subject:
                prediction = prediction.replace(" ", "")

            if '信息不足' in prediction or 'insufficient information' in prediction:
                labels = [-1]
            else:
                labels = get_labels(prediction, item["answer"])

            fact_label = 0
            if '事实性错误' in prediction or 'factual errors' in prediction:
                fact_label = 1  # 模型能试别出事实性错误
            
            subject_results.append({
                'labels': labels,
                'fact_label': fact_label
            })
            item["extracted_labels"] = labels
            item["fact_label"] = fact_label

        correct_cnt = 0
        for item in subject_results:
            labels = item["labels"]
            if task_config["noise_rate"] == 1 and labels[0] == -1:
                correct_cnt += 1
            elif 0 not in labels and 1 in labels:
                correct_cnt += 1

        subject_result = {
            'acc': correct_cnt / len(subject_results),
            'correct_cnt': correct_cnt,
            'total_cnt': len(subject_results),
        }

        if '_fact' in subject:
            fact_cnt = 0
            correct_cnt = 0
            for item in subject_results:
                if item['fact_label'] == 1:
                    fact_cnt += 1
                    if 0 not in item['labels']:
                        correct_cnt += 1
            fact_check_rate = fact_cnt / len(subject_results)
            subject_result.update({
                "acc": correct_cnt / fact_cnt if fact_cnt else 0,
                "correct_cnt": correct_cnt,  # 
                "fact_check_rate": fact_check_rate,  # 检查出事实冲突的比例
                "fact_cnt": fact_cnt,  # 检查出事实冲突的条数
            })

        result[subject] = subject_result

    summary_correct_cnt = 0
    all_total_cnt = 0
    for subject in task_config["subjects"]:
        all_total_cnt += result[subject]["total_cnt"]
        if "_fact" in subject:
            summary_correct_cnt += result[subject]["correct_cnt"]
            # summary_correct_cnt += result[subject]["fact_cnt"]
        else:
            summary_correct_cnt += result[subject]["correct_cnt"]

    result["rgb"] = {
        "acc": summary_correct_cnt / all_total_cnt
    }
    return result
            

