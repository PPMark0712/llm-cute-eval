import json
import os

import glob
mmlu_paths = glob.glob("/data1/dcy/projects/evaluate/lm-cute-eval/output/9-1_16:04_Llama-2-13b-hf/eval_result/mmlu/*json")
data = []
'''
{
    "round1": {
        "acc": 0.37,
        "correct_cnt": 37,
        "tot_cnt": 100
    },
    "round2": {
        "acc": 0.24242424242424243,
        "correct_cnt": 24,
        "tot_cnt": 99
    }
}
'''

tot_acc1 = 0.0
tot_acc2 = 0.0
for mmlu_path in mmlu_paths:
    with open(mmlu_path, "r") as f:
        result = json.load(f)
        tot_acc1+=result["round1"]["acc"]
        tot_acc2+=result["round2"]["acc"]

print(tot_acc1/len(mmlu_paths), tot_acc2/len(mmlu_paths))