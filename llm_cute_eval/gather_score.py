import pandas as pd
import argparse
import os
import json

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_path", type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    path_list = os.listdir(args.exp_path)
    score_dict = {}
    for folder_name in path_list:
        result_fn = os.path.join(args.exp_path, folder_name, "summary.json")
        with open(result_fn, "r") as f:
            result = json.load(f)
            score_dict[folder_name] = result
    
    print(json.dumps(score_dict, indent=4))
    
        

