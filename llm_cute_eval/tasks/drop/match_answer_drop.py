import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import re
from scipy.optimize import linear_sum_assignment
import string
import numpy as np
EXCLUDE = set(string.punctuation)

def normalize(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = s.lower()
    exclude = set(string.punctuation)
    s = "".join(char for char in s if char not in exclude)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = s.replace("%", " ").replace("$", " ").replace(",", "")
    s = " ".join(s.split())
    return s


def match_answer_drop(infer_result, round_idx, args):
    drop_answer_patterns = [
        '\s*answer is\s*([A-Za-z]+)\s*',
        '\s*answer is\s*(\d+\.?\d*)',
        '\s*(\d+\.?\d*)\s*',
        '\s*([A-Za-z]+)\s*',
    ]    
    correct_cnt = 0
    for item in infer_result["drop"]:
        possible_answers = []
        possible_answers.extend(normalize(item["answer"]).split())
        possible_answers.extend(re.split(r'[|]\s*|\s+', normalize(item["ref_text"])))
        norm_answer_item = normalize(item[f"infer_round{round_idx}"])
        extracted_answers = []
        for pattern in drop_answer_patterns:
            extracted_answers.extend(re.findall(pattern, norm_answer_item))
        extracted_answers = list(set(extracted_answers))
        item[f"extracted_answer_round{round_idx}"] = extracted_answers
        item[f"judge_round{round_idx}"] = False
        for extracted_answer in extracted_answers:
            for word in extracted_answer.split():
                if word in possible_answers:
                    correct_cnt += 1
                    item[f"judge_round{round_idx}"] = True
                    break
            if item[f"judge_round{round_idx}"]:
                break
    result = {
        "drop": {
            "acc": correct_cnt / len(infer_result["drop"]),
        }
    }
    return result