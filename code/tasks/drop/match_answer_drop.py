import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import re
from scipy.optimize import linear_sum_assignment
import string, numpy as np

EXCLUDE = set(string.punctuation)
"""
exact_match: Match answer after 'The answer is #### '
flexible_match: Match every number in the response, if any number equals to the answer, the answer is correct.
"""


drop_data_pattern = [
        '\s*answer is\s*([A-Za-z]+)\s*',
        '\s*answer is\s*(\d+\.?\d*)',
        '\s*(\d+\.?\d*)\s*',
        '\s*([A-Za-z]+)\s*',
    ]
def normalize(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = s.lower()
    exclude = set(string.punctuation)
    s = "".join(char for char in s if char not in exclude)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = " ".join(s.split())
    return s

def match_answer_drop(infer_result, round_idx, args):
    exact_match_cnt = 0
    flexible_match_cnt = 0 
    result = {}
    for item in infer_result["drop"]:
        answer = []
        norm_ref_answer = normalize(item["answer"])
        answer_asnwer = re.split(r' ', norm_ref_answer)
        norm_ref_text = normalize(item["ref_text"])
        answer_text = re.split(r'[|]\s*|\s+', norm_ref_text)
        answer.extend(answer_asnwer)
        answer.extend(answer_text)
        norm_answer_item = normalize(item[f"infer_round{round_idx}"])
        for pa in drop_data_pattern:
            exact_answer = re.findall(pa, norm_answer_item)
            if exact_answer:
                break  
        item[f"exact_match{round_idx}"] = False
        if len(exact_answer) > 0:
            model_answer = exact_answer[0].split(' ')
            flag = 0
            for ans1 in model_answer:
                for ans2 in answer:
                    if ans1 == ans2:
                        item[f"exact_match{round_idx}"] = True
                        exact_match_cnt += 1
                        flag = 1
                        break
                if flag == 1:
                    break

    result["drop"] = {
        "exact_match": exact_match_cnt / len(infer_result["drop"]),
    }
    return result