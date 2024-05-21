import re

"""
exact_match: Match answer after 'The answer is #### '
flexible_match: Match every number in the response, if any number equals to the answer, the answer is correct.
"""


number_pattern = r'(-?\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)'
gsm8k_data_pattern = "#### " + number_pattern
exact_pattern = r"The\s+answer\s+is\s+#+\s+" + number_pattern
flexible_pattern = number_pattern

def str_to_float(text: str):
    """convert string like '1,234.00' to float"""
    return float(text.replace(",", ""))


def match_answer_gsm8k(infer_result, round_idx, args):
    exact_match_cnt = 0
    flexible_match_cnt = 0 
    result = {}
    for item in infer_result["gsm8k"]:
        answer = str_to_float(re.findall(gsm8k_data_pattern, item["answer"])[0])
        
        # match answer after 'The answer is #### '
        exact_answer = re.findall(exact_pattern, item[f"infer_round{round_idx}"])
        if len(exact_answer) > 0:
            model_answer = str_to_float(exact_answer[0])
            if abs(model_answer - answer) < 1e-6:
                exact_match_cnt += 1
        
        # match every number in the response
        flexible_answers = re.findall(flexible_pattern, item[f"infer_round{round_idx}"])
        if len(flexible_answers) > 0:
            for num_str in flexible_answers:
                if abs(str_to_float(num_str) - answer) < 1e-6:
                    flexible_match_cnt += 1
                    break

    result["gsm8k"] = {
        "exact_match": exact_match_cnt / len(infer_result["gsm8k"]),
        "flexible_match": flexible_match_cnt / len(infer_result["gsm8k"])
    }
    return result