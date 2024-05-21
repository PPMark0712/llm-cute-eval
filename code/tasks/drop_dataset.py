from typing import Any, Dict, List, Optional, Set, Tuple, Union
import re
from scipy.optimize import linear_sum_assignment
import string, numpy as np

EXCLUDE = set(string.punctuation)


def _white_space_fix(text: str) -> str:
    '''
    这个函数接收一个字符串 text 作为输入，并返回一个新的字符串，其中所有的空白字符（例如空格、制表符、换行符等）被替换为单个空格。
    '''
    return " ".join(text.split())


def _remove_articles(text: str) -> str:
    '''
    这个函数使用正则表达式来移除文本中的定冠词（"the"）、不定冠词（"a"、"an"）。
    '''
    regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
    return re.sub(regex, " ", text)


def _remove_punc(text: str) -> str:
    ''''
    这个函数意在移除文本中的标点符号，但完整的函数实现没有给出。不过，可以根据提供的代码段进行推断：
    '''
    if not _is_number(text):
        return "".join(ch for ch in text if ch not in EXCLUDE)
    else:
        return text


def _is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


def _lower(text: str) -> str:
    return text.lower()


def _tokenize(text: str) -> List[str]:
    '''
    text = "The quick-brown fox jumps over the lazy-dog."
    tokens = _tokenize(text)
    print(tokens)  # 输出: ['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog', '.']
    '''
    return re.split(" |-", text)


def _normalize_number(text: str) -> str:
    if _is_number(text):
        return str(float(text))
    else:
        return text


def _normalize_answer(text: str) -> str:
    '''
    _lower：将文本转换为小写。
    _remove_punc：移除文本中的标点符号。
    _normalize_number：规范化数字（这里没有实际的实现，只是返回了原始文本）。
    _remove_articles：移除文本中的冠词（"a", "an", "the"）。
    _white_space_fix：移除多余的空白字符。
    # 示例文本
    text = "The quick-brown fox, jumps over the lazy-dog. A lazy dog's life is not for me."

    # 输出: "quick brown fox jump over lazy dog a lazy dog life not for me"
    '''
    parts = [
        _white_space_fix(_remove_articles(_normalize_number(_remove_punc(_lower(token)))))
        for token in _tokenize(text)
    ]
    parts = [part for part in parts if part.strip()]
    normalized = " ".join(parts).strip()
    return normalized


def _answer_to_bags(
        answer: Union[str, List[str], Tuple[str, ...]]
) -> Tuple[List[str], List[Set[str]]]:
    '''
    answer = "The quick Brown FOX."
    normalized, bags = _answer_to_bags(answer)
    print(normalized)  # 输出: ["the quick brown fox."]
    print(bags)        # 输出: [{ 'the', 'quick', 'brown', 'fox'}]    
    '''
    if isinstance(answer, (list, tuple)):
        raw_spans = answer
    else:
        raw_spans = [answer]
    normalized_spans: List[str] = []
    token_bags = []
    for raw_span in raw_spans:
        normalized_span = _normalize_answer(raw_span)
        normalized_spans.append(normalized_span)
        token_bags.append(set(normalized_span.split()))
    return normalized_spans, token_bags


def _compute_f1(predicted_bag: Set[str], gold_bag: Set[str]) -> float:
    intersection = len(gold_bag.intersection(predicted_bag))
    if not predicted_bag:
        precision = 1.0
    else:
        precision = intersection / float(len(predicted_bag))
    if not gold_bag:
        recall = 1.0
    else:
        recall = intersection / float(len(gold_bag))
    f1 = (
             (2 * precision * recall) / (precision + recall)
             if not (precision == 0.0 and recall == 0.0)
             else 0.0
         ) * 100
    return f1


def _match_numbers_if_present(gold_bag: Set[str], predicted_bag: Set[str]) -> bool:
    gold_numbers = set()
    predicted_numbers = set()
    for word in gold_bag:
        if _is_number(word):
            gold_numbers.add(word)
    for word in predicted_bag:
        if _is_number(word):
            predicted_numbers.add(word)
    if (not gold_numbers) or gold_numbers.intersection(predicted_numbers):
        return True
    return False


def _align_bags(predicted: List[Set[str]], gold: List[Set[str]]) -> List[float]:
    """
    Takes gold and predicted answer sets and first finds the optimal 1-1 alignment
    between them and gets maximum metric values over all the answers.
    """
    scores = np.zeros([len(gold), len(predicted)])

    for gold_index, gold_item in enumerate(gold):
        for pred_index, pred_item in enumerate(predicted):
            if _match_numbers_if_present(gold_item, pred_item):
                scores[gold_index, pred_index] = _compute_f1(pred_item, gold_item)
    row_ind, col_ind = linear_sum_assignment(-scores)

    max_scores = np.zeros([max(len(gold), len(predicted))])
    for row, column in zip(row_ind, col_ind):
        max_scores[row] = max(max_scores[row], scores[row, column])
    return max_scores


def get_drop_metrics(
        predicted: Union[str, List[str], Tuple[str, ...]], gold: Union[str, List[str], Tuple[str, ...]]
) -> Tuple[float, float]:
    """
    接收一个预测答案和一个黄金答案（两者都可能是字符串或字符串列表），
    并返回预测的精确匹配分数以及DROP F1指标。如果你正在编写一个脚本，
    用于在内存中（如，验证阶段或训练过程中）对预测结果进行评估，
    那么这个就是你想要调用的函数。在从发布数据文件中读取黄金答案时，
    应先使用`answer_json_to_strings`函数将其转换为字符串形式。
    """
    predicted_bags = _answer_to_bags(predicted)
    gold_bags = _answer_to_bags(gold)

    if set(predicted_bags[0]) == set(gold_bags[0]) and len(predicted_bags[0]) == len(gold_bags[0]):
        exact_match = 1.0
    else:
        exact_match = 0.0

    f1_per_bag = _align_bags(predicted_bags[1], gold_bags[1])
    f1 = np.mean(f1_per_bag)
    f1 = round(f1, 2)
    return exact_match, f1


def drop_metric(sample: str, reference: list[str]) -> Tuple[float, float]:
    em_scores = []
    f1_scores = []
    for answer in reference:
        if answer.strip() != "":
            em, f1 = get_drop_metrics(sample, answer)
            em_scores.append(em)
            f1_scores.append(f1)
    return (max(em_scores), max(f1_scores))


def normalize(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = s.lower()
    exclude = set(string.punctuation)
    s = "".join(char for char in s if char not in exclude)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = " ".join(s.split())
    return s


def fuzzy_match(s1: str, s2: str) -> bool:
    s1 = normalize(s1)
    s2 = normalize(s2)

    if s1 == "" or s2 == "":
        return s1 == s2

    return s1 in s2 or s2 in s1


# def zs_drop_match_answer(task_data, response):
#     a = task_data["completion"]
#     correct_answers = task_data["ref_text"].split("|")

#     ANSWER_PATTERN = r"(?i)Answer\s*:\s*([^\n]+)"
#     patterns2 = [
#     '\s*The answer is\s*([A-Za-z]+)\s*',
#     '\s*So, the final answer is \$\s*(\d+\.?\d*)\s*',
#     '\s*The answer is: \s*(\d+\.?\d*)',
#     '\s*The answer is: \$\s*(\d+\.?\d*)',
#      '\s*The answer is: \$\s*[(\d+\.?\d*)]',
#     '\s*The correct answer is \$\s*(\d+\.?\d*)',
#     '\s*The correct answer is \s*(\d+\.?\d*)',
#     '\s*The correct answer is \s*(\d+\.?\d*)'
#     '\s*The correct answer is \$\s*(\d+\.?\d*)'
#     '\s*Therefore, the final answer is \$\s*(\d+\.?\d*)',
#     '\s*Therefore, the final answer is \s*(\d+\.?\d*)',
#     '\s*So, the final answer is \s*(\d+\.?\d*)',
#     '\s*\$\s*(\d+\.?\d*)',
#     '\s*>>(\d+\.?\d*)',

#     ]
#     for pa in patterns2:
#         match = re.findall(pa, response)
#         if match:
#             break  # 找到匹配后立即退出循环

#     if len(match) == 0:
#         match = re.findall(ANSWER_PATTERN, response)
#     extracted_answer = match[-1] if match else response
#     em_score, f1_score = drop_metric(extracted_answer, correct_answers)
#     matches = [
#                         fuzzy_match(extracted_answer, correct_answer)
#                         for correct_answer in correct_answers
#     ]
#     extracted_answers = [
#                         extracted_answer for i in range(len(correct_answers)) if matches[i]
#     ]
#     score = True in matches

#     result = {
#         "score" : score,
#         "correct_answers" : correct_answers,
#         "extracted_answers" : extracted_answers,
#         "em_score" : em_score,
#         "f1_score" : f1_score
#     }
#     if match:
#         return True,correct_answers[-1], extracted_answers[:-1]
#     else :
#         return False,correct_answers[-1], "no answer matches"
def zs_drop_match_answer(task_data, response):
    a = task_data["completion"]
    correct_answers = task_data["ref_text"].split("|")

    ANSWER_PATTERN = r"(?i)Answer\s*:\s*([^\n]+)"
    patterns2 = [
        '\s*The answer is\s*([A-Za-z]+)\s*',
        '\s*So, the final answer is \$\s*(\d+\.?\d*)\s*',
        '\s*The answer is: \s*(\d+\.?\d*)',
        '\s*The answer is: \$\s*(\d+\.?\d*)',
        '\s*The answer is: \$\s*[(\d+\.?\d*)]',
        '\s*The correct answer is \$\s*(\d+\.?\d*)',
        '\s*The correct answer is \s*(\d+\.?\d*)',
        '\s*The correct answer is \s*(\d+\.?\d*)'
        '\s*The correct answer is \$\s*(\d+\.?\d*)'
        '\s*Therefore, the final answer is \$\s*(\d+\.?\d*)',
        '\s*Therefore, the final answer is \s*(\d+\.?\d*)',
        '\s*So, the final answer is \s*(\d+\.?\d*)',
        '\s*\$\s*(\d+\.?\d*)',
        '\s*>>(\d+\.?\d*)',

    ]
    for pa in patterns2:
        pred_numbers = re.findall(pa, response)
        if pred_numbers:
            break  # 找到匹配后立即退出循环

    if len(pred_numbers) == 0:
        pred_numbers = re.findall(r'\d+(?=\D*$)', response)

    # Check if both answer and response contain numbers and compare the last found number
    if correct_answers and pred_numbers:
        # Compare the last number found in both answer and response
        if correct_answers[-1] == pred_numbers[-1]:
            return True, correct_answers[-1], pred_numbers[-1]  # Match found, return 1 and the matching number
        else:
            return False, correct_answers[-1], pred_numbers[
                -1]  # Numbers do not match, return 0 and the last number from the response

    return False, correct_answers[-1], "No match or missing numbers"  # No numbers found or other issues
