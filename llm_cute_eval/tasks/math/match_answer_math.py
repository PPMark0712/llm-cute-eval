import re
from latex2sympy2 import latex2sympy
from .math_equivalence import is_equiv, _strip_string

def find_first_box(text):
    id = text.find(r"\boxed")
    if id == -1:
        return None
    start = id + len(r"\boxed")
    if start >= len(text) or text[start] != '{':
        return None
    count = 1  # 左括号计数
    end = start + 1
    while end < len(text) and count > 0:
        if text[end] == '{':
            count += 1
        elif text[end] == '}':
            count -= 1
        end += 1
    if count == 0:
        return text[start+1:end-1]  # 返回括号内的内容
    return None


def match_re(text):
    patterns = [
        r"[aA]nswer\s*is(?:\s*:)?\s*(.*?)(\n|$)",  # 匹配直到遇到换行符或字符串结尾的内容, 不能匹配句号，因为有小数
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            s = match.group(1).strip()
            if s.startswith("$") and s.endswith("$"):
                s = s[1:-1]
            elif s.startswith("$") and not s.endswith("$"):
                s = s[1:s.rfind("$")]
            elif s.endswith(".") or s.endswith("。"):  # 不在数学代码块$$中，去除句号
                s = s[:-1]
            return s
    return None


def is_latex_equivalent(latex1, latex2):
    flag = False
    try:
        expr1 = latex2sympy(latex1)
        expr2 = latex2sympy(latex2)
        if expr1.equals(expr2):
            flag = True
    except Exception as e:
        # print(e)
        pass
    try:
        expr1 = latex2sympy(_strip_string(latex1))
        expr2 = latex2sympy(_strip_string(latex2))
        if expr1.equals(expr2):
            flag = True
    except Exception as e:
        # print(e)
        pass
    return flag
    
    

def check_equiv(response, answer):
    flag = False
    if is_latex_equivalent(response, answer) or is_equiv(response, answer):
        flag = True
    # 去掉\left(等括号形式
    response = response.replace(r"\left", "").replace(r"\right", "")
    answer = answer.replace(r"\left", "").replace(r"\right", "")
    if is_latex_equivalent(response, answer) or is_equiv(response, answer):
        flag = True
    if response.startswith("(") and response.endswith(")"):
        response = response[1:-1]
    if answer.startswith("(") and answer.endswith(")"):
        answer = answer[1:-1]
    if is_latex_equivalent(response, answer) or is_equiv(response, answer):
        flag = True
    # 匹配坐标点(a, b)的形式
    splited_answer = answer.split(",")
    splited_response = response.split(",")
    if len(splited_answer) == len(splited_response):
        all_equiv = True
        for a, b in zip(splited_answer, splited_response):
            if not is_latex_equivalent(a, b) and not is_equiv(a, b):
                all_equiv = False
                break
        if all_equiv:
            flag = True
    return flag


def extract_answer(text):
    funcs = [find_first_box, match_re]
    for func in funcs:
        s = func(text)
        if s is not None:
            return s.strip()
    return None


def match_answer_math(infer_result:dict, round_idx:int, args):
    result = {}
    correct_cnt = 0
    total_cnt = 0
    for item in infer_result["math"]:
        model_answer = extract_answer(item[f"infer_round{round_idx}"])
        item[f"extract_answer_round{round_idx}"] = model_answer
        if model_answer is not None:
            item[f"judge{round_idx}"] = check_equiv(model_answer, item["answer"])
        else:
            item[f"judge{round_idx}"] = False
        total_cnt += 1
        if item[f"judge{round_idx}"]:
            correct_cnt += 1
    result["math"] = {
        "acc": correct_cnt / total_cnt,
    }
    return result


if __name__ == "__main__":
    tests = [
        (r"The answer is $\boxed{\frac{1}{2}}$. This is more text.", r"\frac{1}{2}"),
        ("The answer is:2x.\nHere is another line.", "2x"),
        (r"The answer is (2(x+1)^2).", "2(x+1)^2"),
        (r"The answer is (2(x+1)^2).", "2x^2+4x+2"),
        (r"The answer is : $\frac{\sqrt{3}+1}{2}$", r"\frac{\sqrt{3}+1}{2}"),
        (r"Answer is: \frac{\pi}{\sqrt{5}}", r"\frac{\pi}{\sqrt{5}}"),
        (r"The answer is 9.", "9"),
        (r"The answer is $\frac{\sqrt3 \pi + 1}{x^2+3x+1}$. This is more text.", r"\frac{\sqrt{3} \pi+1}{x^2+1+3x}"),
        (r"The answer is $\boxed{(3, \frac{\pi}{2})}$. This is more text.", r"\left(3, \frac{\pi}{2}\right)"),
    ]
    for test in tests:
        print(extract_answer(test[0]))
        print(check_equiv(extract_answer(test[0]), test[1]))