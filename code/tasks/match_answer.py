import re


def find_first_selection(text:str, choices="ABCDEF"):
    """找到回答的第一个选项"""
    new_text = ""
    # 把所有标点符号替换为空格
    for c in text:
        if c.isalpha() or c.isspace():
            new_text += c
        else:
            new_text += ' '
    # split后找到第一个选项字母
    lst = new_text.split()
    for s in lst:
        if len(s) == 1 and s.isupper() and s in choices:
                return s[0]
        lst = new_text.split()
    
    # 找到第一个属于1 2 3 4....的数字，1234...可能分别代表了ABCD...
    # for s in lst:
    #     if len(s) == 1:
    #         if s.isdigit() and 1 <= s <= len(choices):
    #             return choices[int(s) - 1]
    return None
