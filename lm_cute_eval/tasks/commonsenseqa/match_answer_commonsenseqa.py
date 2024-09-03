from ..match_answer import find_first_selection

def match_answer_commonsenseqa(infer_result:dict, round_idx:int, args):
    result = {}
    correct_cnt = 0
    for item in infer_result["commonsenseqa"]:
        
        model_answer = None
        
        item[f"judge{round_idx}"] = False
        problem_answer = f'{item["ans"]}'
        problem_answer_list = problem_answer.replace("(", " ").replace(")", " ").replace("\n", " ").upper().split(" ")
        if model_answer is None:
            model_answer = item[f"infer_round{round_idx}"]
            model_answer_list = model_answer.upper().replace("(", " ").replace(")", " ").replace("\n", " ").upper().split("ANSWER IS")[-1].split(" ")
        else:
            model_answer_list = [model_answer.upper()]
        for problem_answer in problem_answer_list:
            if problem_answer in model_answer_list:
                correct_cnt += 1
                item[f"judge{round_idx}"] = True
                break
        item[f"extract_answer_round{round_idx}"] = model_answer
    result["commonsenseqa"] = {
        "acc": correct_cnt / len(infer_result["commonsenseqa"]),
    }
    return result


if __name__ == "__main__":
    str1 = '''
    Sure, I'd be happy to help! Based on the question you provided, it seems that many people wanted to leave their country estates for row houses. To move to these row houses, they would need to relocate to a city.\n\nHere's my reasoning:\n\n1. The question mentions \"country estates\" and \"row houses,\" which suggests that the people are leaving their rural or suburban homes for a more urban environment.\n2. The answer choices do not provide any information about moving from a rural area to a prison or England, so those options can be eliminated.\n3. Living less expensively is not necessarily related to moving from a country estate to a row house in the city, so choice A can also be eliminated.\n\nTherefore, the correct answer is (D) city. This option reflects the idea that many people were looking for a new home in an urban setting with different living arrangements and possibly lower housing costs compared to their previous rural homes.>\r\r\n
       
    '''
    str2 = '''
    (D) city\n
    
    '''
    problem_answer_list = str1.replace("(", " ").replace(")", " ").replace("\n", " ").upper().split("ANSWER IS")[-1].split(" ")
    
    
    print(problem_answer_list)
    
    correct_cnt=0
    problem_answer = 'D'
    problem_answer1 = problem_answer.replace("(", " ").replace(")", " ").replace("\n", " ").upper().split(" ")
    for problem_answer in problem_answer_list:
            if problem_answer in problem_answer1:
                correct_cnt += 1
    print(correct_cnt)