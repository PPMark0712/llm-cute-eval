from FlagEmbedding import BGEM3FlagModel


def calc_similarity(text1:str, text2:str):
    model = BGEM3FlagModel('/data1/dcy/downloads/model/BAAI/bge-m3', use_fp16=True)
    text1 = [text1]
    text2 = [text2]
    text1_ids = model.encode(text1, batch_size=256, max_length=1024, )['dense_vecs']
    text2_ids = model.encode(text2, batch_size=256, max_length=1024, )['dense_vecs']
    similarity = sum(a * b for a, b in zip(text1_ids[0], text2_ids[0]))
    return float(similarity)


def calc_similarity_split(text1: str, text2: str):
    words1 = text1.lower().split()
    words2 = text2.lower().split()
    set1 = set(words1)
    set2 = set(words2)
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    jaccard_similarity = len(intersection) / len(union)
    if jaccard_similarity > 0.7:
        return jaccard_similarity
    else:
        return False
    
    
def match_answer_xsum(infer_result, round_idx, args):
    exact_match_cnt = 0
    result = {}
    for item in infer_result["xsum"]:
        answer = item["answer"]
        exact_answer = item[f"infer_round{round_idx}"]
        exact_match_cnt+=calc_similarity(answer, exact_answer)
        item[f"exact_match{round_idx}"] = calc_similarity(answer, exact_answer)
    result["xsum"] = {
        "similarity": (exact_match_cnt / len(infer_result["xsum"])),
    }
    return result