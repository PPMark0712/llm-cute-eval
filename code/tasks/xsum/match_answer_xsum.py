
# '''
# # {
# #   'colbert': [0.7796499729156494, 0.4621465802192688, 0.4523794651031494, 0.7898575067520142], 
# #   'sparse': [0.195556640625, 0.00879669189453125, 0.0, 0.1802978515625], 
# #   'dense': [0.6259765625, 0.347412109375, 0.349853515625, 0.67822265625], 
# #   'sparse+dense': [0.482503205537796, 0.23454029858112335, 0.2332356721162796, 0.5122477412223816], 
# #   'colbert+sparse+dense': [0.6013619303703308, 0.3255828022956848, 0.32089319825172424, 0.6232916116714478]
# # }
# '''
# from FlagEmbedding import BGEM3FlagModel
def calc_similarity(text1:str, text2:str):
    
    text1 = [text1]
    text2 = [text2]
    text1_ids = model.encode(text1, batch_size=256, max_length=1024, )['dense_vecs']
    text2_ids = model.encode(text2, batch_size=256, max_length=1024, )['dense_vecs']
    similarity = text1_ids @ text2_ids.T
    if similarity > 0.8:
        return True
    else:
        return False

def match_answer_xsum(infer_result, round_idx, args):
    exact_match_cnt = 0
    result = {}
    answer = []
    exact_answer = []
    # model = BGEM3FlagModel('/home/admin/workspace/aop_lab/app_source/dcy/download/model/BAAI/bge-m3', use_fp16=True)
    # for item in infer_result["xsum"]:
    #     answer.append(item["answer"])
    #     exact_answer.append(item[f"infer_round{round_idx}"])
    # text1_ids = model.encode(exact_answer, batch_size=256, max_length=1024, )['dense_vecs']    
    # text2_ids = model.encode(answer, batch_size=256, max_length=1024, )['dense_vecs']
    # similarities = text1_ids @ text2_ids.T
    # for similarity in similarities:
    #     exact_match_cnt+= similarity
    #     item[f"exact_match{round_idx}"] = similarity
        
    result["xsum"] = {
        "similarity": (exact_match_cnt / len(infer_result["xsum"])),
    }
    return result


# import json
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"
# if __name__ == "__main__":
#     path = "/data1/dcy/projects/evaluate/lm-cute-eval/output/6-17_17:46_Llama-2-13b-chat-hf/infer_results/xsum/xsum.json"
    
#     exact_match_cnt = 0
#     result = {}
#     answer = []
#     exact_answer1 = []
#     exact_answer2 = []
#     with open(path, "r") as f:
#         xsum_result = json.load(f)
#     eval_result = xsum_result
#     model = BGEM3FlagModel('/data1/dcy/downloads/model/BAAI/bge-m3', use_fp16=True)
#     for item in xsum_result:
#         answer.append(item["answer"])
#         exact_answer1.append(item[f"infer_round1"])
#         exact_answer2.append(item[f"infer_round2"])
#     exact_answer1_ids = model.encode(exact_answer1, batch_size=256, max_length=1024, )['dense_vecs']    
#     exact_answer2_ids = model.encode(exact_answer2, batch_size=256, max_length=1024, )['dense_vecs']    
#     text2_ids = model.encode(answer, batch_size=256, max_length=1024, )['dense_vecs']
#     similarities1 = exact_answer1_ids @ text2_ids.T
#     similarities2 = exact_answer2_ids @ text2_ids.T
#     for similarity in similarities1:
#         exact_match_cnt+= similarity
#         eval_result[f"exact_match1"] = similarity
#     for similarity in similarities2:
#         exact_match_cnt+= similarity
#         eval_result[f"exact_match2"] = similarity
        
        
        