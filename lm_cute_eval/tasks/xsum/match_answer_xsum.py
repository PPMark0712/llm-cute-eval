# from FlagEmbedding import BGEM3FlagModel


# def match_answer_xsum(infer_result, round_idx, args):
#     similarity_sum = 0
#     result = {}
#     similarity_model = BGEM3FlagModel("/data2/dcy/downloads/model/BAAI/bge-m3", use_fp16=True)
#     for item in infer_result["xsum"]:
#         label_ids = similarity_model.encode(item["answer"], batch_size=256, max_length=1024, )['dense_vecs']    
#         model_answer_ids = similarity_model.encode(item[f"infer_round{round_idx}"], batch_size=256, max_length=1024, )['dense_vecs']
#         similarity = float(label_ids @ model_answer_ids.T)
#         similarity_sum += similarity
#         item[f"similarity_round{round_idx}"] = similarity
        
#     result["xsum"] = {
#         "similarity": similarity_sum / len(infer_result["xsum"]),
#     }
#     return result

def match_answer_xsum(infer_result, round_idx, args):
    return {"xsum": "skipped"}