
'''
# {
#   'colbert': [0.7796499729156494, 0.4621465802192688, 0.4523794651031494, 0.7898575067520142], 
#   'sparse': [0.195556640625, 0.00879669189453125, 0.0, 0.1802978515625], 
#   'dense': [0.6259765625, 0.347412109375, 0.349853515625, 0.67822265625], 
#   'sparse+dense': [0.482503205537796, 0.23454029858112335, 0.2332356721162796, 0.5122477412223816], 
#   'colbert+sparse+dense': [0.6013619303703308, 0.3255828022956848, 0.32089319825172424, 0.6232916116714478]
# }
'''
from argparse import Namespace
from FlagEmbedding import BGEM3FlagModel
def match_answer_xsum(infer_result, round_idx, args):
    # if round_idx == 1:
    #     import os
    #     os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    #     # torch.cuda.empty_cache()
    #     # visible_devices = os.getenv('CUDA_VISIBLE_DEVICES', '')
    #     # if visible_devices:
    #     #     # 如果环境变量设置了，使用第一个设备编号
    #     #     device_id = int(visible_devices.split(',')[0])
    #     #     device = torch.device(f"cuda:{device_id}")
    #     # else:
    #     #     # 如果没有设置环境变量，自动选择设备
    #     #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     # print(device)
    #     model = BGEM3FlagModel('/data1/dcy/downloads/model/BAAI/bge-m3', use_fp16=True)
    #     args.model = model
    result = {}
    # answer = []
    # exact_answer = []
    # for item in infer_result["xsum"]:
    #     answer.append(item["answer"])
    #     exact_answer.append(item[f"infer_round{round_idx}"])
    # sentence_pairs = [[answer[i],exact_answer[i]] for i in range(len(answer))]
    # similarities = args.model.compute_score(sentence_pairs, 
    #                       max_passage_length=1024, # a smaller max length leads to a lower latency
    #                       weights_for_different_modes=[0.4, 0.2, 0.4])['colbert+sparse+dense'] # weights_for_different_modes(w) is used to do weighted sum: w[0]*dense_score + w[1]*sparse_score + w[2]*colbert_score
    # for similarity in similarities:
    #     exact_match_cnt+= similarity
    #     item[f"exact_match{round_idx}"] = similarity
    #     item[f"judge{round_idx}"] = False
    #     if similarity >0.7:
    #         item[f"judge{round_idx}"] = True
        
    result["xsum"] = {
        "similarity": (1 / len(infer_result["xsum"])),
    }
    return result