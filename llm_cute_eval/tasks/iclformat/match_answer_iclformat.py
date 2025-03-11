# import re
# import nltk
# from supar import Parser

# parser_model = None


# def init(nltk_path, parser_path):
#     global parser_model
#     if not parser_model:
#         parser_model = Parser.load(parser_path)
#     if nltk_path and nltk_path not in nltk.data.path:
#         nltk.data.path.append(nltk_path)
    

# def rouge(a, b, beta=1):
#     n = len(a)
#     m = len(b)
#     if n == 0 or m == 0:
#         return 0
#     dp = [[0 for _ in range(m)] for _ in range(n)]
#     for i in range(n):
#         dp[i][0] = 1 if a[i] == b[0] else 0
#     for i in range(m):
#         dp[0][i] = 1 if a[0] == b[i] else 0
#     for i in range(1, n):
#         for j in range(1, m):
#             dp[i][j] = max(dp[i-1][j], dp[i][j-1])
#             if a[i] == b[j]:
#                 dp[i][j] = max(dp[i][j], dp[i-1][j-1] + 1)
#     if dp[n - 1][m - 1] == 0:
#         return 0
#     p1 = dp[n - 1][m - 1] / n
#     p2 = dp[n - 1][m - 1] / m
#     f = (1 + beta**2) * p1 * p2 / ((beta**2 * p1) + p2)
#     return f


# def calc_sentence_similarity(s1, s2):
#     global parser_model
#     tokens1 = nltk.word_tokenize(s1)
#     tokens2 = nltk.word_tokenize(s2)
#     parsed_s1 = parser_model.predict([tokens1])
#     parsed_s2 = parser_model.predict([tokens2])
#     f_arcs = rouge(parsed_s1.arcs[0], parsed_s2.arcs[0])
#     f_rels = rouge(parsed_s1.rels[0], parsed_s2.rels[0])
#     f_tokens = rouge(tokens1, tokens2)
#     return f_arcs, f_rels, f_tokens


# def match_answer_iclformat(infer_result:dict, round_idx, args):
#     task_config = args.tasks_config["iclformat"]
#     init(task_config["nltk_path"], task_config["parser_path"])
#     result = {}
#     acc_sum = 0
#     for subject in task_config["subjects"]:
#         if subject in ["sentence"]:
#             f_arcs_sum = 0
#             f_rels_sum = 0
#             f_tokens_sum = 0
#             rouge_sum = 0
#         else:
#             correct_cnt = 0
#         for item in infer_result[subject]:
#             flag = False
#             if subject in ["sentence"]:
#                 pass
#             else:
#                 item[f"judge_round{round_idx}"] = False
#             model_response = item[f"infer_round{round_idx}"].split("<output>", 1)[-1].strip()
#             item["model_response"] = model_response
#             flag = False
#             if subject in ["format_tree", "struct_to_struct", "struct_to_text", "text_to_struct", "text_to_text"]:
#                 if model_response == item["output"].strip():
#                     flag = True
#             elif subject in ["format_answer", "format_choice"]:
#                 if re.match(item["pattern"], model_response):
#                     flag = True
#             elif subject == "bullet_pointed_response":
#                 flag = True
#                 for label in item["label_list"][:3]:
#                     if label not in model_response:
#                         flag = False
#             elif subject in ["sentence"]:
#                 f_arcs, f_rels, f_tokens = calc_sentence_similarity(model_response, item["input"])
#                 item[f"f_arcs_round{round_idx}"] = f_arcs
#                 item[f"f_rels_round{round_idx}"] = f_rels
#                 item[f"f_tokens_round{round_idx}"] = f_tokens
#                 f_arcs_sum += f_arcs
#                 f_rels_sum += f_rels
#                 f_tokens_sum += f_tokens
#                 rouge_sum += max(f_arcs, f_rels)
#             if flag:
#                 correct_cnt += 1
#                 item[f"judge_round{round_idx}"] = True

#         total = len(infer_result[subject])
#         if subject in ["sentence"]:
#             result[subject] = {
#                 "rouge_avg": rouge_sum / total,
#                 "f_arcs_avg": f_arcs_sum / total,
#                 "f_rels_avg": f_rels_sum / total,
#                 "f_tokens_acg": f_tokens / total,
#                 "total_cnt": total
#             }
#             acc_sum += rouge_sum / total
#         else:
#             acc = correct_cnt / total
#             acc_sum += acc
#             result[subject] = {
#                 "acc": acc,
#                 "correct_cnt": correct_cnt,
#                 "total_cnt": total
#             }

#     result["iclformat"] = {
#         "acc": acc_sum / len(task_config["subjects"])
#     }
#     return result


def match_answer_iclformat(infer_result:dict, round_idx, args):
    return {"iclformat": "skipped"}