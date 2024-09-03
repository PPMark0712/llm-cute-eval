def format_prompt_llama2(query, history):
    llama2_overall_instruction = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
    prompt = ""
    if len(history) == 0:
        prompt += f"<s>[INST] <<SYS>>\n\n{llama2_overall_instruction}\n\n<</SYS>>\n\n{query} [/INST]\n"
    else:
        for i, (old_query, response) in enumerate(history):
            if i == 0:
                prompt += f"<s>[INST] <<SYS>>\n\n{llama2_overall_instruction}\n\n<</SYS>>\n\n{old_query} [/INST] {response} </s>"
            else:
                prompt += f"<s>[INST] {old_query} [/INST]\n\n{response} </s>"
        prompt += f"<s>[INST]\n\n {query} \n\n[/INST]\n\n"
    return prompt