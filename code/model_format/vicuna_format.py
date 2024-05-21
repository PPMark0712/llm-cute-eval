def format_vicuna_prompt(task_prompt, query, history):
    prompt = ""
    if history == ():
        prompt += f"<s>USER: \n\n{task_prompt}\n\nQ: {query} \n ASSISTANT: A:"
    else:
        for i, (old_query, response) in enumerate(history):
            if i == 0:
                prompt += f"<s>USER: \n\n{task_prompt}\n\nQ: {old_query} \n ASSISTANT: A:{response} \n\n</s>"
            else:
                prompt += f"<s>USER: Q: {old_query} ASSISTANT: A:{response} \n\n</s>"
        prompt += f"<s>USER: Q: {query} ASSISTANT: A:"
    return prompt