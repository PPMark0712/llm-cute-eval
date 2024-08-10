def format_prompt_default(query, history):
    HUMAN_BEGIN = ""
    ASSISTANT_BEGIN = ""
    prompt = ""
    for old_query, response in history:
        prompt += HUMAN_BEGIN + old_query
        prompt += ASSISTANT_BEGIN + response
        prompt += "\n"
    prompt += HUMAN_BEGIN + query + ASSISTANT_BEGIN
    return prompt
