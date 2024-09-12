from typing import (
    Literal,
    Sequence,
    TypedDict,
)


class Message(TypedDict):
    role: Literal["user", "assistant"]
    content: str


class ChatFormat:
    def __init__(self):
        self.input = ""

    def add_header(self, message: Message):
        tokens = []
        tokens.append("<|start_header_id|>")
        tokens.extend(message["role"])
        tokens.append("<|end_header_id|>")
        tokens.extend("\n\n")
        return tokens

    def add_message(self, message: Message):
        tokens = self.add_header(message)
        tokens.extend(message["content"].strip())
        tokens.append("<|eot_id|>")
        return tokens

    def add_dialog_prompt(self, dialog: Sequence[Message]):
        tokens = []
        tokens.append("<|begin_of_text|>")
        for message in dialog:
            tokens.extend(self.add_message(message))
        # Add the start of an assistant message for the model to complete.
        tokens.extend(self.add_header({"role": "assistant", "content": ""}))
        return "".join(tokens)  # Return a single string concatenated from the list of tokens.


def format_prompt_llama3(query, history):
    format = ChatFormat()
    if len(history) == 0:
        dialog = [
            {
                "role": "user",
                "content": query,
            },
        ]
    else:
        dialog = []
        for i, (old_query, response) in enumerate(history):
            if i == 0:
                dialog.extend([
                    {
                        "role": "user",
                        "content": f"{old_query}",
                    },
                    {
                        "role": "assistant",
                        "content": f"{response}",
                    },
                ])
            else:
                dialog.extend([
                    {
                        "role": "user",
                        "content": f"{old_query}",
                    },
                    {
                        "role": "assistant",
                        "content": f"{response}",
                    },
                ])
        dialog.extend([
            {
                "role": "user",
                "content": f"{query}",
            },
        ])
    question = format.add_dialog_prompt(dialog)
    return question