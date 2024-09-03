from typing import (
    Literal,
    Sequence,
    TypedDict,
)

Role = Literal["user", "assistant"]


class Message(TypedDict):
    role: Role
    content: str


Dialog = Sequence[Message]


class ChatFormat:
    def __init__(self):
        self.input = ""

    def add_header(self, message: Message):
        tokens = []
        tokens.append("<start_of_turn>")
        tokens.extend(message["content"].strip())
        tokens.append("\n")
        return tokens

    def add_message(self, message: Message):
        tokens = self.add_header(message)
        tokens.extend(message["role"].strip())
        tokens.append("<end_of_turn>")
        tokens.append("\n")
        return tokens

    def add_dialog_prompt(self, dialog: Dialog):
        tokens = []
        for message in dialog:
            tokens.extend(self.add_message(message))
        # Add the start of an assistant message for the model to complete.
        tokens.extend(self.add_header({"role": "assistant", "content": ""}))
        return "".join(tokens)  # Return a single string concatenated from the list of tokens.


def format_prompt_gemma(query, history):
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