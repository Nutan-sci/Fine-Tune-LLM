import openai
from typing import List, Dict


class LLMFunctioncaller:
    """Prepares a list of message dictionaries to be used by a language model.

    This method formats the system role information and input chat history, along with the user's message,
    into a structured list of dictionaries. Each dictionary contains a 'role' key to denote whether the
    message is from the 'system' or the 'user', and a 'content' key with the actual message content.
    """

    @staticmethod
    def prepare_messages(
        llm_function_caller_system_rple: str, input_chat_history: str, user_query: str
    ):
        query = f"# chat history: {input_chat_history}\n\n, # User's new query: {user_query}"

        return [
            {"role": "system", "content": str(llm_function_caller_system_rple)},
            {"role": "user", "content": query},
        ]

    @staticmethod
    def ask(
        gpt_model: str, temperature: float, messages: List, function_json_list: List
    ):
        response = openai.ChatCompletion.create(
            engine=gpt_model,
            messages=messages,
            functions=function_json_list,
            function_call="auto",
            temperature=temperature,
        )

        return response
