from typing import List, Dict
import openai


class InferenceGPT:
    @staticmethod
    def prepare_massages(
        llm_response: str,
        user_query: str,
        llm_system_role: str,
        input_chat_history: str,
    ) -> List[Dict]:
        query = f"# chat history: {input_chat_history}\n\n, # Newton Triangle LLM response:\n\n{llm_response}\n\n, # User's new query: {user_query}"
        messages = [
            {"role": "system", "content": llm_system_role},
            {"role": "user", "content": query},
        ]

        return messages

    @staticmethod
    def ask(gpt_model: str, temperature: float, messages: List):
        """
        Generate a response from an OpenAI ChatCompletion API call without specific function calls.

        Parameters:
            - gpt_model (str): The name of the GPT model to use.
            - temperature (float): The temperature parameter for the API call.
            - messages (List): List of message objects for the conversation.

        Returns:
            The response object from the OpenAI ChatCompletion API call.
        """

        response = openai.ChatCompletion.create(
            engine=gpt_model, messages=messages, temperature=temperature
        )

        return response
