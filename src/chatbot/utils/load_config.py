import yaml
import openai
from dotenv import load_dotenv
import os
from pyprojroot import here

load_dotenv()


class LoadConfig:
    def __init__(self) -> None:
        with open(here("configs/config.yml")) as cfg:
            app_config = yaml.load(cfg, Loader=yaml.FullLoader)

        # memory
        self.memory_directory: str = app_config["memory"]["directory"]
        self.num_entities: int = app_config["memory"]["num_entities"]

        # llm_function_caller
        self.llm_function_caller_temperature: float = app_config["llm_function_caller"][
            "temperature"
        ]
        self.llm_function_caller_system_role: str = app_config["llm_function_caller"][
            "sytem_role"
        ]
        self.llm_function_caller_gpt_model: str = app_config["llm_function_caller"][
            "gpt_model"
        ]

        # Summarizer config
        self.llm_inference_gpt_model: str = app_config["llm_inference"]["gpt_model"]
        self.llm_inference_sytem_role: str = app_config["llm_inference"]["system_role"]
        self.llm_inference_temperature = app_config["llm_inference"]["temperature"]

        self.load_open_ai_credentials(self)

        def _load_open_ai_credentials(self):
            """
            Note:
            Replace "Your API TYPE," "Your API BASE," "Your API VERSION," and "Your API KEY" with your actual
            OpenAI API credentials.
            """

        openai.api_type = os.getenv("OPENAI_API_TYPE")
        openai.api_base = os.getenv("OPENAI_API_BASE")
        openai.api_version = os.getenv("OPENAI_API_VERSION")
        openai.api_key = os.getenv("OPENAI_API_KEY")
