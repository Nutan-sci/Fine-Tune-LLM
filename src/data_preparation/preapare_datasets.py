import pandas as pd
import yaml
import os
import jsonlines
from pyprojroot import here
from typing import List


def prepare_qa_dataset(data_dir):
    """preparing Question and Answer dataset for processing
    Parameter
    - file_path : json file cotaining Question and answers
    - return List[dict]: List of dictionaries with Question and answer
    """
    df = pd.read_json(str(here(data_dir)))
    print("data shape", df.shape)
    finetune_data = []
    for i in range(len(df["question"])):
        question = f"### Question:\n{df['question'][i]}\n\n\n### Answer:\n"
        answer = df["answer"][i]
        finetune_data.append({"question": question, "answer": answer})

    return finetune_data


def prepare_instruction_response_dataset(data_dir):
    df = pd.read_json(str(here(data_dir)))
    print("data shape", df.shape)
    finetune_data = []
    for i in range(len(df["instruction"])):
        instruction = f"### Instruction:\n{df['instruction'][i]}\n\n\n### Response:\n"
        response = df["response"][i]
        finetune_data.append({"isntruction": instruction, "response": response})

    return finetune_data


if __name__ == "__main__":
    with open(here("configs/config.yml")) as cfg:
        app_config = yaml.load(cfg, Loader=yaml.FullLoader)
    dataset1 = prepare_instruction_response_dataset(
        app_config["json_dir"]["product_user_manual_json"]
    )
    dataset2 = prepare_qa_dataset(app_config["json_dir"]["question_answer_json"])

    # with jsonlines.open(
    #     here(app_config["training_data_dir"]["newtontools_instruction_response"]), "w"
    # ) as writer:
    #     writer.write_all(dataset1)

    with jsonlines.open(
        here(app_config["training_data_dir"]["newtontools_questions_answer"]), "w"
    ) as writer:
        writer.write_all(dataset2)
