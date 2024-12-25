from typing import List
from datasets import load_dataset
from pyprojroot import here
import yaml
from functools import partial

with open(here("config.yml"), "r") as ymlfile:
    app_config = yaml.load(ymlfile, Loader=yaml.FullLoader)

newton_qa_dir = str(
    here(app_config["training_data_dir"]["newtontools_questions_answer"])
)
newton_instructions_response = str(
    here(app_config["training_data_dir"]["newtontools_instruction_response"])
)

tokenizer_max_length = 2048


def tokenize_data(
    examples,
    tokenizer,
    tokenizer_max_length: int = tokenizer_max_length,
    column_name: List = ["question", "answer"],
    data_type: str = "newtontools",
):
    if data_type == "newtontools":
        text = examples[column_name[0][0]] + examples[column_name[1][1]]
    elif data_type == "guanaco":
        text = examples["text"][0]
    else:
        raise ValueError("data_type should be either newtontools or guanaco")

    tokenizer.pad_token = tokenizer.eos_token
    tokenized_inputs = tokenizer(text, retun_tensors="pt", padding=True)

    max_length = min(tokenizer_max_length, tokenized_inputs["input_ids"].shape[1])

    tokenizer.truncation_side = "left"
    tokenized_inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=max_length
    )

    return tokenized_inputs


def prepare_pq_datasets(
    tokenizer,
    tokenizer_max_length: int = tokenizer_max_length,
    colums: List = ["question", "answer"],
    data_dir: str = newton_qa_dir,
    data_type: str = "newtontools",
):
    finetuning_datasets = load_dataset("json", data_files=data_dir, split="train")
    print(" raw dataset shape", finetuning_datasets.shape)

    # define a partial function to pass the tokenizer and other arguments

    partial_tokenize_func = partial(
        tokenize_data,
        tokenizer=tokenizer,
        tokenizer_max_length=tokenizer_max_length,
        column_name=colums,
        data_type=data_type,
    )

    tokenized_dataset = finetuning_datasets.map(
        partial_tokenize_func, batched=True, batch_size=1, drop_last_batch=True
    )

    tokenized_data = tokenized_dataset.add_column(
        "labels", tokenized_dataset["input_ids"]
    )

    return tokenized_data


def prepapre_instrct_response(
    tokenizer,
    tokenizer_max_length: int = tokenizer_max_length,
    column_name: List = ["instruction", "response"],
    data_dir: str = newton_instructions_response,
    data_type: str = "newtontools",
):
    finetuning_datasets = load_dataset("json", data_files=data_dir, split="train")

    partial_tokenize_func = partial(
        tokenize_data,
        tokenizer=tokenizer,
        tokenizer_max_length=tokenizer_max_length,
        column_name=column_name,
        data_type=data_type,
    )
    tokenized_dataset = finetuning_datasets.map(
        partial_tokenize_func, batched=True, batch_size=1, drop_last_batch=True
    )

    tokenized_data = tokenized_dataset.add_column(
        "labels", tokenized_dataset["input_ids"]
    )

    return tokenized_data
