# import os 
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import evaluate
from datasets import load_dataset, Dataset
import pandas as pd
import numpy as np
import nltk
import yaml
from torch.nn.parallel import DistributedDataParallel
from transformers import (DataCollatorForSeq2Seq, 
                          AutoTokenizer, 
                          AutoModelForCausalLM, 
                          BloomForCausalLM,
                          Seq2SeqTrainingArguments, 
                          Seq2SeqTrainer,
                          Trainer,
                          TrainerCallback,
                          )

import random
random.seed(42)
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType

"""
Code to finetune FLOR-760M to the synthetic dataset generated with LLAMA3-70B in a two-step process

First, the model is fine-tuned with CaBreu dataset, with the weights of the word embeddings frozen
Then, the model is fine-tuned, using LoRA, with the synthetic dataset generated with LLAMA3-70B
"""

model_name_pretrain = 'Pretrain_FLOR_Articles_summarization_v2'
model_name_finetune = 'LoRA_Finetune_MedicalReports_v2'

def preprocess_function(examples):
    reports = ["Resumeix: " + report for report in examples["content"]]
    # Tokenize the inputs and the targets
    reports_input = tokenizer(reports, max_length=1300, truncation=True, padding=False)

    # Select a random summary
    summaries= []
    for summary in examples["summaries"]:
        i = random.randint(1, 3)
        summary = summary["extractive"][f"a{i}"]
        summaries.append("\nResum:\n" + summary)
        
    # Tokenize the inputs and the targets
    summaries_input = tokenizer(summaries, max_length=500, truncation=True, padding=False)
    
    # Concatenate the reports and the summaries inputsids
    model_inputs = {"input_ids": [reports_input["input_ids"][i] + summaries_input["input_ids"][i] for i in range(len(reports_input["input_ids"]))]}
    model_inputs["labels"] = model_inputs["input_ids"]
    return model_inputs

# Load the tokenizer
checkpoint = "projecte-aina/FLOR-760M"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Add padding tokens
tokenizer.pad_token = tokenizer.eos_token


# Loading CaBreu dataset (https://huggingface.co/datasets/projecte-aina/caBreu)
dataset = load_dataset("projecte-aina/caBreu")

dataset_train = dataset["train"]
for instance in dataset["validation"]:
    dataset_train.add_item(instance)
    
dataset_eval = dataset["test"]

# Preprocess the dataset
tokenized_train = dataset_train.map(preprocess_function, batched=True)
tokenized_eval = dataset_eval.map(preprocess_function, batched=True)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)

# Load the pretrained model
model = BloomForCausalLM.from_pretrained(checkpoint, attention_dropout=0.2, hidden_dropout=0.2)

# Freeze the model word embeddings
for param in model.named_parameters():
    if "word_embeddings" in param[0]:
        print(param[0])
        param[1].requires_grad = False

# Load a yaml file with the training arguments
with open(f"/hhome/nlp2_g05/Asho_NLP/configs_weights/{model_name_pretrain}/config.yaml", "r") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

config['training_args']['output_dir'] = f"/hhome/nlp2_g05/Asho_NLP/src/Trained_models/{model_name_pretrain}"
training_args = Seq2SeqTrainingArguments(
    **config['training_args'])


# In order to chose randomly one of the three sumaries at the start of each epoch
modify_dataset = TrainerCallback()
def modify_dataset_fn(trainer, datasets, epoch, **kwargs):
    print("Modifying dataset")
    # Modify the dataset
    tokenized_train = dataset_train.map(preprocess_function, batched=True)
    tokenized_eval = dataset_eval.map(preprocess_function, batched=True)
    
    trainer.train_dataset = tokenized_train
    trainer.eval_dataset = tokenized_eval

# Modify on epoch beging
modify_dataset.on_epoch_begin = modify_dataset_fn


# Initialize the trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=[modify_dataset],
)
print(len(trainer.train_dataset))

# Train the model
trainer.train()

# Save the logs
import json
with open(f"/hhome/nlp2_g05/Asho_NLP/src/Logs/logs_{model_name_pretrain}.json", "w") as f:
    json.dump(trainer.state.log_history, f)


# Save the pretrained model
trainer.save_model(f"/hhome/nlp2_g05/Asho_NLP/src/Trained_models/last_model_{model_name_pretrain}")


del model
del trainer
del tokenized_train
del tokenized_eval
del dataset_train
del dataset_eval
del dataset

# Finetune with LoRA for Medical reports summarization
checkpoint = f"/hhome/nlp2_g05/Asho_NLP/src/Trained_models/last_model_{model_name_pretrain}"
model = BloomForCausalLM.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM, 
    inference_mode=False, 
    r=8, 
    lora_alpha=30, 
    lora_dropout=0.2
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()


# Getting the medical reports data

def preprocess_function(examples):
    reports = ["Resume: " + report if lang == "es" else "Resumeix:" + report for report, summary, lang in zip(examples["Text"], examples["summary"], examples["language"])]
    # Tokenize the inputs and the targets
    reports_input = tokenizer(reports, max_length=1900, truncation=True, padding=False)

    summaries= ["\nResumen:\n" + summary if lang == "es" else "\nResum:\n" + summary for report, summary, lang in zip(examples["Text"], examples["summary"], examples["language"])]
    # Tokenize the inputs and the targets
    summaries_input = tokenizer(summaries, max_length=300, truncation=True, padding=False)

    # Concatenate the reports and the summaries inputsids
    model_inputs = {"input_ids": [reports_input["input_ids"][i] + summaries_input["input_ids"][i] for i in range(len(reports_input["input_ids"]))]}
    model_inputs["labels"] = model_inputs["input_ids"]
    return model_inputs

# Read json sumaries generated with chatgpt API
df = pd.read_json('/hhome/nlp2_g05/Asho_NLP/src/CLEANED_final_summaries_llama3_70B.json')
# Changing the columns by the rows
df = df.T

# Add language column to the dataframe
df["language"] = ["es"] * len(df)

# Read json sumaries generated with llama3
df_llama = pd.read_json('/hhome/nlp2_g05/Asho_NLP/src/CLEANED_CAT_final_summaries_llama3_70B.json')
df_llama = df_llama.T
df_llama["language"] = ["cat"] * len(df_llama)

# Reading the first two examples from the Asho dataset, which where used to generate the summaries
asho_train_gt = pd.read_json("/hhome/nlp2_g05/Asho_NLP/src/Datasets/Asho_Dataset/train_cleaned.json")
example_text_0 = asho_train_gt["Text"][5]
example_text_1 = asho_train_gt["Text"][6]
example_summary_0 = asho_train_gt["Summary"][5]
example_summary_1 = asho_train_gt["Summary"][6]

# Add the two examples to the dataframe
new_row = {"Text": example_text_0, "summary": example_summary_0}
df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
new_row = {"Text": example_text_1, "summary": example_summary_1}
df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

# Join ChatGPT and llama3 summaries
df = pd.concat([df, df_llama], ignore_index=True)

# Convert the Pandas dataFrame to a Hugging Face Dataset
dataset = Dataset.from_pandas(df.astype(str))

splits = dataset.train_test_split(test_size=0.1, seed=42)
dataset_train = splits["train"]
dataset_eval = splits["test"]


# Preprocess the dataset
tokenized_train = dataset_train.map(preprocess_function, batched=True)
tokenized_eval = dataset_eval.map(preprocess_function, batched=True)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)



# Load a yaml file with the training arguments
with open(f"/hhome/nlp2_g05/Asho_NLP/configs_weights/{model_name_finetune}/config.yaml", "r") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

config['training_args']['output_dir'] = f"/hhome/nlp2_g05/Asho_NLP/src/Trained_models/{model_name_finetune}"
training_args = Seq2SeqTrainingArguments(
    **config['training_args'])

# Initialize the trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Train the model
trainer.train()

# Save the logs
import json
with open(f"/hhome/nlp2_g05/Asho_NLP/src/Logs/logs_{model_name_finetune}.json", "w") as f:
    json.dump(trainer.state.log_history, f)


# Save the best model
trainer.save_model(f"/hhome/nlp2_g05/Asho_NLP/src/Trained_models/last_model_{model_name_finetune}")