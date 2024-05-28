import evaluate
from datasets import load_dataset, Dataset
import pandas as pd
import numpy as np
import nltk
import yaml
from transformers import (DataCollatorForSeq2Seq, 
                          AutoTokenizer, 
                          AutoModelForCausalLM, 
                          Seq2SeqTrainingArguments, 
                          Seq2SeqTrainer
                          )

def preprocess_function(examples):
    reports = ["sumarize: " + report for report, summary in zip(examples["Text"], examples["summary"])]
    # Tokenize the inputs and the targets
    reports_input = tokenizer(reports, max_length=1800, truncation=True)

    print("REPORTS:")
    # Avergae the length of the inputs
    print("Mean lenght:", np.mean([len(reports_input["input_ids"][i]) for i in range(len(reports_input["input_ids"]))]))
    
    # Median the length of the inputs
    print("Median lenght:", np.median([len(reports_input["input_ids"][i]) for i in range(len(reports_input["input_ids"]))]))
    # Max and min the length of the inputs
    print("Max lenght:", np.max([len(reports_input["input_ids"][i]) for i in range(len(reports_input["input_ids"]))]))
    print("Min lenght:", np.min([len(reports_input["input_ids"][i]) for i in range(len(reports_input["input_ids"]))]))
    
    summaries= ["\nSummary:\n" + summary for report, summary in zip(examples["Text"], examples["summary"])]
    # Tokenize the inputs and the targets
    summaries_input = tokenizer(summaries, max_length=450, truncation=True)
    print("SUMMARIES:")
    # Avergae the length of the inputs
    print("Mean lenght:", np.mean([len(summaries_input["input_ids"][i]) for i in range(len(summaries_input["input_ids"]))]))
    
    # Median the length of the inputs
    print("Median lenght:", np.median([len(summaries_input["input_ids"][i]) for i in range(len(summaries_input["input_ids"]))]))
    # Max and min the length of the inputs
    print("Max lenght:", np.max([len(summaries_input["input_ids"][i]) for i in range(len(summaries_input["input_ids"]))]))
    print("Min lenght:", np.min([len(summaries_input["input_ids"][i]) for i in range(len(summaries_input["input_ids"]))]))
    
    # Concatenate the reports and the summaries inputsids
    model_inputs = {"input_ids": [reports_input["input_ids"][i] + summaries_input["input_ids"][i] for i in range(len(reports_input["input_ids"]))]}
    model_inputs["labels"] = model_inputs["input_ids"]
    return model_inputs

def compute_metrics(eval_pred):
    # Get the predictions and the target texts
    predictions, labels = eval_pred

    # Decode into tokens the predictions and the targets
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace the -100 tokens with the pad token
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Compute the ROUGE metrics
    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    # Compute the generation length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}


# Load the tokenizer
checkpoint = "projecte-aina/FLOR-760M"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Add padding tokens
tokenizer.pad_token = tokenizer.eos_token


# Read jsonl files
df = pd.read_json('/hhome/nlp2_g05/Asho_NLP/src/chatgpt_summaries_gpt4.json')
# Changing the columns by the rows
df = df.T


# Convert the Pandas dataFrame to a Hugging Face Dataset
dataset = Dataset.from_pandas(df.astype(str))

splits = dataset.train_test_split(test_size=0.1, seed=42)
dataset_train = splits["train"]
dataset_eval = splits["test"]

# Preprocess the dataset
tokenized_train = dataset_train.map(preprocess_function, batched=True)
tokenized_eval = dataset_eval.map(preprocess_function, batched=True)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)

# Load the ROUGE evaluator
rouge = evaluate.load("rouge")

# Load the model
model = AutoModelForCausalLM.from_pretrained(checkpoint, attention_dropout= 0.2, hidden_dropout=0.2)


# Define the training arguments

# Load a yaml file with the training arguments
model_name = 'gpt_drop'
with open(f"/hhome/nlp2_g05/Asho_NLP/configs_weights/{model_name}/config.yaml", "r") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

config['training_args']['output_dir'] = f"./Trained_models/{model_name}"
training_args = Seq2SeqTrainingArguments(
    **config['training_args'])

# Initialize the trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    tokenizer=tokenizer,
    data_collator=data_collator)

# Train the model
trainer.train()

# Save the logs
import json
with open(f"./Logs/logs_{model_name}.json", "w") as f:
    json.dump(trainer.state.log_history, f)


# Save the best model
trainer.save_model(f"./Trained_models/last_model_{model_name}")


# Create summaries for each report