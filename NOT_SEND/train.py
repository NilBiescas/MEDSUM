import evaluate
from datasets import load_dataset, Dataset
import pandas as pd
import numpy as np
import nltk
from transformers import (DataCollatorForSeq2Seq, 
                          AutoTokenizer, 
                          AutoModelForCausalLM, 
                          Seq2SeqTrainingArguments, 
                          Seq2SeqTrainer
                          )

def preprocess_function(examples):
    inputs = ["sumarize: " + report + "\nThe summary is:\n" + summary 
              for report, summary in zip(examples["inputs"], examples["target"])]
    # Tokenize the inputs and the targets
    model_inputs = tokenizer(inputs, max_length=256, truncation=True)

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
df_train = pd.read_json('/hhome/nlp2_g05/Asho_NLP/train.jsonl', lines=True).drop(columns=['idx'])
df_val = pd.read_json('/hhome/nlp2_g05/Asho_NLP/validate.jsonl', lines=True).drop(columns=['idx'])
df_test = pd.read_json('/hhome/nlp2_g05/Asho_NLP/test.jsonl', lines=True).drop(columns=['idx'])



print(df_train.head())
# Join the dataframes for training and validation
df = pd.concat([df_train, df_val])

# Convert the Pandas dataFrame to a Hugging Face Dataset
dataset_train = Dataset.from_pandas(df.astype(str))
dataset_val = Dataset.from_pandas(df_test.astype(str))

# Preprocess the dataset
tokenized_train = dataset_train.map(preprocess_function, batched=True)
tokenized_eval = dataset_val.map(preprocess_function, batched=True)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)

# Load the ROUGE evaluator
rouge = evaluate.load("rouge")

# Load the model
model = AutoModelForCausalLM.from_pretrained(checkpoint)

# Freeze the model word embeddings
for param in model.named_parameters():
    if "word_embeddings" in param[0]:
        print(param[0])
        param[1].requires_grad = False


# Define the training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./Trained_models/model",
    evaluation_strategy="no",
    # save_strategy="epoch",
    learning_rate=5.6e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    weight_decay=0.01,
    num_train_epochs=10,
    logging_strategy="epoch",
    predict_with_generate=True,
    save_total_limit=3,
    fp16=True,
    push_to_hub=False,
    # load_best_model_at_end=True,
)

# Initialize the trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    tokenizer=tokenizer,
    data_collator=data_collator,
    # compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Save the logs
import json
with open("./Logs/logs.json", "w") as f:
    json.dump(trainer.state.log_history, f)


# Save the best model
trainer.save_model("./Trained_models/best_model")
