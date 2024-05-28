from datasets import load_dataset, Dataset
import pandas as pd

# Read jsonl files
df = pd.read_json('/hhome/nlp2_g05/Asho_NLP/src/chatgpt_summaries_gpt4.json')
# Changing the columns by the rows
df = df.T


# Convert the Pandas dataFrame to a Hugging Face Dataset
dataset = Dataset.from_pandas(df.astype(str))

splits = dataset.train_test_split(test_size=0.1, seed=42)
dataset_train = splits["train"]
dataset_eval = splits["test"]

print(dataset_train[0]['Text'])