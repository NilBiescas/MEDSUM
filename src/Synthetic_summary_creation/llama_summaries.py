import os
import pandas as pd
import json
import transformers
import torch
from tqdm import tqdm

"""
Code to generate summaries in Spanish using localy llama3-8b. 
It uses one example from the Asho dataset to generate summaries in Spanish for the rest of data.
"""

FILENAME = "short_summaries_llama3_8b"

model_id = "/hhome/nlp2_g05/Meta-Llama-3-8B-Instruct-transofrmers-format"

pipeline = transformers.pipeline(
    "text-generation", model=model_id, 
    model_kwargs={"torch_dtype": torch.bfloat16}, 
    device_map="auto", 
    min_length=60,
    max_new_tokens=150,
)

tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)

template_string = """
Para resumir utiliza este ejemplo:
Ejemplo de texto: {example_text}
Ejemplo de resumen: {example_summary}

Ahora, por favor, resume el siguiente texto. Asegúrate de incluir información relevante sobre el paciente y medicamentos prescritos. 
Utiliza el ejemplo anterior como base para proporcionar el resumen.
El resúmenes debe tener entre 50 y 130 palabras.
Texto a resumir: {text2summarize}
Resumen: 

"""
### Load the data ###

### Example data ##
asho_train = pd.read_json("/hhome/nlp2_g05/Asho_NLP/src/Datasets/Asho_Dataset/train_cleaned.json")
example_text = asho_train["Text"][5]
example_summary = asho_train["Summary"][5]

## Data to summarize ##

# Data-1 (Last year data)
with open("/hhome/nlp2_g05/Asho_NLP/src/Datasets/Asho_negation_Dataset/neg_data_cleaned.json", "r") as f:
    negations_data = json.load(f)
negations_data_text = [negations_data[key] for key in negations_data]

# Data-2
asho_no_summaries = pd.read_json("/hhome/nlp2_g05/Asho_NLP/src/Datasets/Asho_Dataset/asho_no_summaries.json")

# Both data
texts2summarize = list(asho_no_summaries['Text'].values) + negations_data_text

# If the json file does not exist, create it
if not os.path.exists(f"/hhome/nlp2_g05/Asho_NLP/src/{FILENAME}.json"):
    with open(f"/hhome/nlp2_g05/Asho_NLP/src/{FILENAME}.json", "w") as f:
        json.dump({}, f)

with open(f"/hhome/nlp2_g05/Asho_NLP/src/{FILENAME}.json", "r") as f:
    out_summaries = json.load(f)

try:
    for idx, text in tqdm(enumerate(texts2summarize), total=len(texts2summarize)):        
        if str(idx) in out_summaries.keys():
            continue
        
        # Create the input text
        input_text = template_string.format(
            example_text=example_text,
            example_summary=example_summary,    
            text2summarize=text
        )

        # Get the model output
        out = pipeline(input_text)
        # Remove the prompt from the output
        summary = out[0]["generated_text"].replace(input_text, "")
        out_summaries[f'{idx}'] = {"Text": text, "summary": summary}
        print("Finished with: ", idx, flush=True)
        
        # Save the output every 10 iterations
        if idx % 10 == 0:
            with open(f"/hhome/nlp2_g05/Asho_NLP/src/{FILENAME}.json", "w") as f:
                json.dump(out_summaries, f, indent=4)
        
except Exception as e:
    print("Error: ", e)
    print("Last idx: ", idx)
    pass

with open(f"/hhome/nlp2_g05/Asho_NLP/src/{FILENAME}.json", "w") as f:
    json.dump(out_summaries, f, indent=4)
    # Add ident
    