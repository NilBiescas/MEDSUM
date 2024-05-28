import os
import pandas as pd
import json
import transformers
import torch
from tqdm import tqdm

FILENAME = "Key_elements_llama3"

model_id = "/hhome/nlp2_g05/Meta-Llama-3-8B-Instruct-transofrmers-format"

pipeline = transformers.pipeline(
    "text-generation", model=model_id, 
    model_kwargs={"torch_dtype": torch.bfloat16}, 
    device_map="auto", 
    min_length=100,
    max_new_tokens=500,
    # num_beams=2,
    # early_stopping=True,
    # use_cache=True,
    # repetition_penalty=2.5,
    # no_repeat_ngram_size=2,
    # length_penalty=1.0,
    # do_sample = True,
    # temperature = 0.8,
    # top_k = 50,
    # top_p = 0.95,
)

tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
# Per provar el model
# out = pipeline("Continua el seguent text: Hola com estas avui?")

# print(out[0]["generated_text"])

# Define the template string
template_string = """
Mira este ejemplo de resumen:
Ejemplo de texto: {example_text}
Ejemplo de resumen: {example_summary}

Del siguiente texto haz una lista de los elementos clave necesarios para hacer un resumen como el del ejemplo. 
Asegúrate de incluir información relevante sobre el paciente y medicamentos prescritos.
Texto que usar: {text2summarize}
Elementos clave para el resumen: 

"""
### Load the data ###

### Example data ##
asho_train = pd.read_json("/hhome/nlp2_g05/Asho_NLP/src/Datasets/Asho_Dataset/train_cleaned.json")
example_text = asho_train["Text"][1]
example_summary = asho_train["Summary"][1]

## Data to summarize ##

# Data-1
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

# Load the json file
with open(f"/hhome/nlp2_g05/Asho_NLP/src/{FILENAME}.json", "r") as f:
    out_summaries = json.load(f)

try:
    for idx, text in tqdm(enumerate(texts2summarize), total=len(texts2summarize)):        
        if str(idx) in out_summaries.keys():
            continue
        # Create the input text using the template
        input_text = template_string.format(
            example_text=example_text,
            example_summary=example_summary,
            text2summarize=text
        )
        
        # Compute the number of tokens in the input text
        # num_tokens = len(tokenizer(text)["input_ids"])
        # print("Num tokens: ", num_tokens)
        # print("New tokens: ", int(num_tokens*0.25))
        
        out = pipeline(input_text)#, max_new_tokens=int(num_tokens*0.023))
        # Remove the prompt from the output
        summary = out[0]["generated_text"].replace(input_text, "")
        out_summaries[f'{idx}'] = {"Text": text, "summary": summary}
        print("Finished with: ", idx, flush=True)
        
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
    