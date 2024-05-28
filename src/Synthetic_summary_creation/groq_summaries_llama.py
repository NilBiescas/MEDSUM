import os
from groq import Groq
import json
import pandas as pd
from tqdm import tqdm


"""
Code to generate summaries in Spanish with GROQ API using the model llama3-70b-8192. 
It uses two examples from the Asho dataset to generate summaries in Spanish for the rest of data.
"""

GROQ_API_KEY = ''
FILENAME = "final_summaries_llama3_70B"

os.environ['GROQ_API_KEY'] = GROQ_API_KEY
client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

# Template string to create the input text
template_string = """
Para resumir utiliza este ejemplo:
Ejemplo de texto 1: {example_text}
Ejemplo de resumen detallado 1: {example_summary}

Ejemplo de texto 2: {example_text2}
Ejemplo de resumen detallado 2: {example_summary2}

Ahora, por favor, resume el siguiente texto médico crudo. 
Asegúrate de incluir información relevante sobre el 
paciente y medicamentos prescritos. Utiliza los ejemplos 
anteriores como base para proporcionar un resumen 
detallado y relevante. El resúmenes debe tener entre 50 y 130 palabras.
Texto a resumir: {text2summarize}
Resumen Detallado: 

"""


### Example data ##
asho_train = pd.read_json("/hhome/nlp2_g05/Asho_NLP/src/Datasets/Asho_Dataset/train_cleaned.json")
example_text = asho_train["Text"][5]
example_summary = asho_train["Summary"][5]

example_text2 = asho_train["Text"][6]
example_summary2 = asho_train["Summary"][6]

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

# Load the json file
with open(f"/hhome/nlp2_g05/Asho_NLP/src/{FILENAME}.json", "r") as f:
    out_summaries = json.load(f)


for idx, text in tqdm(enumerate(texts2summarize), total=len(texts2summarize)):        
    try:
        if str(idx) in out_summaries.keys():
            continue
        
        # Create the input text
        input_text = template_string.format(
            example_text=example_text,
            example_summary=example_summary,
            example_text2=example_text2,
            example_summary2=example_summary2,
            text2summarize=text
        )
        
        # Get the model output from the GROQ API
        chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": input_text,
                    }
                ],
                model="llama3-70b-8192",
                )
        content = chat_completion.choices[0].message.content

        # Remove the prompt from the output
        summary = content.replace(input_text, "")
        out_summaries[f'{idx}'] = {"Text": text, "summary": summary}
        print("Finished with: ", idx, flush=True)

        # Save the summaries every 10 iterations
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

