import warnings
import os
import openai
import pandas as pd
import json
import time

from openai import OpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

warnings.filterwarnings("ignore")

API_KEY = "sk-proj-ZUsn1aTIord6YX34LpkyT3BlbkFJrddrioZJDT68LZeHCitF"

openai.api_key = API_KEY
llm_model = "gpt-3.5-turbo-0125"
os.environ["OPENAI_API_KEY"] = API_KEY

chat = ChatOpenAI(temperature=0.0, model=llm_model, max_tokens=1000)

template_string = """
Para resumir, mira este ejemplo:
Ejemplo de texto: {ejemplo_text}
Ejemplo de resumen: {ejemplo_resumen}

Ahora, por favor, resume el siguiente texto médico crudo. Asegúrate de incluir información relevante sobre los síntomas del paciente y los medicamentos prescritos. Utiliza como base el ejemplo anterior para proporcionar un resumen detallado y relevante.
Los resumenes tienen que ser LARGOS, LARGOS, de unas 500 palabras. El resumen tiene que ser en la lengua original del texto, si esta en catalan hazlo en catalan.
Texto a resumir: ```{text}```
"""

prompt_template = ChatPromptTemplate.from_template(template_string)
### Load the data ###

### Example data ##
asho_train = pd.read_json("/hhome/nlp2_g05/Asho_NLP/src/Datasets/Asho_Dataset/train_cleaned.json")
asho_train_text = asho_train["Text"][0]
asho_train_summary = asho_train["Summary"][0]

## Data to summarize ##

# Data-1
with open("/hhome/nlp2_g05/Asho_NLP/src/Datasets/Asho_negation_Dataset/neg_data_cleaned.json", "r") as f:
    negations_data = json.load(f)
negations_data_text = [negations_data[key] for key in negations_data]

# Data-2
asho_no_summaries = pd.read_json("/hhome/nlp2_g05/Asho_NLP/src/Datasets/Asho_Dataset/asho_no_summaries.json")

# Both data
texts2summarize = list(asho_no_summaries['Text'].values) + negations_data_text

#### Summarize the data ####
ejemplo_text = asho_train_text
ejemplo_resumen = asho_train_summary

with open("/hhome/nlp2_g05/Asho_NLP/src/chatgpt_summaries_gpt4.json", "r") as f:
    out_summaries = json.load(f)

try:
    for idx,text in enumerate(texts2summarize):        
        if str(idx) in out_summaries.keys():
            continue
        
        customer_messages = prompt_template.format_messages(
                        ejemplo_text=ejemplo_text,
                        ejemplo_resumen=ejemplo_resumen,
                        text=text
                        )
        chat.invoke(customer_messages)
        customer_response = chat.invoke(customer_messages)
        out_summaries[f'{idx}'] = {"Text": text, "summary": customer_response.content}
        print("Finished with: ", idx, flush=True)
        # Save the summaries
        
except Exception as e:
    print("Error: ", e)
    print("Last idx: ", idx)
    pass

with open("/hhome/nlp2_g05/Asho_NLP/src/chatgpt_summaries_gpt4.json", "w") as f:
    json.dump(out_summaries, f, indent=4)
    # Add ident
    