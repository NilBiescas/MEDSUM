# Introduction

This project presents a comparison of different fine-tuning techniques for text summarization and an analysis of the currently available pretrained models in minority languages, such as Catalan. We propose an augmentation technique based on guidance from large language models (LLMs) that helps in domain adaptation from general knowledge learned during pretraining to specific smaller domains, such as those found in medical reports. By leveraging the capabilities of LLAMA 70B, we create a rich synthetic dataset from unannotated medical reports, which is used to guide the fine-tuning of smaller LLMs. This pipeline provides a systematic framework for fine-tuning smaller models in data-scarce domains.

A more detailed description of the project can be found [here](https://uab-my.sharepoint.com/:b:/g/personal/1523726_uab_cat/EajJ-LkzdnhBmF-npeYAL0kB8RmtREyLqSTeMfRxY1SXDg?e=QrEnTI).


# Repository Strucutre

- **src/Datasets**: Contains the different non-synthetic datasets used on experiments.
- **src/*.json:** Contains the different files of synthetic summaries generated for the experiments.
- **src/Utils/create_examples_and_eval_*.py**: contains the templates and code used for generating the synthetic dataset both catalan and spanish summaries.
- **src/Utils/gen_plots_from_logs.py**: Code to generate the training and validation loss evolution during a model training.
- **src/Utils/Data_Exploration_and_cleaning.ipynb**: Contains the code to clean the synthetic summaries generated and the data exploration.
- **src/*.py**: contains the different training strategies that we tried: Using LoRa, standard fine-tuning or two step fine-tuneing. There is an explanation at the start of each file.
- **src/Summaries_FLOR-760M**: Contains the test set sumaries generated and evaluated with the different strategies.
- **src/Synthetic_summary_creation**: Contains the files used to generate synthetic summaries with different LLM and languages.
- **src/Trained_models**: Path where are expected to be saved the different trained models weights.

# Fine-tuned model weights

The model weights are available [here](https://uab-my.sharepoint.com/:u:/g/personal/1523726_uab_cat/ES-1e_5T1dxCvAq7LfDwbVUBDqJmml5zcaer5QZaxtr1wQ?e=y5tK0r).

The names and training procedure of each of the models are the following:
- *last_model_LoRA_Finetune_MedicalReports_v2*                    
    - pre-fine-tuned: standard,  fine-tuned: lora  -> Word embeddings freeze in the prefine-tuning 
- *last_model_LoRA_Finetune_MedicalReports_v2_LoRA_pretrained*   
    - pre-fine-tuned: lora,      fine-tuned: lora 
- *last_model_NO_LoRA_Finetune_MedicalReports_v2*                 
    - pre-fine-tuned: standard,  fine-tuned: standard -> Word embeddings freeze in the prefine-tuning 
- *last_model_LoRA_MedicalReports_new_data*                       
    - pre-fine-tuned: None,      fine-tuned: lora 
- *last_model_train_Medical_newdata_without_LoRA*                 
    - pre-fine-tuned: None,      fine-tuned: standard

*pre-fine-tuned means fine-tuned first on CaBreu dataset. 

We make use of this names for the different folders and filenames during the models evaluation.

# Setup

Create the env with the necessary libraries
```
conda env create --file environment.yml --name 'YOUR ENV NAME'
```
