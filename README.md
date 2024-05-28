# Introduction

This project presents a comparison of different fine-tuning techniques for text summarization and an analysis of the currently available pretrained models in minority languages, such as Catalan. We propose an augmentation technique based on guidance from large language models (LLMs) that helps in domain adaptation from general knowledge learned during pretraining to specific smaller domains, such as those found in medical reports. By leveraging the capabilities of LLAMA 70B, we create a rich synthetic dataset from unannotated medical reports, which is used to guide the fine-tuning of smaller LLMs. This pipeline provides a systematic framework for fine-tuning smaller models in data-scarce domains.


# Repository Strucutre

- src/Datasets: Contains the train/val and test sets
- src/utils/create_examples_and_eval_*.py: contains the templates and code used for generating the synthetic dataset both catalan and spanish summaries.
- src/*.py: contains the different training strategies that we tried: Using LoRa or standard fine-tuning.
- src/Summaries_FLOR-760M : Contains the test set evaluation with the different strategies tried.

# Fine-tuned model weights

The model weights are available [here](link.com).

The names and training procedure of each of the models are the following:
- last_model_LoRA_Finetune_MedicalReports_v2                    
    - pre-fine-tuned: standard,  fine-tuned: lora  -> Word embeddings freeze in the prefine-tuning 
- last_model_LoRA_Finetune_MedicalReports_v2_LoRA_pretrained    
    - pre-fine-tuned: lora,      fine-tuned: lora 
- last_model_NO_LoRA_Finetune_MedicalReports_v2                 
    - pre-fine-tuned: standard,  fine-tuned: standard -> Word embeddings freeze in the prefine-tuning 
- last_model_LoRA_MedicalReports_new_data                       
    - pre-fine-tuned: None,      fine-tuned: lora 
- last_model_train_Medical_newdata_without_LoRA                 
    - pre-fine-tuned: None,      fine-tuned: standard

\*pre-fine-tuned means fine-tuned first on CaBreu dataset. 

We make use of this names for the diefferen folders and filenames during the models evaluation.

# Setup

Create the env with the necessary libraries
```
conda env create --file environment.yml --name 'YOUR ENV NAME'
```

