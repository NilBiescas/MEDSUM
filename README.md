# Introduction

This project presents a comparison of different fine-tuning techniques for text summarization and an analysis of the currently available pretrained models in minority languages, such as Catalan. We propose an augmentation technique based on guidance from large language models (LLMs) that helps in domain adaptation from general knowledge learned during pretraining to specific smaller domains, such as those found in medical reports. By leveraging the capabilities of LLAMA 70B, we create a rich synthetic dataset from unannotated medical reports, which is used to guide the fine-tuning of smaller LLMs. This pipeline provides a systematic framework for fine-tuning smaller models in data-scarce domains.


# Repository Strucutre

- src/Datasets: Contains the train/val and test sets
- src/utils/create_examples_and_eval_*.py: contains the templates and code used for generating the synthetic dataset both catalan and spanish summaries.
- src/*.py: contains the different training strategies that we tried: Using LoRa or standard fine-tuning.
- src/Summaries_FLOR-760M : Contains the test set evaluation with the different strategies tried.

# Setup

Create the env with the necessary libraries
```
conda env create --file environment.yml --name 'YOUR ENV NAME'
```

