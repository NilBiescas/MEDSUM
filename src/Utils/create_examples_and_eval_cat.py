# Make visible the GPU
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BloomForCausalLM
from peft import AutoPeftModelForCausalLM, PeftModelForSeq2SeqLM
import pandas as pd
from datasets import load_dataset, Dataset
from tqdm import tqdm
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import spacy
import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer

model_sentence = SentenceTransformer('hiiamsid/sentence_similarity_spanish_es')

model_id = "/hhome/nlp2_g05/Asho_NLP/src/Trained_models/last_model_train_Medical_newdata_without_LoRA"
# last_model_LoRA_Finetune_MedicalReports_v2                    -> prefintuning: standard,  finetune: lora  -> Word embeddings freeze in the prefine-tuning 
# last_model_LoRA_Finetune_MedicalReports_v2_LoRA_pretrained    -> prefintuning: lora,      finetune: lora 
# last_model_NO_LoRA_Finetune_MedicalReports_v2                 -> prefintuning: standard,  finetune: standard -> Word embeddings freeze in the prefine-tuning 

# last_model_LoRA_MedicalReports_new_data                       -> prefintuning: None,      finetune: lora 
# last_model_train_Medical_newdata_without_LoRA                 -> prefintuning: None,      finetune: standard

DATA2COMPARE = "gt" # gt or synthetic
COMPUTE_METRICS = True

word2vec = spacy.load('es_core_news_lg')

def compute_metrics(gt: str, pred: str, i: int):
    # Tokenize the sentences
    gt_sentences = nltk.sent_tokenize(gt)
    pred_sentences = nltk.sent_tokenize(pred)
    
    # Join the sentences with a new line
    gt_sentences_joined = "\n".join(gt_sentences)
    pred_sentences_joined = "\n".join(pred_sentences)
    
    # Compute bleu and rouge scores
    rouge_score = rouge.compute(predictions=[pred_sentences_joined], references=[gt_sentences_joined])
    bleu_score = bleu.compute(predictions=[pred_sentences_joined], references=[[gt_sentences_joined]])
    
    # Remove stopwords
    gt_sentences_no_stopwords = " ".join([word.lower() for word in nltk.word_tokenize(gt_sentences_joined) if word.lower() not in stopwords2use])
    pred_sentences_no_stopwords = " ".join([word.lower() for word in nltk.word_tokenize(pred_sentences_joined) if word.lower() not in stopwords2use])

    # Compute bleu and rouge scores without stopwords
    rouge_score_no_stopwords = rouge.compute(predictions=[pred_sentences_no_stopwords], references=[gt_sentences_no_stopwords])
    bleu_score_no_stopwords = bleu.compute(predictions=[pred_sentences_no_stopwords], references=[[gt_sentences_no_stopwords]])
    
    
    # Compute cosine similarity of the tf-idf of the sentences
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([gt_sentences_joined, pred_sentences_joined])
    cos_similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

    # Compute cosine similarity of the tf-idf of the sentences without stopwords
    tfidf_matrix_no_stopwords = vectorizer.fit_transform([gt_sentences_no_stopwords, pred_sentences_no_stopwords])
    cosine_similarity_no_stopwords = cosine_similarity(tfidf_matrix_no_stopwords[0:1], tfidf_matrix_no_stopwords[1:2])[0][0]


    # Compute word2vec similarity with and without stopwords
    ground_truth = word2vec(gt_sentences_joined)
    pred_sentence = word2vec(pred_sentences_joined)
    
    ground_truth_no_stopwords = word2vec(gt_sentences_no_stopwords)
    pred_sentence_no_stopwords = word2vec(pred_sentences_no_stopwords)

    similarity = ground_truth.similarity(pred_sentence)
    similarity_no_stopwords = ground_truth_no_stopwords.similarity(pred_sentence_no_stopwords)
    
    # Tokenize by words
    gt_tokens = nltk.word_tokenize(gt_sentences_joined)
    pred_tokens = nltk.word_tokenize(pred_sentences_joined)
    
    # Computing the word histogram of the sentences
    unique_tokens_gt = set(gt_tokens)
    unique_tokens_pred = set(pred_tokens)
    all_unique_words = unique_tokens_pred.union(unique_tokens_gt)

    hist_gt = np.array([gt_sentences_joined.lower().count(word) for word in all_unique_words])
    hist_pred = np.array([pred_sentences_joined.lower().count(word) for word in all_unique_words])

    # Compute the intersection of histograms
    intersection = np.minimum(hist_gt, hist_pred)
    intersection_sum = np.sum(intersection)
    length_summ = np.sum(hist_gt)
    length_predicted = np.sum(hist_pred)
    hist_similarity = intersection_sum / (length_summ + length_predicted - intersection_sum)
    
    # Save the histograms
    plt.figure(figsize=(20, 10))
    plt.subplot(3, 1, 1)
    plt.bar(range(len(hist_gt)), hist_gt)
    plt.title("Ground truth histogram")
    plt.subplot(3, 1, 2)
    plt.bar(range(len(hist_pred)), hist_pred)
    plt.title("Predicted histogram")
    plt.subplot(3, 1, 3)
    plt.bar(range(len(intersection)), intersection)
    plt.title("Intersection histogram")
    os.makedirs(f"/hhome/nlp2_g05/Asho_NLP/src/Plots/{model_id.split('/')[-1]}_cat/freq_histograms", exist_ok=True)
    plt.savefig(f"/hhome/nlp2_g05/Asho_NLP/src/Plots/{model_id.split('/')[-1]}_cat/freq_histograms/histograms_{i}.png")
    
    # Compute cosine similarity of the sentence embeddings
    embeddings = model_sentence.encode([gt, pred])
    cos_similarity_sentence = cosine_similarity(embeddings[0].reshape(1, -1), embeddings[1].reshape(1, -1))[0][0]
    
    # Return the metrics
    metrics = {
        "rouge": rouge_score,
        "bleu": bleu_score,
        "rouge_no_stopwords": rouge_score_no_stopwords,
        "bleu_no_stopwords": bleu_score_no_stopwords,
        "cosine_similarity": cos_similarity,
        "cosine_similarity_no_stopwords": cosine_similarity_no_stopwords,
        "word2vec_similarity": similarity,
        "word2vec_similarity_no_stopwords": similarity_no_stopwords,
        "histogram_intersection": hist_similarity,
        "cos_similarity_sentence_embedder": str(cos_similarity_sentence)
    }
    return metrics

if COMPUTE_METRICS:
    # If we need to compute metrics load the necessary libraries and functions
    import evaluate
    import nltk
    
    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")
    
    stopwords = nltk.corpus.stopwords
    cat_stopwords = set(stopwords.words("catalan"))
    spa_stopwords = set(stopwords.words("spanish"))
    stopwords2use = cat_stopwords.union(spa_stopwords)

if DATA2COMPARE == "synthetic":
    # Read json sumaries generated with chatgpt API
    df = pd.read_json('/hhome/nlp2_g05/Asho_NLP/src/chatgpt_summaries_gpt4.json')
    # Changing the columns by the rows
    df = df.T

    # Read json sumaries generated with llama3
    df_llama = pd.read_json('/hhome/nlp2_g05/Asho_NLP/src/summaries_llama3_v2.json')
    df_llama = df_llama.T

    # Reading the first two examples from the Asho dataset, which where used to generate the summaries
    asho_train_gt = pd.read_json("/hhome/nlp2_g05/Asho_NLP/src/Datasets/Asho_Dataset/train_cleaned.json")
    example_text_0 = asho_train_gt["Text"][0]
    example_text_1 = asho_train_gt["Text"][1]
    example_summary_0 = asho_train_gt["Summary"][0]
    example_summary_1 = asho_train_gt["Summary"][1]

    # Add the two examples to the dataframe
    new_row = {"Text": example_text_0, "summary": example_summary_0}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    new_row = {"Text": example_text_1, "summary": example_summary_1}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # Join ChatGPT and llama3 summaries
    df = pd.concat([df, df_llama], ignore_index=True)

    # Convert the Pandas dataFrame to a Hugging Face Dataset
    dataset = Dataset.from_pandas(df.astype(str))

    splits = dataset.train_test_split(test_size=0.05, seed=42)
    dataset_eval = splits["test"]
    
elif DATA2COMPARE == "gt":
    asho_train_gt = pd.read_json("/hhome/nlp2_g05/Asho_NLP/src/Datasets/Asho_Dataset/train_cleaned.json")
    
    # Remove the 5 and 6 line, which where used to generate the summaries
    asho_train_gt = asho_train_gt.drop([5, 6])

    dataset_eval = Dataset.from_pandas(asho_train_gt.astype(str))
    
else:
    raise ValueError("data2compare must be 'gt' or 'synthetic'")


# Load the trained model
tokenizer = AutoTokenizer.from_pretrained(model_id)
try:
    # In case is fine-tune with LoRA
    model = PeftModelForSeq2SeqLM.from_pretrained(BloomForCausalLM.from_pretrained(model_id), model_id).to("cuda:0")
except Exception as e:
    # Regular fine-tune
    print("\n\n Loading model: ", model_id, flush=True, end = "\n\n")
    model = BloomForCausalLM.from_pretrained(model_id).to("cuda:0")
    
print("Model loaded")
print("Device:", model.device)

# Create the path of output file
output_file = f"/hhome/nlp2_g05/Asho_NLP/src/gen_sumaries_all_metrics_{model_id.split('/')[-1]}_cat.json"
output_data = []

metriques = {"cos_similarity_sentence_embedder": [], "rouge_1": [], "bleu_1": [],"rouge_no_stopwords": [], "bleu_no_stopwords": [], "cosine_similarity": [], "cosine_similarity_no_stopwords": [], "word2vec_similarity": [], "word2vec_similarity_no_stopwords": [], "histogram_intersection": []}

for i in tqdm(range(len(dataset_eval))):
    # Prepare the input text, it is done in to parts in 
    # case the text is too long for the model the keyword 
    # to resume is added and not truncated
    text = dataset_eval["Text"][i]
    input_text = "Resumeix:" + text 
    input = tokenizer(input_text, return_tensors="pt", max_length=3500, truncation=True)
    keyword = "\nResum:\n"
    keyword = tokenizer(keyword, return_tensors="pt", max_length=3500, truncation=True)
    
    input["input_ids"] = torch.cat((input["input_ids"], keyword["input_ids"]), dim=1)
    input["attention_mask"] = torch.cat((input["attention_mask"], keyword["attention_mask"]), dim=1)
    
    # Generate the summary
    generation = model.generate(
        input_ids=input["input_ids"].to(model.device),
        attention_mask=input["attention_mask"].to(model.device),
        eos_token_id=tokenizer.eos_token_id,
        min_length=15,
        max_new_tokens=200,
        num_beams=6,
        repetition_penalty=2.5,
        length_penalty=1.0,
        early_stopping=True,
        no_repeat_ngram_size=2,
        use_cache=True,
        do_sample=True,
        temperature=0.05,
        top_k=50,
    )
    text_generated = tokenizer.decode(generation[0], skip_special_tokens=True)
    # Remove the input text from the generated text
    input_text_2 = tokenizer.decode(input["input_ids"][0], skip_special_tokens=True)
    text_generated = text_generated.replace(input_text_2, "")

    if not COMPUTE_METRICS:
        output_data.append({
            "Summary": i,
            "Input": input_text,
            "Result": text_generated
        })
    else:
        metrics = compute_metrics(dataset_eval["Summary"][i], text_generated, i)
        
        output_data.append({
            "Summary": i,
            "Input": input_text,
            "Result": text_generated,
            "gt": dataset_eval["Summary"][i],
            "metrics": metrics
        })
        
        # Saving the metrics in a list to make plots later
        metriques["cosine_similarity"].append(metrics["cosine_similarity"])
        metriques["cosine_similarity_no_stopwords"].append(metrics["cosine_similarity_no_stopwords"])
        metriques["word2vec_similarity"].append(metrics["word2vec_similarity"])
        metriques["word2vec_similarity_no_stopwords"].append(metrics["word2vec_similarity_no_stopwords"])
        metriques["histogram_intersection"].append(metrics["histogram_intersection"])
        metriques['rouge_1'].append(metrics["rouge"]['rouge1'])
        metriques['bleu_1'].append(metrics["bleu"]['precisions'][0])
        metriques["bleu_no_stopwords"].append(metrics["bleu_no_stopwords"]['precisions'][0])
        metriques["rouge_no_stopwords"].append(metrics["rouge_no_stopwords"]['rouge1'])
        metriques["cos_similarity_sentence_embedder"].append(metrics["cos_similarity_sentence_embedder"])
        
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=4)
        

# Making a bar plot of the metrics for each of the 8 test summaries
os.makedirs(f"/hhome/nlp2_g05/Asho_NLP/src/Plots/{model_id.split('/')[-1]}_cat", exist_ok=True)
for i, key in enumerate(metriques.keys()):
    fig = plt.figure()
    plt.bar(range(len(metriques[key])), np.array(metriques[key]).astype(float))
    plt.title(key)
    plt.savefig(f"/hhome/nlp2_g05/Asho_NLP/src/Plots/{model_id.split('/')[-1]}_cat/{key}_cat.png")
    
# Compute the mean of the metrics
results = {}    
for key, value in metriques.items():
    results[key] = round(np.mean(np.array(value).astype(float)), 2)
    
# Save it with output_data json
output_data.append({'results': results})
with open(output_file, "w") as f:
    json.dump(output_data, f, indent=4)