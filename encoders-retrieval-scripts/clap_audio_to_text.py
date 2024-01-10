from transformers import ClapModel, AutoTokenizer, AutoFeatureExtractor, AutoProcessor
from sklearn.metrics import average_precision_score
from gill import utils
from tqdm import tqdm

import pandas as pd
import numpy as np
import os
import torch
import librosa
import argparse
import json

split = "test"
base_path = "datasets/AudioCaps"
audio_files_path = os.path.join(base_path,f"{split}")
df = pd.read_csv(os.path.join(base_path,f"{split}.csv")) 
# captions = audiocaps_df["caption"].tolist()

contrastive_model_name = "laion/clap-htsat-fused"
contrastive_model = ClapModel.from_pretrained(contrastive_model_name).eval()
contrastive_tokenizer = AutoTokenizer.from_pretrained(contrastive_model_name)
contrastive_feature_extractor = AutoFeatureExtractor.from_pretrained(contrastive_model_name)
contrastive_processor = AutoProcessor.from_pretrained(contrastive_model_name)

def get_scores(df):
    all_captions = df["caption_1"].tolist() + df["caption_2"].tolist() + df["caption_3"].tolist() + df["caption_4"].tolist() + df["caption_5"].tolist()
    scores = {"R@5":[],"R@10":[],"AP@10":[]}

    audio_file_names = df["file_name"]
    for index,audio_file_name in tqdm(enumerate(audio_file_names),total=len(audio_file_names)):
        gt_captions =  [df.iloc[index][f"caption_{i}"] for i in range(1,6)]

        audio_data, sampling_rate = librosa.load(os.path.join(base_path,split,audio_file_name),sr=48000)
        # target_sampling_rate = 48000 
        # audio_data = librosa.resample(audio_data, orig_sr=sampling_rate, target_sr=target_sampling_rate)
        inputs = utils.get_audio_values_for_model(contrastive_feature_extractor,audio_data,sampling_rate)
        audio_features = contrastive_model.get_audio_features(**inputs)

        text_sets = [all_captions[0:957],all_captions[957:2*957],all_captions[2*957:3*957],all_captions[3*957:4*957],all_captions[4*957:]]
        index_score_mapping = {}
        for i,text_set in enumerate(text_sets[1:]):
            i += 1
            inputs = contrastive_tokenizer(text_set, padding=True, return_tensors="pt")
            text_features = contrastive_model.get_text_features(**inputs)

            similarity_scores = audio_features @ torch.transpose(text_features,0,1)
            similarity_scores = similarity_scores.squeeze(0)
            sorted_indices = torch.argsort(similarity_scores,descending=True)[:10]

            new_dict = {i*957 + sorted_index.item():similarity_scores[sorted_index.item()].item() for sorted_index in sorted_indices}
            index_score_mapping.update(new_dict)

        similarity_scores = list(index_score_mapping.values())
        similarity_scores.sort(reverse=True)
        similarity_scores = similarity_scores[:10]
        sorted_indices = []
        for similarity_score in similarity_scores:
            # retrieve indices from similarity_score
            for k,v in index_score_mapping.items():
                if v == similarity_score:
                    sorted_indices.append(k)

        if len(sorted_indices) > 10:
            sorted_indices = sorted_indices[:10]

        retrieved_captions = [all_captions[i] for i in sorted_indices]
        relevance = [1 if pred in gt_captions else 0 for pred in retrieved_captions]
        precision = [sum(relevance[:j+1])/(j+1) for j in range(10)]
        recall = [sum(relevance[:j+1])/len(gt_captions) for j in range(10)]
        scores["R@5"].append(recall[-5])
        scores["R@10"].append(recall[-1])
        delta_recall = []

        for i in range(10):
            if i == 0:
                diff_recall = recall[i]
            else:
                diff_recall = recall[i] - recall[i-1]
            delta_recall.append(diff_recall)
        average_precision =  sum([precision[j] * delta_recall[j] for j in range(10)])
        scores["AP@10"].append(average_precision)

    return scores

if __name__ == "__main__":

    scores = get_scores(df)
    print(scores)
    r5 = sum(scores["R@5"])/len(df)
    r10 = sum(scores["R@10"])/len(df)
    map10 = sum(scores["AP@10"])/ len(df)
    
    row = pd.DataFrame.from_dict({
        "Model":[contrastive_model_name + "(laion implementation)"], "R@5":[r5], "R@10":[r10],"MAP@10":[map10],"Direction":["A->T"]   
    })
    if os.path.exists("encoders-retrieval-scripts/results.csv"):
        results_df = pd.read_csv("encoders-retrieval-scripts/results.csv")
    else:
        results_df = pd.DataFrame.from_dict({})
    results_df = pd.concat([results_df,row],axis=0)
    print("-"*20,"Results","-"*20)
    print(row)
    results_df.to_csv("encoders-retrieval-scripts/results.csv",index=False)
    # print("Length of dataset:",len(audiocaps_df))

