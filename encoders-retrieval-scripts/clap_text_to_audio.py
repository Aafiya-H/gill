from transformers import ClapModel, AutoTokenizer, AutoFeatureExtractor, AutoProcessor
from torch.utils.data import Dataset
from gill import utils
from tqdm import tqdm

import pandas as pd
import os
import torch
import librosa
import argparse
import gc
import subprocess
import json

split = "test"
base_path = "datasets/AudioCaps"
audio_files_path = os.path.join(base_path,f"{split}")
df = pd.read_csv(os.path.join(base_path,f"{split}.csv")) 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

contrastive_model_name = "laion/clap-htsat-fused"
contrastive_model = ClapModel.from_pretrained(contrastive_model_name)#.to(device=device)
contrastive_tokenizer = AutoTokenizer.from_pretrained(contrastive_model_name)
contrastive_feature_extractor = AutoFeatureExtractor.from_pretrained(contrastive_model_name)
contrastive_processor = AutoProcessor.from_pretrained(contrastive_model_name)

class AudioClips(Dataset):
    def __init__(self,df,audioclips_path):
        self.df = df
        self.audioclips_path = audioclips_path
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        audio_file_path = os.path.join(self.audioclips_path,row["file_name"])
        audio_data, sampling_rate = librosa.load(audio_file_path)
        target_sampling_rate = 48000 
        audio_data = librosa.resample(audio_data, orig_sr=sampling_rate, target_sr=target_sampling_rate)
        sampling_rate = target_sampling_rate
        inputs = utils.get_audio_values_for_model(contrastive_feature_extractor,audio_data,sampling_rate)
        # inputs["input_features"] = inputs["input_features"].to(device=device)
        # inputs["is_longer"] = inputs["is_longer"].to(device=device)
        audio_features = contrastive_model.get_audio_features(**inputs)
        return audio_features

def get_scores(df):
    scores = {"R@1":[],"R@5":[],"R@10":[],"AP@10":[]}
    all_captions = df["caption_1"].tolist() + df["caption_2"].tolist() + df["caption_3"].tolist() + df["caption_4"].tolist() + df["caption_5"].tolist()
    batch_size = 100
    audioclips_dataset = AudioClips(df,audio_files_path)
    loader = torch.utils.data.DataLoader(
        audioclips_dataset, batch_size = batch_size
        )
    interval = 100
    for gt_index,caption in tqdm(enumerate(all_captions),total=len(all_captions)):     
        
        # torch.cuda.empty_cache()
        inputs = contrastive_tokenizer([caption], padding=True, return_tensors="pt")
        # inputs["input_ids"] = inputs["input_ids"].to(device=device)
        # inputs["attention_mask"] = inputs["attention_mask"].to(device=device)

        text_features = contrastive_model.get_text_features(**inputs)
        # text_features = text_features.to(device=device)
        sim_scores = []
        for i,audio_features in enumerate(loader):
            # audio_features = audio_features.to(device=device)
            similarity_scores = text_features @ torch.transpose(audio_features,0,1).view(512,audio_features.shape[0])
            similarity_scores = similarity_scores.squeeze(0).tolist()
            sim_scores.extend(similarity_scores)
            gc.collect()
            # torch.cuda.empty_cache()
        
        sorted_indices = torch.argsort(torch.tensor(sim_scores),descending=True)
        sorted_indices = sorted_indices[:10].tolist()
        gt_filename = str(df.iloc[gt_index%len(df)]["file_name"])
        
        ## file_name from sorted_indices
        retrieved_audios = [df.iloc[index]["file_name"] for index in sorted_indices]       
        relevance = [1 if pred == gt_filename else 0 for pred in retrieved_audios]
        precision = [sum(relevance[:j+1])/(j+1) for j in range(10)]
        recall = [sum(relevance[:j+1]) for j in range(10)]
        scores["R@1"].append(recall[0])
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
        
        if gt_index % interval == 0:
            with open("intermediate_result.json","+w") as f:
                json.dump(scores,f)
                print("Intermediate results stored !!")

    return scores

if __name__ == "__main__":
    scores = get_scores(df)
    print(scores)
    r1 = sum(scores["R@1"])/len(df)
    r5 = sum(scores["R@5"])/len(df)
    r10 = sum(scores["R@10"])/len(df)
    map10 = sum(scores["AP@10"])/ len(df)

    row = pd.DataFrame.from_dict({
        "Model":[contrastive_model_name], "R@1":[r1],"R@5":[r5], "R@10":[r10],"MAP@10":[map10],"Direction":["T->A"]   
    })

    if os.path.exists("encoders-retrieval-scripts/results.csv"):
        results_df = pd.read_csv("encoders-retrieval-scripts/results.csv")
    else:
        results_df = pd.DataFrame.from_dict({})
    results_df = pd.concat([results_df,row],axis=0)
    print("-"*20,"Results","-"*20)
    print(row)
    results_df.to_csv("encoders-retrieval-scripts/results.csv")