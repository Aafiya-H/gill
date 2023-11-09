from transformers import ClapModel, AutoTokenizer, AutoFeatureExtractor, AutoProcessor
from sklearn.metrics import average_precision_score
from gill import utils
from tqdm import tqdm

import pandas as pd
import os
import torch
import librosa
import argparse
import json

split = "test"
base_path = "../../../../../mnt/media/wiseyak/reasoning-datasets/AudioCaps"
audio_files_path = os.path.join(base_path,f"{split}")
audiocaps_df =pd.read_csv(os.path.join(base_path,f"{split}-10s-downloaded.csv")) 
captions = audiocaps_df["caption"].tolist()

contrastive_model_name = "laion/clap-htsat-fused"
contrastive_model = ClapModel.from_pretrained(contrastive_model_name)
contrastive_tokenizer = AutoTokenizer.from_pretrained(contrastive_model_name)
contrastive_feature_extractor = AutoFeatureExtractor.from_pretrained(contrastive_model_name)
contrastive_processor = AutoProcessor.from_pretrained(contrastive_model_name)

def get_scores(audio_file_names, batch_size = 16):
    batches = json.load(open("encoders-retrieval-scripts/clap_audio_to_text.json"))
    print("Batch-size:",batch_size)
    
    if "last-index-completed" not in batches:
        batches["last-index-completed"] = -1
    elif batches["last-index-completed"] == len(audio_file_names) - 1:
        return batches
    print("Resuming from ",batches["last-index-completed"]+1)
    
    for index,audio_file_name in tqdm(enumerate(audio_file_names[batches["last-index-completed"] + 1:]),total=len(audio_file_names[batches["last-index-completed"] + 1:])):
        audio_data, sampling_rate = librosa.load(os.path.join(audio_files_path,audio_file_name+".wav"))
        target_sampling_rate = 48000 
        audio_data = librosa.resample(audio_data, orig_sr=sampling_rate, target_sr=target_sampling_rate)
        
        inputs = utils.get_audio_values_for_model(contrastive_feature_extractor,audio_data,target_sampling_rate)
        audio_features = contrastive_model.get_audio_features(**inputs)
        
        start_index = (index//batch_size) * batch_size
        end_index = min(start_index + batch_size,len(audiocaps_df))
        
        if f"{start_index}-{end_index}" not in batches:
            batches[f"{start_index}-{end_index}"] = {"R@1":0,"R@5":0,"R@10":0,"start":start_index,"end":end_index,"y_scores":[],"y_true":[]}

        inputs = contrastive_tokenizer(captions[start_index:end_index], padding=True, return_tensors="pt")
        text_features = contrastive_model.get_text_features(**inputs)
        
        similarity_scores = audio_features @ torch.transpose(text_features,0,1)
        similarity_scores = similarity_scores.squeeze(0)
        sorted_indices = torch.argsort(similarity_scores,descending=True)
        caption_index = start_index + sorted_indices[:10]
        gt_caption_index = audiocaps_df[audiocaps_df["audiocap_id"] == audio_file_name].index
        
        if gt_caption_index.item() in caption_index[:1]:
            batches[f"{start_index}-{end_index}"]["R@1"] += 1
        if gt_caption_index.item() in caption_index[:5]:
            batches[f"{start_index}-{end_index}"]["R@5"] += 1
        if gt_caption_index.item() in caption_index[:10]:
            batches[f"{start_index}-{end_index}"]["R@10"] += 1
        
        batches[f"{start_index}-{end_index}"]["y_scores"].append(similarity_scores[sorted_indices].detach().tolist()[:10])
        batches[f"{start_index}-{end_index}"]["y_true"].append([1 if i == gt_caption_index.item() else 0 for i in caption_index])
        
        if index % 200 == 0:
            print("Rewrote log file!!")
            batches["last-index-completed"] = index
            with open("encoders-retrieval-scripts/clap_audio_to_text.json","+w") as f:
                json.dump(batches,f)
    del batches["last-index-completed"]
    return batches

if __name__ == "__main__":
    parser = argparse.ArgumentParser() # 1000 works
    parser.add_argument("batch_size",type=int)
    args = parser.parse_args()
    
    audiocaps_df["audiocap_id"] = audiocaps_df["audiocap_id"].astype('str')
    batches = get_scores(audiocaps_df["audiocap_id"],args.batch_size)
    r1, r5, r10, map_scores = 0,0,0,0
    
    for key,value in batches.items():
        r1 += value["R@1"]
        r5 += value["R@5"]
        r10 += value["R@10"]
        ap_scores = [
            average_precision_score(true, scores) for true, scores in zip(value["y_true"],value["y_scores"])
        ]
        map_score = sum(ap_scores) / len(ap_scores)
        map_scores += map_score
    
    r1 /= len(audiocaps_df)
    r5 /= len(audiocaps_df)
    r10 /= len(audiocaps_df)
    map_scores = map_scores/len(audiocaps_df)
    
    row = pd.DataFrame.from_dict({
        "R@1":[r1], "R@5":[r5], "R@10":[r10],"MAP@10":[map_scores],"SlidingWindow":[args.batch_size],"Direction":["A->T"]   
    })
    
    results_df = pd.read_csv("encoders-retrieval-scripts/results.csv")
    results_df = pd.concat([results_df,row],axis=0)
    print("-"*20,"Results","-"*20)
    print(row)
    results_df.to_csv("encoders-retrieval-scripts/results.csv")
    # print("Length of dataset:",len(audiocaps_df))

