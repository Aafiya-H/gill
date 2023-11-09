from transformers import ClapModel, AutoTokenizer, AutoFeatureExtractor, AutoProcessor
from sklearn.metrics import average_precision_score
from gill import utils
from tqdm import tqdm

import pandas as pd
import os
import torch
import librosa
import argparse

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

def get_scores(captions, batch_size = 16):
    batches = {}
    for gt_index,caption in tqdm(enumerate(captions),total=len(captions)):        
        start_index = (gt_index//batch_size) * batch_size
        end_index = min(start_index + batch_size,len(audiocaps_df))
        inputs = contrastive_tokenizer([caption], padding=True, return_tensors="pt")
        text_features = contrastive_model.get_text_features(**inputs)
        
        audio_data_list = []
        audiofiles = audiocaps_df[start_index:end_index]["audiocap_id"].tolist()
        for audio_file_name in audiofiles:
            audio_data, sampling_rate = librosa.load(os.path.join(audio_files_path,audio_file_name+".wav"))
            target_sampling_rate = 48000 
            audio_data = librosa.resample(audio_data, orig_sr=sampling_rate, target_sr=target_sampling_rate)
            audio_data_list.append(audio_data)
        
        inputs = utils.get_audio_values_for_model(contrastive_feature_extractor,audio_data_list,target_sampling_rate)
        audio_features = contrastive_model.get_audio_features(**inputs)
        
        if f"{start_index}-{end_index}" not in batches:
            batches[f"{start_index}-{end_index}"] = {"R@1":0,"R@5":0,"R@10":0,"start":start_index,"end":end_index,"y_scores":[],"y_true":[]}
        
        similarity_scores = text_features @ torch.transpose(audio_features,0,1)
        similarity_scores = similarity_scores.squeeze(0)
        sorted_indices = torch.argsort(similarity_scores,descending=True)
        audio_index = start_index + sorted_indices[:10]
        
        if gt_index in audio_index[:1]:
            batches[f"{start_index}-{end_index}"]["R@1"] += 1
        if gt_index in audio_index[:5]:
            batches[f"{start_index}-{end_index}"]["R@5"] += 1
        if gt_index in audio_index[:10]:
            batches[f"{start_index}-{end_index}"]["R@10"] += 1
            
        batches[f"{start_index}-{end_index}"]["y_scores"].append(similarity_scores[sorted_indices].detach().numpy())
        batches[f"{start_index}-{end_index}"]["y_true"].append([1 if i == gt_index else 0 for i in audio_index])
        
        if gt_index % 200 == 0:
            print(start_index,batches)
    return batches

if __name__ == "__main__":
    parser = argparse.ArgumentParser() # 1000 works
    parser.add_argument("batch_size",type=int)
    args = parser.parse_args()
    args.batch_size = 1000 # to be removed
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
    
    row = {
        "R@1":r1, "R@5":r5, "R@10":r10,"MAP@10":map_scores,"SlidingWindow":args.batch_size,"Direction":"T->A"   
    }
    
    results_df = pd.read_csv("encoders-retrieval-scripts/results.csv")
    results_df = pd.concat(results_df,row,axis=0)
    print("-"*20,"Results","-"*20)
    print(row)
    results_df.to_csv("encoders-retrieval-scripts/results.csv")
