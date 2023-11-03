from transformers import ClapModel, AutoTokenizer, AutoFeatureExtractor, AutoProcessor
from gill import utils
from tqdm import tqdm

import pandas as pd
import os
import torch
import librosa

split = "test"
base_path = "../../../../mnt/media/wiseyak/reasoning-datasets/AudioCaps"
audio_files_path = os.path.join(base_path,f"{split}")
audiocaps_df =pd.read_csv(os.path.join(base_path,f"{split}-10s-downloaded.csv")) 
captions = audiocaps_df["caption"].tolist()

contrastive_model_name = "laion/clap-htsat-fused"
contrastive_model = ClapModel.from_pretrained(contrastive_model_name)
contrastive_tokenizer = AutoTokenizer.from_pretrained(contrastive_model_name)
contrastive_feature_extractor = AutoFeatureExtractor.from_pretrained(contrastive_model_name)
contrastive_processor = AutoProcessor.from_pretrained(contrastive_model_name)

def contrastive_dot_product(audio_file_names, batch_size = 16):
    count = 0
    for index,audio_file_name in tqdm(enumerate(audio_file_names),total=len(audio_file_names)):
        audio_data, sampling_rate = librosa.load(os.path.join(audio_files_path,audio_file_name+".wav"))
        target_sampling_rate = 48000 
        audio_data = librosa.resample(audio_data, orig_sr=sampling_rate, target_sr=target_sampling_rate)
        
        inputs = utils.get_audio_values_for_model(contrastive_feature_extractor,audio_data,target_sampling_rate)
        audio_features = contrastive_model.get_audio_features(**inputs)
        
        start_index = (index//batch_size) * batch_size
        end_index = min(start_index + batch_size,len(audiocaps_df))
        # print(start_index,end_index)
        inputs = contrastive_tokenizer(captions[start_index:end_index], padding=True, return_tensors="pt")
        text_features = contrastive_model.get_text_features(**inputs)
        
        probability_distribution = audio_features @ torch.transpose(text_features,0,1)
        caption_index = start_index + torch.argmax(probability_distribution)
        
        gt_caption_index = audiocaps_df[audiocaps_df["audiocap_id"] == audio_file_name].index
        if caption_index.item() == gt_caption_index.item():
            count += 1
        if index % 200 == 0:
            print(start_index,count)
    return count

if __name__ == "__main__":
    audiocaps_df["audiocap_id"] = audiocaps_df["audiocap_id"].astype('str')
    # random_seed = 42
    # audiocaps_df = audiocaps_df.sample(frac=1, random_state=random_seed)
    # audiocaps_df.reset_index(inplace=True,drop=True)
    
    count = contrastive_dot_product(audiocaps_df["audiocap_id"])
    print("Count:",count)
    print("Length of dataset:",len(audiocaps_df))

