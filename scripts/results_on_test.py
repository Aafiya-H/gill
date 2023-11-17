from gill import models
from gill import utils
from tqdm import tqdm

import pandas as pd
import librosa
import argparse
import os
import json

split = "test"
base_path = "../../../../../mnt/media/wiseyak/reasoning-datasets/AudioCaps"
base_test_dir = os.path.join(base_path,f"{split}")
audiocaps_df =pd.read_csv(os.path.join(base_path,f"{split}-10s-downloaded.csv")) 

def main(args):
    result = {}
    model = models.load_gill(args.model_dir,args.checkpoint_name)
    exp_name = os.path.basename(os.path.dirname(os.path.join(args.model_dir,args.checkpoint_name)))
    print("Exp name: ",exp_name)
    print("Model loaded !!")
    
    for audiofile in tqdm(audiocaps_df["audiocap_id"],total=len(audiocaps_df["audiocap_id"])):
        audiofile = str(audiofile) + ".wav"
        audio_data, sr = librosa.load(os.path.join(base_test_dir,audiofile))
        target_sampling_rate = 48000 
        audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=target_sampling_rate)
        audio = {"audio_data":audio_data,"sampling_rate":target_sampling_rate}
        prompts = [
        audio,
        'This is the sound of'
        ]
        return_outputs = model.generate_for_images_and_texts(prompts, num_words=16, min_word_tokens=16)
        generated_caption = 'This is the sound of ' + return_outputs[0]
        result[audiofile[:-4]] = generated_caption
    
    ckpt_name = args.checkpoint_name[:-len(".pth.tar")]
    result_file_path = os.path.join(base_path,f"results/{exp_name}-{ckpt_name}.json")
    with open(result_file_path,"w") as f:
        json.dump(result,f)
        
    print("Stored result!!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-name")
    parser.add_argument("--model-dir")
    args = parser.parse_args()
    main(args)
