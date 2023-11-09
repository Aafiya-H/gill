import pandas as pd
import os
import wav2clip

split = "test"
base_path = "../../../../mnt/media/wiseyak/reasoning-datasets/AudioCaps"
audio_files_path = os.path.join(base_path,f"{split}")
audiocaps_df =pd.read_csv(os.path.join(base_path,f"{split}-10s-downloaded.csv")) 
captions = audiocaps_df["caption"].tolist()

model = wav2clip.get_model()
embeddings = wav2clip.embed_audio(audio, model)