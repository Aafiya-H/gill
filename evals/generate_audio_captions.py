from gill import models
from gill import utils
import torch
import librosa

model_dir = 'runs/frozen_1/'
model = models.load_gill(model_dir,"ckpt.pth.tar")
model = model.float()
print("Loaded model")

audio_data, sampling_rate = librosa.load("datasets/AudioCaps/test/845.wav")
target_sampling_rate = 48000 
audio_data = librosa.resample(audio_data, orig_sr=sampling_rate, target_sr=target_sampling_rate)
sampling_rate = target_sampling_rate

prompts = [
    {"audio_data":audio_data,"sampling_rate":sampling_rate},
    "The sound of"
]

return_outputs = model.generate_for_images_and_texts(prompts, num_words=16, min_word_tokens=16)
