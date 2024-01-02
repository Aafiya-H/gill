from moviepy.editor import AudioFileClip
from tqdm import tqdm
import os

def get_audio_duration(file_path):
    audio_clip = AudioFileClip(file_path)
    duration = audio_clip.duration
    audio_clip.close()
    return duration
    
split = "test"
root_path = f"datasets/AudioCaps/{split}"
total_files = len(os.listdir(root_path))
erronous_files = 0
total_10s_files = 0 
dur = {}

for f in tqdm(os.listdir(root_path),total=len(os.listdir(root_path))):
    target_file_path = os.path.join(root_path,f)
    try:
        duration = get_audio_duration(target_file_path)
        if duration in dur:
            dur[duration] += 1
        else:
            dur[duration] = 1
        if duration == 10:
            total_10s_files += 1
    except:
        erronous_files += 1

print("Total files :",total_files)
print("Files where duration = 10s :",total_10s_files)
print("Erronous files :",erronous_files)
print("Durations :",dur)
