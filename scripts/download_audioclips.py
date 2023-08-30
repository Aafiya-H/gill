from audiocaps_download import Downloader
from tqdm import tqdm
import pandas as pd
import os

split = "test"
# print(os.getcwd())
# os.system("ls")
d = Downloader(root_path='datasets/AudioCaps', n_jobs=10)
df = pd.read_csv(f"datasets/AudioCaps/{split}.csv")

root_path = f"datasets/AudioCaps/{split}"

for index,row in tqdm(df.iterrows(),total=len(df)):
    target_file_path = os.path.join(root_path,str(row["audiocap_id"]) + ".wav")
    start_seconds = row["start_time"]
    yt_id = row["youtube_id"]
    os.system(f'yt-dlp -x --audio-format wav --audio-quality 5 --output "{target_file_path}" --postprocessor-args "-ss {start_seconds} -to {start_seconds+10}" https://www.youtube.com/watch?v={yt_id}')
