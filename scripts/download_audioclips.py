from audiocaps_download import Downloader
from tqdm import tqdm
import pandas as pd
import os
import argparse

def download_audiocaps(split):
    df = pd.read_csv(f"datasets/AudioCaps/{split}.csv")

    root_path = f"datasets/AudioCaps/{split}"
    if not os.path.exists(root_path):
        os.mkdir(root_path)

    for index, row in tqdm(df.iterrows(), total=len(df)):
        target_file_path = os.path.join(root_path, str(row["audiocap_id"]) + ".wav")
        start_seconds = row["start_time"]
        yt_id = row["youtube_id"]
        os.system(f'yt-dlp -x --audio-format wav --audio-quality 5 --output "{target_file_path}" --postprocessor-args "-ss {start_seconds} -to {start_seconds+10}" https://www.youtube.com/watch?v={yt_id}')

def main():
    parser = argparse.ArgumentParser(description="Download AudioCaps data for a specific split.")
    parser.add_argument("--split", type=str, help="Specify the split to download (e.g., 'train', 'val', 'test').")

    args = parser.parse_args()
    split = args.split

    download_audiocaps(split)

if __name__ == "__main__":
    main()
