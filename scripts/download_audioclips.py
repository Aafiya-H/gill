# from audiocaps_download import Downloader
from tqdm import tqdm
import pandas as pd
import os
import argparse
from moviepy.editor import AudioFileClip

def get_audio_duration(file_path):
    audio_clip = AudioFileClip(file_path)
    duration = audio_clip.duration
    audio_clip.close()
    return duration

def download_audiocaps(split):
    print("Split :",split)
    df = pd.read_csv(f"datasets/AudioCaps/{split}.csv")

    root_path = f"datasets/AudioCaps/{split}"
    if not os.path.exists(root_path):
        os.mkdir(root_path)
        print("Path")

    for index, row in tqdm(df.iterrows(), total=len(df)):
        target_file_path = os.path.join(root_path, str(row["youtube_id"]) + ".wav")
        if os.path.exists(target_file_path):
            try:
                duration = get_audio_duration(target_file_path)
                if duration == 10:
                    continue
                else:
                    NotImplementedError
            except:
                os.remove(target_file_path)
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
