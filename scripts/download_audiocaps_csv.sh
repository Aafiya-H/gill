#!/bin/bash
# URL of the CSV file on GitHub
url="https://raw.githubusercontent.com/cdjkim/audiocaps/master/dataset/train.csv"

# Local file path to save the downloaded file
local_file_path="../datasets/AudioCaps/train.csv"

# Use wget to download the CSV file
wget -O "$local_file_path" "$url"

# Check if the download was successful
if [ $? -eq 0 ]; then
   echo "CSV file downloaded and saved as $local_file_path"
else
   echo "Failed to download CSV file"
fi
