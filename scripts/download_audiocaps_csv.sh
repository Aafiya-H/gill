#!/bin/bash
# URL of the CSV file on GitHub
split="$1" # ["train","val","test"]
url="https://raw.githubusercontent.com/cdjkim/audiocaps/master/dataset/$split.csv"

# Local file path to save the downloaded file
local_file_path="../datasets/AudioCaps/$split.csv"

# Use wget to download the CSV file
wget -O "$local_file_path" "$url"

# Check if the download was successful
if [ $? -eq 0 ]; then
   echo "CSV file downloaded and saved as $local_file_path"
else
   echo "Failed to download CSV file"
fi
