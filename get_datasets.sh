#!/bin/bash
dir="image_embeddings"
google_dir="https://drive.google.com/drive/folders"

if [ ! -d "$dir" ]; then
  mkdir image_embeddings
  echo "Create $dir directory"
fi

cd $dir

gdown -O "cplfw" --folder "${google_dir}/1YjZ-z8V3eZu00lixAo_w9HNMyKV2U6B6"
gdown -O "ms1m" --folder "${google_dir}/1YdT9kcjNPaGyc3jXsrsT08S6ADv2lgQd"
gdown -O "megaface" --folder "${google_dir}"
#gdown --folder
#gdown --folder
