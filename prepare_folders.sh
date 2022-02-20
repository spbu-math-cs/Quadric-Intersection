#!/bin/bash
dir="image_embeddings"
if [ ! -d "$dir" ]; then
  tar -xvzf image_embeddings.tar.gz
fi
models_dir=$(jq -r '.models_dir' config.json)
features_dir=$(jq -r '.models_dir' config.json)
mkdir $features_dir $models_dir
mkdir $features_dir/calfw $features_dir/cplfw $features_dir/megaface $features_dir/ms1m $features_dir/flickr

