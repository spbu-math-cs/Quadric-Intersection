#!/bin/bash
dir="image_embeddings"
dir_quadrics="quadrics_model"
if [ ! -d "$dir" ]; then
  tar -xvzf image_embeddings.tar.gz
fi
models_dir=$(jq -r '.models_dir' config.json)
features_dir=$(jq -r '.features_dir' config.json)
mkdir $features_dir $models_dir
mkdir $features_dir/calfw $features_dir/cplfw $features_dir/megaface $features_dir/ms1m $features_dir/flickr $features_dir/outliers
if [ ! -d "$dir_quadrics" ]; then
  tar xvf quadrics_models.tar.gz
fi
mv quadrics_models/300_ms1m_new_48.pth models/Quadrics.pth
mv quadrics_models/30_ms1m_80k_1000.pth models/Quadrics_algebraic.pth
rm $dir_quadrics
