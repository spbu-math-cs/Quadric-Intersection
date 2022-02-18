import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import argparse
import json
import pickle
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods",
                        nargs='+',
                        default=None,
                        help='Set empty to calculate all methods from config file')
    args = parser.parse_args()

    assert 'ms1m' in os.listdir('image_embeddings'), 'We train model on ms1m dataset!'
    
    # we use embeddings normalization for training
    embeddings = normalize(np.load('image_embeddings/ms1m.npy')) 
    shuffle_indices = np.arange(len(embeddings))
    np.random.shuffle(shuffle_indices)
        
    with open("config.json", "r") as read_file:
        config_file = json.load(read_file)

    config_dict = config_file['train_params']
    dir_models = config_file['models_dir']

    if args.methods is None:
        methods = config_dict.keys()
    else: methods = args.methods

    if 'OneClassSVM' in methods:
        embs_indices = shuffle_indices[:config_dict['OneClassSVM']['n_points']]
        config_dict['OneClassSVM'].pop('n_points')
        clf = OneClassSVM(**config_dict['OneClassSVM']).fit(embeddings[embs_indices])
        pickle.dump(clf, open(dir_models+'/OneClassSVM.pickle', 'wb'))

    if 'PCA' in methods:
        if config_dict['PCA']['n_points'] == 'all':
            embs_indices = shuffle_indices
        else:
            embs_indices = shuffle_indices[:config_dict['PCA']['n_points']]
        config_dict['PCA'].pop('n_points')
        
        clf = PCA(**config_dict['PCA']).fit(embeddings[embs_indices])
        pickle.dump(clf, open(dir_models+'/PCA.pickle', 'wb'))
