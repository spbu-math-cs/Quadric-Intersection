import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
import argparse
import json
import pickle
import os


# argparse
methods = ['OneClassSVM', 'PCA']
dir_models = 'models'


def train_pca(embeddings, n_samples):
    """Returns basis"""
    pass


if __name__ == 'main':

    assert 'ms1m' in os.listdir('image_embeddings'), 'We train model on ms1m dataset!' 
    
    with open("config.json", "r") as read_file:
        config_dict = json.load(read_file)['train_params']

    if 'OneClassSVM' in methods:
        embs_train = np.random.choice(embeddings, 
                                      size=config_dict['OneClassSVM']['n_samples'], replace=False)
        config_dict['OneClassSVM'].pop('n_samples')
        clf = OneClassSVM(**config_dict['OneClassSVM']).fit(emb_train) 
        pickle.dump(clf, open(dir_models+'/OneClassSVM.pickle', 'wb'))

    if 'PCA' in methods:
        if config_dict['PCA']['n_samples'] == 'all':
            embs_train = np.copy(embeddings)
        else:
            embs_train = np.random.choice(embeddings, size=config_dict['PCA']['n_samples'], replace=False)
        config_dict['PCA'].pop('n_samples')
        clf = PCA(**config_dict['PCA']).fit(embs_train)
        pickle.dump(clf, open(dir_models+'/PCA.pickle', 'wb')) 

 
