import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import pickle 
import os
import argparse


dir_models = 'models/'

dir_distances = 'distances/'

def get_dist(model, model_type, embs, normalize=True, extra_params=None):
    """Insert extra_params in the dictionary format"""
    if normalize and model_type != 'norms':
        embs = normalize(embs)

    if model_type == 'quadrics':
        pass
    if model_type == 'OneClassSVM':
        return model.score_samples(embs)
    if model_type == 'KPCA':
        return None
    if model_type == 'PCA':
        # use '-' for roc_auc_score function
        emb_projected = np.dot(embs, model.components_.T)
        emb_length = np.linalg.norm(emb_projected, axis=1)
        return - np.sqrt(1 - emb_length**2)
    if model_type == 'norms':
        return np.sqrt(embs, axis=1)

if __name__ == 'main':
    parser = argparse.ArgumentParser()
    parser.add_argument("--shuffle",
                        type=bool,
                        default=True,
                        help='Set True for IR calculation, False for ood test')

    os.mkdir('distances_to_models')
    with open("config.json", "r") as read_file:
        config_dict = json.load(read_file)['test_params']
        models = json.load(read_file)['models_to_calculate']
        datasets = json.load(read_file)['datasets_to_calculate']
    indices_full = np.random.shuffle(6000000)
#     datasets_
    
    for embeddings in datasets:
        if 'quadrics' in methods:
            pass
        if 'OneClassSVM' in methods:
            clf = pickle.load(open(dir_models+'OneClassSVM.pickle', 'rb'))
            dist = get_dist(clf, 'OneClassSVM', embeddings, config_dict['normalize'])
            np.save(dir_distances+'OneClassSVM_dist.npy', dist)
        if 'PCA' in methods:
            clf = pickle.load(open(dir_models+'PCA.pickle', 'rb'))
            dist = get_dist(clf, 'PCA', embeddings, config_dict['normalize'])
            np.save(dir_distances+'PCA_dist.npy', dist
        if 'norms'in methods:
            dist = get_dist(None, 'norms', embeddings, False)
            np.save(dir_distances+'norms_dist.npy', dist 

 
