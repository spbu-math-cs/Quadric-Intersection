import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import pickle 
import os
import argparse
import json
from tqdm import tqdm

from quadrics import Quadrics


def preprocess_cplfw(embeddings):
    """Discard outliers from cplfw datatset"""
    with open('image_embeddings/labels/cplfw_labels_full.txt', encoding='utf-8') as txt_labels_file:
        lines = txt_labels_file.readlines()
    cplfw_full_labels = np.array([i.rstrip('\n') for i in lines])

    with open('image_embeddings/labels/cplfw_outliers_labels.json') as json_file:
        outliers = np.array(json.load(json_file))
    ind = (1 - np.in1d(cplfw_full_labels, outliers)).astype(bool)
    
    return embeddings[ind]


def get_dist(model, model_type, embs, normalize_emb=True, extra_params=None):
    """Insert extra_params in the dictionary format"""
    if normalize_emb and model_type != 'norms':
        embs = normalize(embs)

    if model_type == 'OneClassSVM':
        return - model.score_samples(embs)
    if model_type == 'KPCA':
        return None
    if model_type == 'PCA':
        # use '-' for roc_auc_score function
        emb_projected = np.dot(embs, model.components_.T)
        emb_length = np.linalg.norm(emb_projected, axis=1)
        return np.sqrt(1 - emb_length**2)
    if model_type == 'norms':
        return - np.linalg.norm(embs, axis=1)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--shuffle",
                        action='store_true',
                        help='Usethis flag for ood test')                        
    parser.add_argument("--datasets",
                        nargs='+',
                        default=None,
                        help='To use datasets from config file leave empty')
    parser.add_argument("--methods",
                        nargs='+',
                        default=None,
                        help='Set empty to calculate all methods from config file')
    args = parser.parse_args()

    with open("config.json", "r") as read_file:
        config_file = json.load(read_file)
    
    config_dict = config_file['test_params']
    dir_models = config_file['models_dir']
    dir_features = config_file["features_dir"]
    if args.datasets is None:
        args.datasets = config_file['datasets_to_calculate']
    if args.methods is None:
        args.methods = config_file['models_to_calculate']
        
    proportion_of_outliers = config_dict['proportion_of_outliers']
    n_experiments = config_dict['n_experiments']
    
    outliers = np.load('image_embeddings/cplfw_anime_outliers.npy')
    emb_list = {ds: np.load('image_embeddings/'+ds+'.npy')
                for ds in args.datasets}
    
    if args.shuffle:
        if 'cplfw' in emb_list.keys():
            emb_list['cplfw'] = preprocess_cplfw(emb_list['cplfw'])
        n_emb_experiment = int(len(outliers) / proportion_of_outliers * n_experiments)
        emb_list = {key: value[np.random.choice(len(value), n_emb_experiment)]
                   for key, value in emb_list.items()}
    
    emb_list['outliers'] = outliers
    
    pbar = tqdm(emb_list.items())
    for dataset_name, embeddings in pbar:
        pbar.set_description(dataset_name)        
        dir_dataset = dir_features + '/' + dataset_name
        if 'quadrics' in args.methods:
            clf = Quadrics(dist='dist2', device='gpu')
            clf.load(dir_models + '/Quadrics.pth')
            dist = clf.get_distances(embeddings, dist='dist2')
            np.save(dir_dataset + '/Quadrics.npy')
        if 'quadrics_alg' in args.methods:
            clf = Quadrics(dist='dist0', device='gpu')
            clf.load(dir_models + '/Quadrics_algebraic.pth')
            dist = clf.get_distances(embeddings, dist='dist0')
            np.save(dir_dataset + '/Quadrics_algebraic.npy')
        if 'OneClassSVM' in args.methods:
            clf = pickle.load(open(dir_models + '/OneClassSVM.pickle', 'rb'))
            dist = get_dist(clf, 'OneClassSVM', embeddings, config_dict['normalize'])
            np.save(dir_dataset + '/OneClassSVM_dist.npy', dist)
        if 'PCA' in args.methods:
            clf = pickle.load(open(dir_models+'/PCA.pickle', 'rb'))
            dist = get_dist(clf, 'PCA', embeddings, config_dict['normalize'])
            np.save(dir_dataset + '/PCA_dist.npy', dist)
        if 'norms' in args.methods:
            dist = get_dist(None, 'norms', embeddings, False)
            np.save(dir_dataset + '/norms_dist.npy', dist)
