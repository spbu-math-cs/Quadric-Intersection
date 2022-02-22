import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import argparse
import json
import pickle
import os

from quadrics import Quadrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods",
                        nargs='+',
                        default=['OneClassSVM', 'PCA'],
                        help='Set empty to calculate OneClassSVM and PCA methods from config file') 
    args = parser.parse_args()
    
    # we use embeddings normalization for training
    print('Loading embeddings and config')
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

    if not(os.path.isdir(dir_models)):
        os.mkdir(dir_models)

    if 'OneClassSVM' in methods:
        print('OneClassSVM training...')
        embs_indices = shuffle_indices[:config_dict['OneClassSVM']['n_points']]
        config_dict['OneClassSVM'].pop('n_points')
        clf = OneClassSVM(**config_dict['OneClassSVM']).fit(embeddings[embs_indices])
        pickle.dump(clf, open(dir_models+'/OneClassSVM.pickle', 'wb'))
        print('Model saved to '+dir_models+'/OneClassSVM.pickle')

    if 'PCA' in methods:
        print('PCA training...')
        if config_dict['PCA']['n_points'] == 'all':
            embs_indices = shuffle_indices
        else:
            embs_indices = shuffle_indices[:config_dict['PCA']['n_points']]
        config_dict['PCA'].pop('n_points')
        
        clf = PCA(**config_dict['PCA']).fit(embeddings[embs_indices])
        pickle.dump(clf, open(dir_models+'/PCA.pickle', 'wb'))
        print('Model saved to '+dir_models+'/PCA.pickle')

    if 'quadrics' in methods:
        print('Quadrics with type-2 distance training...')
        if config_dict['quadrics']['n_points'] == 'all':
            embs_indices = shuffle_indices
        else:
            embs_indices = shuffle_indices[:config_dict['quadrics']['n_points']]
        n_quadrics = config_dict["quadrics"]["n_quadrics"]
        distance = config_dict['quadrics']['distance']
        lr = config_dict['quadrics']['lr']
        n_epoch = config_dict['quadrics']['n_epoch']
        device = config_dict['quadrics']['device']
        batch_size = config_dict['quadrics']['batch_size']
        val_size = config_dict['quadrics']['val_size']
        clf = Quadrics(n_quadrics=n_quadrics, dist=distance, device=device)
        if val_size > 0:
            train_size = len(embeddings) - val_size
            assert train_size > 0, "Validation size bigger than total length!!!"
            val_dataset = embeddings[train_size:(train_size + val_size), :]
            train_dataset = embeddings[:train_size, :]
        else:
            train_dataset = embeddings
            val_dataset = None 
        clf.fit(train_dataset, 
                n_epoch, 
                learning_rate=lr, 
                batch_size=batch_size, 
                val_data=val_dataset)
        clf.save(dir_models+'/Quadrics.pth')
        print('Model saved to '+dir_models+'/Quadrics.pth')

    if "quadrics_algebraic" in methods:
        print('Quadrics with algebraic distance training...')
        if config_dict['quadrics_algebraic']['n_points'] == 'all':
            embs_indices = shuffle_indices
        else:
            embs_indices = shuffle_indices[:config_dict['quadrics_algebraic']['n_points']]
        n_quadrics = config_dict["quadrics_algebraic"]["n_quadrics"]
        distance = 'dist0'
        lr = config_dict['quadrics_algebraic']['lr']
        n_epoch = config_dict['quadrics_algebraic']['n_epoch']
        device = config_dict['quadrics_algebraic']['device']
        batch_size = config_dict['quadrics_algebraic']['batch_size']
        val_size = config_dict['quadrics_algebraic']['val_size']
        
        clf = Quadrics(n_quadrics=n_quadrics, dist=distance, device=device)
        if val_size > 0:
            train_size = len(embeddings) - val_size
            assert train_size > 0, "Validation size bigger than total length!!!"
            val_dataset = embeddings[train_size:(train_size + val_size), :]
            train_dataset = embeddings[:train_size, :]
        else:
            train_dataset = embeddings
            val_dataset = None 
        clf.fit(train_dataset, 
                n_epoch, 
                learning_rate=lr, 
                batch_size=batch_size, 
                val_data=val_dataset)
        clf.save(dir_models+'/Quadrics_algebraic.pth')
        print('Model saved to '+dir_models+'/Quadrics_algebraic.pth')
        