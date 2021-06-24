import torch
import sys
from sklearn.preprocessing import normalize
import pandas as pd
from tqdm import tqdm
from ood_tests.rocauc import *
from ood_tests.data_reader import read_ood, read_dataset
from quadrics.model import HSQuadricsModel


params = {
    'quadrics': {
        'n_samples_to_train': 'all', 
        'QPATH': '/workspace/models/100_kPCA_quadrics_46.pth',
        'N_QUADRICS': 200,
        'N_COMP': 512,
        'DEVICE': torch.device('cuda:0'),
        'alg_dist': True
    },
    'OneClassSVM': {
        'n_samples_to_train': 4000, 
        'kernel': 'poly',
        'degree': 3
    },
    'PCA': {
        'n_samples_to_train': 100000,
        'n_components': 170
    }
}


config = dict()
config['anime'] = {'path':'/workspace/data/anime.npy', 'n': 235}
config['cplfw'] = '/workspace/data/cplfw_retina_embs'
config['our_ood'] = '/workspace/data/cplfw_outv2.json'
config['mtcnn_ood'] = '/workspace/data/cplfw_intersection_out.json'

datasets = {'ms1m':'/workspace/data/ms1m_new_float32.npy',
            'vgg2':'/workspace/data/embs_vgg2.npy',
            'megaface':'/workspace/data/megaface.npy',
            'ffhq':'/workspace/data/ffhq_embs',
            'calfw':'/workspace/data/calfw_embs'}

# models we train
models_we_use = ['OneClassSVM', 'quadrics', 'PCA']
filename_to_save = 'results.csv'


if __name__ == '__main__':
    # read dataset to train on
    data_to_train = normalize(np.load(datasets['megaface']))
    
    # load ood embeddings, cplfw with no outliers
    ood_embs, cplfw_embs = read_ood(config)
    
    # prepare results table
    iterator = iter([i for i in range(1000)])
    results_table = pd.DataFrame(columns=['dataset', 'model', 'mean', 'std'])
    
    # train and load models
    models_dict = {}
    for model in models_we_use:
        # fit a model, return a dict
        models_dict[model] = train_model(model, data_to_train, params[model])
    
    if 'quadrics' in set(models_we_use):
        # load quadrics model
        model = HSQuadricsModel(params['quadrics']['N_QUADRICS'], dim=params['quadrics']['N_COMP'])
        model.load(params['quadrics']['QPATH'], map_location=params['quadrics']['DEVICE'])
        model = model.to(params['quadrics']['DEVICE'])
        models_dict['quadrics'] = model
    
    # cplfw no distractors
    for result in rocauc_experiments(cplfw_embs, ood_embs, models_dict, extra_params=params['quadrics']):
        results_table.loc[next(iterator)] = ['cplfw'] + result
    
    for dataset_name, dataset_dir in tqdm(datasets.items(), desc="Dataset"):
        in_distr_embs = read_dataset(dataset_dir)
        for result in rocauc_experiments(in_distr_embs, ood_embs, models_dict, extra_params=params['quadrics']):
            results_table.loc[next(iterator)] = [dataset_name] + result
    
    results_table.index = pd.MultiIndex.from_arrays([results_table.dataset.tolist(), results_table.model.tolist()])
    results_table.to_csv(filename_to_save, index=False)
                
            
