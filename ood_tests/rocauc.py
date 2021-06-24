import torch
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


def train_model(model_type, data_to_train, model_params):
    """Return a trained model"""
    n_samples_to_train = model_params['n_samples_to_train']
    model_params.pop('n_samples_to_train')
    
    if n_samples_to_train == 'all':
        indices = [i for i in range(len(data_to_train))]
    else:
        try:
            indices = np.random.choice(len(data_to_train), n_samples_to_train,
                                       replace=False)
        except ValueError:
            indices = np.random.choice(len(data_to_train), n_samples_to_train)
        
    if model_type == 'quadrics':
        pass
 
    if model_type == 'OneClassSVM':
        clf = OneClassSVM(**model_params).fit(data_to_train[indices])
        return clf
    
    if model_type == 'KPCA':
        pass
    
    if model_type == 'KPCA_rff':
        pass
    
    if model_type == 'PCA':
        pca = PCA(**model_params).fit(data_to_train)
        return pca

def build_quadratic_monoms(point):
    monoms = []
    for i in range(len(point)):
        cur_monoms = point[i:] * point[i]
        monoms.append(cur_monoms)
    return np.concatenate(monoms)

def get_dists_quadrics_(model, embs, DEVICE='cpu', batch_size=257, alg_dist=False):
    N_steps = embs.shape[0] // batch_size + 1
    res = []
    with torch.no_grad():
        for i in range(N_steps):
            q = np.array([build_quadratic_monoms(e) for e in embs[i*batch_size:(i+1)*batch_size]])
            p = embs[i*batch_size:(i+1)*batch_size]
            q = torch.from_numpy(q.astype('float32')).to(DEVICE)
            p = torch.from_numpy(p.astype('float32')).to(DEVICE)
            values = torch.abs(model.get_values((q,p)))
            if alg_dist:
                scores = values
            else:
                grad = torch.sum(model.get_gradients((q,p))**2, dim=2)
                scores = torch.sqrt(grad/4 + values) - torch.sqrt(grad)/2
            res.append(scores.detach().cpu().numpy())
    return np.concatenate(res, axis=0)

def get_dist(model, model_type, embs, extra_params=None):
    """Insert extra_params in the dictionary format"""
    
    if model_type == 'quadrics':
        extra_params.pop('QPATH', None)
        extra_params.pop('N_QUADRICS', None)
        extra_params.pop('N_COMP', None)
        return np.mean(get_dists_quadrics_(model, embs, **extra_params), axis=1)
    if model_type == 'OneClassSVM':
        return model.score_samples(embs)
    if model_type == 'KPCA':
        return None
    if model_type == 'KPCA_rff':
        return None
    if model_type == 'PCA':
        # use - for roc_auc_score function
        emb_projected = np.dot(embs, model.components_.T)
        emb_length = np.linalg.norm(emb_projected, axis=1)
        return - np.sqrt(1 - emb_length**2)


def get_rocauc(distances_id, distances_ood):
    """Return roc score with 2 arrays as in-distribution
       and out-of-distribution arrays respectively"""
    y = np.array([1]*len(distances_id) + [0]*len(distances_ood))
    return roc_auc_score(1-y, np.concatenate((distances_id, distances_ood), axis=0))

def rocauc_experiments(in_distr_embs, ood_embs, models_dict, in_distr_fraction=4,
                       n_experiments=2, filename='results_ood.csv', extra_params=None):
    """Generates roc auc score for each model in models_dict"""
    in_distr_num = in_distr_fraction*len(ood_embs)
    
    with tqdm(models_dict.keys(), desc="Model", leave=False) as t:
        for model_type in t:
            t.set_description('Model {}'.format(model_type))
            experiment_results = []
            for _ in range(n_experiments):
                try:
                    id_inds = np.random.choice(len(in_distr_embs), in_distr_num, replace=False)
                except ValueError:
                    id_inds = np.random.choice(len(in_distr_embs), in_distr_num)
                
                distances_in_distr = get_dist(models_dict[model_type], model_type, 
                                              in_distr_embs[id_inds], extra_params)
                distances_ood = get_dist(models_dict[model_type], model_type,
                                         ood_embs, extra_params)
                
                experiment_results.append(get_rocauc(distances_in_distr, distances_ood))
            
            yield [model_type, np.mean(experiment_results), np.std(experiment_results)]

