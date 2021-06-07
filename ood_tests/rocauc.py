import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


def build_quadratic_monoms(point):
    monoms = []
    for i in range(len(point)):
        cur_monoms = point[i:] * point[i]
        monoms.append(cur_monoms)
    return np.concatenate(monoms)

def get_dists(model, embs, device, batch_size=257, alg_dist=False):
    N_steps = embs.shape[0] // batch_size + 1
    res = []
    with torch.no_grad():
        for i in range(N_steps):
            q = np.array([build_quadratic_monoms(e) for e in embs[i*batch_size:(i+1)*batch_size]])
            p = embs[i*batch_size:(i+1)*batch_size]
            q = torch.from_numpy(q.astype('float32')).to(device)
            p = torch.from_numpy(p.astype('float32')).to(device)
            values = torch.abs(model.get_values((q,p)))
            if alg_dist:
                scores = values
            else:
                grad = torch.sum(model.get_gradients((q,p))**2, dim=2)
                scores = torch.sqrt(grad/4 + values) - torch.sqrt(grad)/2
            res.append(scores.detach().cpu().numpy())
    return np.concatenate(res, axis=0)

def get_rocauc(model, id_embs, ood_embs, device, batch_size=1500, alg_dist=False):
    y = np.array([1]*len(id_embs) + [0]*len(ood_embs))
    embs = np.concatenate((id_embs, ood_embs), axis=0)
    dists = np.mean(get_dists(model, embs, device, 
        batch_size=batch_size, alg_dist=alg_dist), axis=1)
    d = roc_auc_score(1-y, dists)
    return d

def rocauc_stats(model, id_embs, ood_embs, device, 
                lam=99, iterations=10, batch_size=1500, alg_dist=False):
    id_num = lam*len(ood_embs)
    if id_num >= len(id_embs):
        mean = get_rocauc(model, id_embs, ood_embs, device, batch_size=batch_size, alg_dist=alg_dist)
        std = 0.
        return mean, std
    
    res = []
    for _ in tqdm(range(iterations)):
        id_inds = np.random.choice(len(id_embs), id_num)
        res.append(get_rocauc(model, id_embs[id_inds], ood_embs, device, batch_size=batch_size, alg_dist=alg_dist))
    return np.mean(res), np.std(res)
