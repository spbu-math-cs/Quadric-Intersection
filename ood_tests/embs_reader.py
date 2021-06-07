import numpy as np
import os
import json


def get_emb_full(name, path):
    emb = np.load(os.path.join(path, name))
    return emb

def read_one_file(path):
    embs = np.load(path)
    embs = embs/np.linalg.norm(embs, axis=1)[:, np.newaxis]
    return embs

def read_embs(names, embs_path):
    embs_full = []
    for name in names:
        emb = get_emb_full(name, embs_path)
        embs_full.append(emb)
    embs_full = np.array(embs_full)
    ood_embs = embs_full/np.linalg.norm(embs_full, axis=1)[:, np.newaxis]
    return ood_embs

def cplfw_reader(embs_path, our_ood_path, mtcnn_ood_path):
    emb_list = set(os.listdir(embs_path))
    
    with open(our_ood_path) as file:
        l = file.read()
    our_names = json.loads(l)

    with open(mtcnn_ood_path) as file:
        l = file.read()
    mtcnn_names = json.loads(l)
    
    ood_names = list(set(our_names)|set(mtcnn_names))
    ood_names = set([name+'.npy' for name in ood_names])
    id_names = list(emb_list - ood_names)
    ood_names = list(ood_names)

    ood_embs = read_embs(ood_names, embs_path)
    id_embs = read_embs(id_names, embs_path)
    return id_embs, ood_embs
