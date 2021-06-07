import numpy as np
import os

from ood_tests.embs_reader import cplfw_reader, read_one_file, read_embs


def read_ood(config):
    ood_anime = read_one_file(config['anime']['path'])[:config['anime']['n']]
    id_cplfw, ood_cplfw = cplfw_reader(config['cplfw'], 
                                       config['our_ood'], 
                                       config['mtcnn_ood'])
    ood_embs = np.concatenate((ood_anime, ood_cplfw), axis=0)
    return ood_embs, id_cplfw

def read_dataset(path):
    if path[-4:] == '.npy':
        embs = read_one_file(path)
        return embs
    emb_list = os.listdir(path)
    return read_embs(emb_list, path)