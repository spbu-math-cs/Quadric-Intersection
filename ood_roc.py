import torch
from quadrics.model import HSQuadricsModel
from ood_tests.data_reader import read_ood, read_dataset
from ood_tests.rocauc import rocauc_stats

QPATH = '/workspace/models/100_kPCA_quadrics_46.pth'
N_QUADRICS = 200
N_COMP = 512
DEVICE = torch.device('cuda:0')
alg_dist = True

config = dict()
config['anime'] = {'path':'/workspace/data/anime.npy', 'n':235}
config['cplfw'] = '/workspace/data/cplfw_retina_embs'
config['our_ood'] = '/workspace/data/cplfw_outv2.json'
config['mtcnn_ood'] = '/workspace/data/cplfw_intersection_out.json'

datasets = {'ms1m':'/workspace/data/ms1m_new_float32.npy',
            'vgg2':'/workspace/data/embs_vgg2.npy',
            'megaface':'/workspace/data/megaface.npy',
            'ffhq':'/workspace/data/ffhq_embs',
            'calfw':'/workspace/data/calfw_embs'}


model = HSQuadricsModel(N_QUADRICS, dim=N_COMP)
model.load(QPATH, map_location=DEVICE)
model = model.to(DEVICE)
ood_embs, id_embs = read_ood(config)
m, s = rocauc_stats(model, id_embs, ood_embs, DEVICE, alg_dist=alg_dist)
print(f'cplfw: mean {m}, std {s}')
with open('results.txt', 'w+') as file:
    file.write(f'cplfw: mean {m}, std {s}\n')
for data in datasets:
    dataset = datasets[data]
    id_embs = read_dataset(dataset)
    m, s = rocauc_stats(model, id_embs, ood_embs, DEVICE, alg_dist=alg_dist)
    print(data + f': mean {m}, std {s}')
    with open('results.txt', 'a+') as file:
        file.write(data + f': mean {m}, std {s}\n')