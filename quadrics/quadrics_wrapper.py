import torch
from torch.utils.data import Dataset
from quadrics.HSQuadricsModel import HSQuadricsModel
from quadrics.PointsDataset import PointsDataset, build_quadratic_monoms


class Quadrics:

    def __init__(self, n_quadrics=100, dist='dist2'):
        assert dist in ['dist2', 'dist2_full', 'dist1', 'dist0']
        self.dist = dist
        self.n_quadrics = n_quadrics
        self.model = None

    def get_distance(self, point, dist=None):
        assert dist in [None, 'dist2', 'dist2_full', 'dist1', 'dist0']
        assert self.model is not None
        assert len(point) == self.model.dim
        if dist is None:
            dist = self.dist
        point = torch.from_numpy(point.astype('float32')).unsqueeze(0)
        with torch.no_grad():
            r = self.model(point, dist=dist).detach().cpu().item()
        return r

    def load(self, path):
        self.model = HSQuadricsModel(self.n_quadrics)
        self.model.load(path)
        self.n_quadrics = self.model.n_quadrics
