import torch
from quadrics.HSQuadricsModel import HSQuadricsModel
from quadrics.trainer import Trainer


class Quadrics:

    def __init__(self, n_quadrics=100, dist='dist2', device='cpu'):
        assert dist in ['dist2', 'dist2_full', 'dist1', 'dist0']
        assert device in ['cpu', 'gpu']
        self.dist = dist
        self.n_quadrics = n_quadrics
        self.model = None
        if device == 'gpu':
            try:
                self.device = torch.device('cuda:0')
            except Exception:
                print('GPU not available, CPU will be used')
                device = 'cpu'
        if device == 'cpu':
            self.device = torch.device('cpu')

    def get_distances(self, point, dist=None):
        assert dist in [None, 'dist2', 'dist2_full', 'dist1', 'dist0']
        assert self.model is not None
        assert len(point) == self.model.dim
        if dist is None:
            dist = self.dist
        point = torch.from_numpy(point.astype('float32')).unsqueeze(0).to(self.device)
        with torch.no_grad():
            r = self.model(point, dist=dist).detach().cpu().squeeze(0).numpy()
        return r

    def load(self, path):
        self.model = HSQuadricsModel(self.n_quadrics)
        self.model.load(path, map_location=self.device)
        self.n_quadrics = self.model.n_quadrics

    def save(self, path):
        assert self.model is not None
        self.model.save(path)

    def fit(self, data,
            n_epochs,
            val_data=None,
            batch_size=256,
            learning_rate=0.1,
            save_path=None,
            log_path=None,
            shuffle=True,
            lam=1,
            start_epoch=1,
            ):
        if self.model is not None:
            assert self.model.dim == data.shape[1]
        else:
            self.model = HSQuadricsModel(self.n_quadrics, dim=data.shape[1])
        trainer = Trainer(self.model,
                          data,
                          val_data=val_data,
                          batch_size=batch_size,
                          learning_rate=learning_rate,
                          save_path=save_path,
                          log_path=log_path,
                          device=self.device,
                          shuffle=shuffle,
                          dist=self.dist,
                          lam=lam,
                          start_epoch=start_epoch
                          )
        trainer.train_loop(n_epochs)
