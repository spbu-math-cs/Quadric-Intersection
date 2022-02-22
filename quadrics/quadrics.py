import torch
import numpy as np
from quadrics.model import HSQuadricsModel
from quadrics.trainer import Trainer


class Quadrics:
    """
    An interface to interact with quadrics
    Parameters:
        n_quadrics: number of quadrics
        dist: type of distance used. Possible values: 'dist2', 'dist2_full', 'dist1', 'dist0'
        device: device used. Posible values: 'cpu', 'gpu'
    """

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

    def get_distances(self, point, dist=None, batch_size=32):
        """
        Compute distance from point to quadrics
        Parameters:
            point: ndarray with point coordinates or batch of points
            dist: type of distance to use. If None the default distance from __init__ will be used
        Returns:
            ndarray with distances to each quadric
        """
        assert dist in [None, 'dist2', 'dist2_full', 'dist1', 'dist0']
        assert self.model is not None

        if dist is None:
            dist = self.dist

        if len(point.shape) == 1:
            assert len(point) == self.model.dim
        
            point = torch.from_numpy(point.astype('float32')).unsqueeze(0).to(self.device)
            with torch.no_grad():
                r = self.model(point, dist=dist).detach().cpu().squeeze(0).numpy()
            return r
        if len(point.shape) == 2:
            assert point.shape[1] == self.model.dim
            if point.shape[0] % batch_size == 0:
                N_steps = point.shape[0] // batch_size
            else:
                N_steps = point.shape[0] // batch_size + 1
            r = []
            with torch.no_grad():
                for i in range(N_steps):
                    batch = torch.from_numpy(point[i*batch_size:(i+1)*batch_size].astype('float32')).to(self.device)
                    scores = self.model(batch, dist=dist).detach().cpu().numpy()
                    r.append(scores)
            return np.concatenate(r, axis=0)
        else:
            raise ValueError('point must be ndarray of dim 1 or 2')
    
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
            vebrose=1,
            ):
        """
        Fit quadrics. If there is a trained model, continue training
        Parameters:
            data: ndarray with data to train
            n_epochs: number of epochs to train
            val_data: None or ndarray with data for validation
            batch_size: batch size
            learning_rate: learning rate
            save_path: path to save weights every epoch. '_{step_number}.pth' will be added. If None model will no be saved
            log_path: path to text file to save log. If None no log will be saved
            shuffle: If true will shuffle data befor training
            lam: weidht of regularization term
            start_epoch: number of starting epoch to correct loging
        """
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
                          start_epoch=start_epoch,
                          vebrose=vebrose,
                          )
        trainer.train_loop(n_epochs)
