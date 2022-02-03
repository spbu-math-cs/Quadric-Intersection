from quadrics.dataset import PointsDataset

from torch.utils.data import DataLoader
import torch

from tqdm import tqdm
import numpy as np


class Trainer:

    def __init__(self,
                 model,
                 data,
                 val_data=None,
                 batch_size=256,
                 learning_rate=0.1,
                 save_path=None,
                 log_path=None,
                 start_epoch=1,
                 device=torch.device('cpu'),
                 shuffle=True,
                 dist='dist2',
                 lam=1,
                 vebrose=1,
                 ):
        assert dist in ['dist2', 'dist2_full', 'dist1', 'dist0']
        self.dist = dist
        self.model = model.to(device)
        self.batch_size = batch_size
        self.save_path = save_path
        self.log_path = log_path
        self.start_epoch = start_epoch
        self.device = device
        self.lam = lam
        self.logger = Logger(start_epoch=start_epoch, print_every=vebrose)
        self.train_loader = DataLoader(PointsDataset(data),
                                       batch_size=self.batch_size,
                                       shuffle=shuffle)
        if val_data is not None:
            self.val_loader = DataLoader(PointsDataset(val_data), batch_size=self.batch_size)
        else:
            self.val_loader = None
        self.optimizer = torch.optim.SGD(
            [self.model.q_coefs, self.model.l_coefs, self.model.free_coefs],
            lr=learning_rate)

    def get_dists(self, batch):
        if self.dist == 'dist2':
            return torch.sum(self.model.get_second_order_distances(batch), dim=1)
        if self.dist == 'dist1':
            return torch.sum(self.model.get_first_order_distances(batch), dim=1)
        if self.dist == 'dist0':
            return abs(self.model.get_values(batch))
        if self.dist == 'dist2_full':
            return torch.sum(self.model.get_second_order_distances_full(batch), dim=1)

    def train_step(self, batch):
        self.optimizer.zero_grad()
        dists = torch.mean(self.get_dists(batch))
        penalty = self.model.get_stiefel_error()
        loss = dists + self.lam * penalty
        loss.backward()
        self.optimizer.step()

    def train_epoch(self):
        for points in self.train_loader:
            q, p = points
            q = q.to(self.device)
            p = p.to(self.device)
            self.train_step((q, p))

    def train_loop(self, n_epoch):
        for i in range(self.start_epoch, self.start_epoch + n_epoch):
            self.logger.print(f'Epoch {i}')
            self.train_epoch()
            if self.val_loader is not None:
                mean_loss = self.validate()
            er = self.model.get_stiefel_error().cpu().item()
            self.logger.print(f'Orthonormal error: {er}')
            if self.log_path is not None:
                s = f'Epoch {i}:'
                if self.val_loader is not None:
                    s += f' Val loss {mean_loss}'
                s += f' Orthonormal error {er}'
                with open(self.log_path, 'a+') as file:
                    file.write('\n' + s)
            if self.save_path is not None:
                self.model.save(self.save_path + '_' + str(i) + '.pth')
            self.logger.step()
        self.start_epoch += n_epoch

    def validate(self):
        loss_accum = []
        with torch.no_grad():
            n = 0
            for n, x in enumerate(self.val_loader):
                q, p = x
                q = q.to(self.device)
                p = p.to(self.device)
                loss_accum.append(np.mean(self.get_dists((q, p)).detach().cpu().numpy()))
            mean_loss = np.mean(loss_accum)
            self.logger.print('Mean val distance', mean_loss)
        return mean_loss

    
class Logger:
    
    def __init__(self, start_epoch=1, print_every=1):
        self.cur_epoch = start_epoch
        self.print_every = print_every
        
    def print(self, message):
        if (self.cur_epoch % self.print_every) == 0:
            print(message)
            
    def step(self):
        self.cur_epoch += 1