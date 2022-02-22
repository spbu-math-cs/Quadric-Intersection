import numpy as np
import torch
import torch.nn as nn
from math import sqrt


def build_quadratic_monoms_batch(points):
    monoms = []
    for i in range(points.size()[1]):
        cur_monoms = points[:, i:] * points[:, i].unsqueeze(1)
        monoms.append(cur_monoms)
    return torch.cat(monoms, dim=1)


class HSQuadricsModel(nn.Module):
    """
    Main parameters:
        n_quadric: number of quadrics
        dim: dimention of quadrics
        device: device for coefficients
        
        q_coefs: quadratic coefficients matrix. For each quadric coefficient stored as one vector and non-diagonal coefficients
        multiplied by sqrt(2). This matrix orthonormal <=> quadrics orthonormal in sense of Hillbert-Schmidt
        l_coefs: linear coefficients
        free_coefs: free coefficients
        
    Additional parameters:
        inds_trans: indicies to transform q_coefs to quadratic matrix
        coefs_trans: matrix to convert normed q_coefs to real coefficients
    """
    
    def __init__(self, n_quadrics, dim=512):
        super(HSQuadricsModel, self).__init__()
        self.dim = dim
        self.n_quadrics = n_quadrics
        
        #initialize quadratic coefficients
        self.q_coefs =  torch.empty(self.dim * (self.dim-1) // 2 + self.dim, n_quadrics)
        self.q_coefs = torch.nn.init.orthogonal_(self.q_coefs)
        self.q_coefs.requires_grad = True
        self.q_coefs = nn.Parameter(self.q_coefs)
        
        #initialize linear coeficients
        self.l_coefs = (2 * torch.rand((self.dim, self.n_quadrics)) - 1)
        self.l_coefs.requires_grad = True
        self.l_coefs = nn.Parameter(self.l_coefs)
        
        #initialize free coefficients
        self.free_coefs = (2 * torch.rand(1, self.n_quadrics) - 1)
        self.free_coefs.requires_grad = True
        self.free_coefs = nn.Parameter(self.free_coefs)
        
        self.build_transitions()
        # self.coef_trans.requires_grad = True

    def build_transitions(self):
        """
        Build additional transition matrices
        """
        hash_map = np.arange(self.dim*(self.dim-1)//2 + self.dim)
        m = np.ones((self.dim, self.dim), dtype='int')*(-1)
    
        for i in range(self.dim):
            for j in range(i):
                m[i, j] = m[j, i]
            for j in range(i, self.dim):
                start = self.dim*i - i*(i-1)//2
                ind  = start + j - i
                if hash_map[ind] != -1:
                    m[i, j] = hash_map[ind]
                
        self.coef_trans = np.zeros((len(hash_map), ))
        for i in range(self.dim):
            ind = self.dim*i - i*(i-1)//2
            self.coef_trans[hash_map[ind]] = 1
        self.coef_trans[self.coef_trans == 0] = sqrt(2)
        self.coef_trans = np.reshape(self.coef_trans, (-1, 1)).astype('float32')
        self.coef_trans = torch.from_numpy(self.coef_trans)
    
        self.inds_trans = m.reshape((-1,))

    def forward(self, points, dist='dist2'):
        assert dist in ['dist2', 'dist2_full', 'dist1', 'dist0']
        q = build_quadratic_monoms_batch(points)
        if dist=='dist2':
            return self.get_second_order_distances((q, points))
        if dist=='dist1':
            return self.get_first_order_distances((q, points))
        if dist=='dist0':
            return abs(self.get_values((q, points)))
        if dist=='dist2_full':
            return self.get_second_order_distances_full((q, points))

    def build_full_matrix(self):
        """
        Build simetric martices of quadratic parts
        """
        coef_trans = self.coef_trans.to(self.q_coefs.device)
        m = (self.q_coefs*coef_trans)[self.inds_trans]
        return m.view((self.dim, self.dim, self.n_quadrics))

    def get_values(self, points):
        """
        Compute the values of quadrics in the batch of points
        points: (q, p) q - values of base quadratic monoms, p - values of points
        """
        q, p = points
        coef_trans = self.coef_trans.to(self.q_coefs.device)
        q_values = torch.matmul(q, self.q_coefs*coef_trans)
        l_values = torch.matmul(p, self.l_coefs)
        return q_values + l_values + self.free_coefs

    def get_gradients(self, points):
        """
        Compute the values of quadrics gradient in the batch of points
        points: (q, p) q - values of base quadratic monoms, p - values of points
        """
        q, p = points
        
        #compute quadratic part of the gradient
        m = self.build_full_matrix()
        dm = (torch.eye(self.dim) + torch.ones((self.dim, self.dim))).unsqueeze(2).to(m.device)
        m = (m * dm).permute((2, 0, 1))
        q_grad = torch.matmul(p, m).permute((1, 0, 2))
        
        return q_grad + self.l_coefs.permute((1, 0)).unsqueeze(0)

    def get_stiefel_error(self):
        """
        Compute non-orthogonality error of the quadratic part
        """
        I = torch.eye(self.q_coefs.size()[1]).to(self.q_coefs.device)
        return torch.sum((torch.matmul(self.q_coefs.T, self.q_coefs) - I) ** 2)

    def get_first_order_distances(self, points):
        values = torch.abs(self.get_values(points))
        grad = torch.sum(self.get_gradients(points)**2, dim=2)
        scores = values/torch.sqrt(grad)
        return scores
    
    def get_second_order_distances(self, points):
        """
        Compute second order distance to quadrics from the batch of points
        points: (q, p) q - values of base quadratic monoms, p - values of points
        """
        values = torch.abs(self.get_values(points))
        grad = torch.sum(self.get_gradients(points)**2, dim=2)
        norm = torch.norm(self.q_coefs, dim=0)
        scores = (torch.sqrt(grad/4 + values*norm) - torch.sqrt(grad)/2)/norm
        return scores
    
    def get_second_order_distances_full(self, points):
        """
        Compute second order distance to quadrics from the batch of points
        points: (q, p) q - values of base quadratic monoms, p - values of points
        """
        values = torch.abs(self.get_values(points))
        grad = torch.sum(self.get_gradients(points)**2, dim=2)
        scores = torch.sqrt(grad/4 + values) - torch.sqrt(grad)/2
        return scores

    def save(self, path):
        m = self.build_full_matrix()
        params = {'quadratic' : m,
                 'linear' : self.l_coefs,
                 'free' : self.free_coefs}
        torch.save(params, path)

    def load(self, path, **kwargs):
        params = torch.load(path, **kwargs)
        self.dim = params['quadratic'].size()[0]
        self.n_quadrics = params['quadratic'].size()[2]
        q_coefs = torch.empty((self.dim*(self.dim-1)//2 + self.dim, self.n_quadrics))
        self.build_transitions()
        for i in range(self.dim):
            start = self.dim*i - i*(i-1)//2
            q_coefs[start:start + self.dim - i, :] = params['quadratic'][i, i:, :]
        l_coefs = params['linear']
        free_coefs = params['free']
        q_coefs /= self.coef_trans
        q_coefs = q_coefs.to(l_coefs.get_device())
        q_coefs = q_coefs.detach()
        free_coefs = free_coefs.detach()
        l_coefs = l_coefs.detach()
        q_coefs.requires_grad = True
        l_coefs.requires_grad = True
        free_coefs.requires_grad = True
        self.q_coefs = nn.Parameter(q_coefs)
        self.l_coefs = nn.Parameter(l_coefs)
        self.free_coefs = nn.Parameter(free_coefs)
