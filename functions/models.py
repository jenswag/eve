# Enhanced Vignes Extension (EVE): functions/models.py
# Models
# ---
# Jens Wagner, 11.01.2026
# ----------------------------------------------------------------------------------------------------------------------

# Import Packages and Modules
import torch
import math
import torch.nn as nn
import numpy as np

from torch.nn import init


# Invariant layer
class InvLinear(nn.Module):

    def __init__(self, in_features, out_features, bias=True, reduction='sum'):
        super(InvLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        assert reduction in ['mean', 'sum', 'max', 'min'],  \
            '\'reduction\' should be \'mean\'/\'sum\'\'max\'/\'min\', got {}'.format(reduction)

        self.reduction = reduction

        self.beta = nn.Parameter(torch.Tensor(self.in_features,
                                              self.out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, self.out_features))

        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):

        init.xavier_uniform_(self.beta)

        if self.bias is not None:

            fan_in, _ = init._calculate_fan_in_and_fan_out(self.beta)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, X, mask=None):

        N, M, _ = X.shape
        device = X.device

        if mask is None:
            mask = torch.ones(N, M).byte().to(device)

        if self.reduction == 'mean':
            sizes = mask.float().sum(dim=1).unsqueeze(1)
            Z = X * mask.unsqueeze(2).float()
            y = (Z.sum(dim=1) @ self.beta)/sizes

        elif self.reduction == 'sum':
            Z = X * mask.unsqueeze(2).float()
            y = Z.sum(dim=1) @ self.beta

        elif self.reduction == 'max':
            Z = X.clone()
            Z[~mask] = float('-Inf')
            y = Z.max(dim=1)[0] @ self.beta

        else:  # min
            Z = X.clone()
            Z[~mask] = float('Inf')
            y = Z.min(dim=1)[0] @ self.beta

        if self.bias is not None:
            y += self.bias

        return y

    def extra_repr(self):

        return 'in_features={}, out_features={}, bias={}, reduction={}'.format(
            self.in_features, self.out_features,
            self.bias is not None, self.reduction)


# Equivariant layer
class EquivLinear(InvLinear):

    def __init__(self, in_features, out_features, bias=True, reduction='sum'):

        super(EquivLinear, self).__init__(in_features, out_features,
                                          bias=bias, reduction=reduction)

        self.alpha = nn.Parameter(torch.Tensor(self.in_features,
                                               self.out_features))

        self.reset_parameters()

    def reset_parameters(self):

        super(EquivLinear, self).reset_parameters()
        if hasattr(self, 'alpha'):
            init.xavier_uniform_(self.alpha)

    def forward(self, X, mask=None):

        device = X.device
        N, M, _ = X.shape

        if mask is None:
            mask = torch.ones(N, M).byte().type(torch.bool).to(device)

        Y = torch.zeros(N, M, self.out_features).to(device)
        h_inv = super(EquivLinear, self).forward(X, mask=mask)
        Y[mask] = (X @ self.alpha + h_inv.unsqueeze(1))[mask]

        return Y


# Invariant Deep Set Model (iDSM)
class iDSM(nn.Module):
    def __init__(self, inp=6, nodes=32, out=6, afunc=nn.SiLU(inplace=True), phi=0, rho=0):
        super(iDSM, self).__init__()

        self.inp = inp
        self.nodes = nodes
        self.out = out
        self.phi = phi
        self.rho = rho
        self.afunc = afunc

        # Encoder phi
        layers = []
        for _ in range(self.phi):
            layers.append(nn.Linear(self.nodes, self.nodes))
            layers.append(self.afunc)
        self.enc = nn.Sequential(nn.Linear(self.inp, self.nodes), self.afunc, *layers)

        self.equ = EquivLinear(self.nodes, self.nodes)

        # Decoder phi
        layers = []
        for _ in range(self.rho):
            layers.append(nn.Linear(self.nodes, self.nodes))
            layers.append(self.afunc)
        self.dec = nn.Sequential(*layers, nn.Linear(self.nodes, self.out), nn.Softplus())

        # SEB
        self.seb = nn.Sequential(nn.Linear(12, 32), self.afunc,
                    nn.Linear(32, 16), self.afunc,
                    nn.Linear(16, 1), nn.Softplus())

    def forward(self, x, mask):

        smask = mask.clone()
        smask[smask != -42.] = 1.

        M = self.enc(x)

        M = self.equ(M * mask.unsqueeze(-1), smask.type(torch.bool))
        M = self.afunc(M)

        M = self.dec(M)

        M = (x * mask.unsqueeze(-1) * M).sum(-2)

        return self.seb(torch.cat((x[:, 0], M), -1)), M


# Equivariant Deep Set Model (eDSM)
class eDSM(nn.Module):
    def __init__(self, inp=8, nodes=32, out=1, afunc=nn.SiLU(inplace=True), phi=0, rho=0):
        super(eDSM, self).__init__()

        self.inp = inp
        self.nodes = nodes
        self.out = out
        self.phi = phi
        self.rho = rho
        self.afunc = afunc

        # Encoder phi
        layers = []
        for _ in range(self.phi):
            layers.append(nn.Linear(self.nodes, self.nodes))
            layers.append(self.afunc)
        self.enc = nn.Sequential(nn.Linear(self.inp, self.nodes), self.afunc, *layers)

        self.equ = EquivLinear(self.nodes, self.nodes)

        # Decoder phi
        layers = []
        for _ in range(self.rho):
            layers.append(nn.Linear(self.nodes, self.nodes))
            layers.append(self.afunc)
        self.dec = nn.Sequential(*layers, nn.Linear(self.nodes, self.out))

    def forward(self, x, mask, dI, dJ):

        smask = mask.clone()
        smask[smask != -42.] = 1.

        I = torch.cat([x[:, 0], dI[:, :2]], -1)
        J = torch.cat([x[:, 1], dJ[:, :2]], -1)

        y = self.enc(torch.stack([I, J], 1))
        y = self.equ(y, smask.type(torch.bool))
        y = self.afunc(y)
        y = self.dec(y)
        y = y.sum(-2)

        pred = torch.exp((mask[:, 0] * torch.log(dJ[:, -1]) + mask[:, 1] * torch.log(dI[:, -1])) + (
                    1 - mask[:, 0].pow(12) - mask[:, 1].pow(12)) * y.squeeze())

        eps = torch.finfo(pred.dtype).eps

        return torch.clamp(pred, min=eps)


# Ideal Viscosity
def IV(v_j, x_j):

    return np.exp(np.dot(x_j, np.log(v_j)))


# Stokes-Einstein (SE) Equation: M -> D
def SE(mma_i, eta_j, tmp_j, packing=0.64, rho_bulk=1.05e3, k_B=1.380649e-23, N_A=6.02214076e23):

    sdc = (k_B * tmp_j) / (6 * np.pi * eta_j * np.cbrt((3 * packing * mma_i) / (4 * np.pi * rho_bulk * N_A)))

    return sdc


# Vignes Equation (VE)
def VE(D_ij, x_j):

    return np.exp((x_j * np.log(D_ij)).sum())