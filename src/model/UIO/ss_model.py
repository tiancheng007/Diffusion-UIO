import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

scale_level = 1
veh_lf = 1.08
veh_lr = 1.24
veh_l_sum = 2.32
veh_m = scale_level * 1077
veh_Iz = scale_level * 1442
veh_Rt = 0.26
veh_Cf = 47135
veh_Cr = 56636
veh_Cx = 0.3877
veh_Cy = 0.43

r_min = -1.1
r_max = 1.1
vx_min = 5
vx_max = 30

delta_t = 0.01

veh_lf = torch.tensor(veh_lf, device='cuda', requires_grad=False)
veh_lr = torch.tensor(veh_lr, device='cuda', requires_grad=False)
veh_l_sum = torch.tensor(veh_l_sum, device='cuda', requires_grad=False)
veh_m = torch.tensor(veh_m, device='cuda', requires_grad=False)
veh_Iz = torch.tensor(veh_Iz, device='cuda', requires_grad=False)
veh_Rt = torch.tensor(veh_Rt, device='cuda', requires_grad=False)

veh_Cf = torch.tensor(veh_Cf, device='cuda', requires_grad=False)
veh_Cr = torch.tensor(veh_Cr, device='cuda', requires_grad=False)
veh_Cx = torch.tensor(veh_Cx, device='cuda', requires_grad=False)
veh_Cy = torch.tensor(veh_Cy, device='cuda', requires_grad=False)

r_min = torch.tensor(r_min, device='cuda', requires_grad=False)
r_max = torch.tensor(r_max, device='cuda', requires_grad=False)
vx_min = torch.tensor(vx_min, device='cuda', requires_grad=False)
vx_max = torch.tensor(vx_max, device='cuda', requires_grad=False)


class GSSModel():
    def __init__(self):
        self.x_dim = None
        self.y_dim = None

    def h_cal(self, vx_obs):
        raise NotImplementedError()

    def A_D_cal(self):
        raise NotImplementedError()


class UIO_Model(GSSModel):
    def __init__(self, mode='test'):
        super().__init__()
        self.mode = mode

        self.x_dim = 2
        self.y_dim = 1
    
    def h_cal(self, vx_obs):
        h1 = (1 / vx_obs - 1 / vx_max) / (1 / vx_min - 1 / vx_max)
        h3 = (vx_obs - vx_min) / (vx_max - vx_min)

        h2 = 1 - h1 - h3

        return h1, h2, h3

    def A_D_cal(self):

        I_mat = torch.eye(self.x_dim)

        D11 = 2 * veh_Cf / veh_m
        D21 = 2 * veh_lf * veh_Cf / veh_Iz

        a1 = -2 * (veh_Cf + veh_Cr) / (veh_m * vx_max)
        a2 = -2 * (veh_Cf + veh_Cr) / (veh_m * vx_min)

        a3 = 2 * (veh_lr * veh_Cr - veh_lf * veh_Cf) / (veh_Iz * vx_max)
        a4 = 2 * (veh_lr * veh_Cr - veh_lf * veh_Cf) / (veh_Iz * vx_min)

        a5 = -2 * (veh_lf ** 2 * veh_Cf + veh_lr ** 2 * veh_Cr) / (veh_Iz * vx_max)
        a6 = -2 * (veh_lf ** 2 * veh_Cf + veh_lr ** 2 * veh_Cr) / (veh_Iz * vx_min)

        A1 = delta_t * torch.tensor([[a2, a4 - vx_min], [a4, a6]])+I_mat

        A2 = delta_t * torch.tensor([[a1, a3 - vx_min], [a3, a5]])+I_mat

        A3 = delta_t * torch.tensor([[a1, a3 - vx_max], [a3, a5]])+I_mat

        Dv_mat = torch.tensor([[D11], [D21]]).cuda()

        Dv_mat *= delta_t

        return A1.cuda(), A2.cuda(), A3.cuda(), Dv_mat


class Uncertainty_Est_NN(nn.Module):
    def __init__(self):
        super(Uncertainty_Est_NN, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 512)
        self.fc3 = nn.Linear(512, 64)
        self.fc4 = nn.Linear(64, 2)
        # self.dropout = nn.Dropout(p=0.1)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)
        # nn.init.xavier_uniform_(self.fc5.weight)

    def forward(self, obs):
        obs_norm = F.normalize(obs, p=2, dim=0, eps=1e-12)
        x = F.leaky_relu(self.fc1(obs_norm.transpose(0, 1)))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))

        uncertain_est = self.fc4(x)

        return uncertain_est.reshape(2, 1)



