from cProfile import label
from sympy import total_degree
import torch
from torch._refs import softmax
import torch.nn as nn
import numpy as np
from torch.nn.modules import loss
from transformer import multi_head_self_attention_Block
import torch.nn.functional as F


class MultivariateNormalDiag():  

    def __init__(self, locs, scales):
        super(MultivariateNormalDiag, self).__init__()
        self.locs = locs
        self.scales = scales  

    def sample(self, shape=()): 
        device = self.locs.device
        eps = torch.randn(shape + (self.locs.shape[1],), device=device)
        return self.locs + self.scales * eps


class Encoder_Special(nn.Module):
    def __init__(self, in_img, in_txt, fea_dim, cluster_num, batch_size):
        super(Encoder_Special, self).__init__()
        _initialize_weights(self)
        self.in_img = in_img  #4096
        self.in_txt = in_txt  #768
        self.fea_dim = fea_dim
        self.cluster_num = cluster_num
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.fc1_img = nn.Sequential(
            nn.Linear(self.in_img, int(self.in_img / 2)),
            nn.BatchNorm1d(int(self.in_img / 2)),
            nn.ReLU(),
            nn.Linear(int(self.in_img / 2), 512),
        )

        self.fc1_txt = nn.Sequential(
            nn.Linear(self.in_txt, int(self.in_txt / 2)),
            nn.BatchNorm1d(int(self.in_txt / 2)),
            nn.ReLU(),
            nn.Linear(int(self.in_txt / 2), 512),
        )

        self.encoder_z_img = nn.Linear(512, 2 * 256)
        self.encoder_z_txt = nn.Linear(512, 2 * 256)


        self.fc4_img = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, int(self.in_img / 2)),
            nn.BatchNorm1d(int(self.in_img / 2)),
            nn.ReLU(),
            nn.Linear(int(self.in_img / 2), self.in_img),
            nn.Sigmoid(),
        )

        self.fc4_txt = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, int(self.in_txt / 2)),
            nn.BatchNorm1d(int(self.in_txt / 2)),
            nn.ReLU(),
            nn.Linear(int(self.in_txt / 2), self.in_txt),
            nn.Sigmoid(),
        )

        self.cluster_z_txt = nn.Sequential(
            nn.Linear(256, self.cluster_num),
            nn.BatchNorm1d(self.cluster_num),
            nn.ReLU(),
        )
        self.cluster_z_img = nn.Sequential(
            nn.Linear(256, self.cluster_num),
            nn.BatchNorm1d(self.cluster_num),
            nn.ReLU(),
        )

        self.cluster_txt = nn.Sequential(
            nn.Linear(256, self.cluster_num),
            nn.BatchNorm1d(self.cluster_num),
            nn.ReLU(),
        )
        self.cluster_img = nn.Sequential(
            nn.Linear(256, self.cluster_num),
            nn.BatchNorm1d(self.cluster_num),
            nn.ReLU(),
        )

        self.mhsa = multi_head_self_attention_Block(model_dim=256, num_heads=8)

    def multivariate_normal_diag(self, locs, scales):
        return MultivariateNormalDiag(locs, scales)

    def forward(self, img, txt, epoch):
        fea_img = self.fc1_img(img)
        fea_txt = self.fc1_txt(txt)

        z_img = self.encoder_z_img(fea_img)
        z_txt = self.encoder_z_txt(fea_txt)

        dim = int(z_img.shape[1] / 2)

        prior = self.multivariate_normal_diag(torch.zeros(dim).cuda(), torch.ones(dim).cuda())

        mu_img = z_img[:, :dim]  
        logvar_img = z_img[:, dim:] 
        std_img = F.softplus(logvar_img - 5) 
        z_dist_img = MultivariateNormalDiag(mu_img, std_img)
        input_rdn_img = z_dist_img.sample((mu_img.shape[0],)) 
        input_rdn_l2_img = F.normalize(input_rdn_img, p=2, dim=1)

        mu_txt = z_txt[:, :dim]
        logvar_txt = z_txt[:, dim:] 
        std_txt = F.softplus(logvar_txt - 5) 
        z_dist_txt = MultivariateNormalDiag(mu_txt, std_txt)
        input_rdn_txt = z_dist_txt.sample((mu_txt.shape[0],))  
        input_rdn_l2_txt = F.normalize(input_rdn_txt, p=2, dim=1)

        fea_img = input_rdn_l2_img + self.mhsa(input_rdn_l2_img, input_rdn_l2_txt)
        fea_txt = input_rdn_l2_txt + self.mhsa(input_rdn_l2_txt, input_rdn_l2_img)

        cluster_img = self.cluster_img(fea_img)
        cluster_txt = self.cluster_txt(fea_txt)
        cluster_img = torch.softmax(cluster_img, dim=1)
        cluster_txt = torch.softmax(cluster_txt, dim=1)
        
        recon_img = self.fc4_img(fea_img)
        recon_txt = self.fc4_txt(fea_txt)

#         recon_img = self.fc4_img(input_rdn_l2_img)
#         recon_txt = self.fc4_txt(input_rdn_l2_txt)

        loss_vae = (0.1 * 0.5 * (mu_img.pow(2) + logvar_img.exp() - logvar_img - 1).sum(dim=1).mean() +
                    0.1 * 0.5 * (mu_txt.pow(2) + logvar_txt.exp() - logvar_txt - 1).sum(dim=1).mean() +
                    2 * F.mse_loss(recon_img,img) +
                    2 * 0.5 * F.l1_loss(recon_img, img) +
                    F.mse_loss(recon_txt, txt) +
                    0.5 * F.l1_loss(recon_txt, txt))
        
        return fea_img, fea_txt, cluster_img, cluster_txt, loss_vae



def _initialize_weights(self):
    print("initialize")
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            assert (m.track_running_stats == self.batchnorm_track)
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()


def UD_constraint(classer):
    CL = classer.detach().cpu().numpy()
    N, K = CL.shape
    CL = CL.T
    r = np.ones((K, 1)) / K
    c = np.ones((N, 1)) / N
    CL **= 10
    inv_K = 1. / K
    inv_N = 1. / N
    err = 1e3
    _counter = 0
    while err > 1e-2 and _counter < 100:
        r = inv_K / (CL @ c)
        c_new = inv_N / (r.T @ CL).T
        if _counter % 10 == 0:
            err = np.nansum(np.abs(c / c_new - 1))
        c = c_new
        _counter += 1
    CL *= np.squeeze(c)
    CL = CL.T
    CL *= np.squeeze(r)
    CL = CL.T
    argmaxes = np.nanargmax(CL, 0)
    newL = torch.LongTensor(argmaxes)
    return newL
