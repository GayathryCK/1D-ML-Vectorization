# ----------------------------
# pptnet_entropy_regressor.py
#_____________________________
# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F

# PPT-Net dependencies
from libs.pointops.functions import pointops
from util import pt_util
from models import loupe as lp

# PPT-Net Backbone
class SA_Layer(nn.Module):
    def __init__(self, channels, gp):
        super().__init__()
        self.gp = gp
        assert channels % 4 == 0
        self.q_conv = nn.Conv1d(channels, channels, 1, bias=False, groups=gp)
        self.k_conv = nn.Conv1d(channels, channels, 1, bias=False, groups=gp)
        self.q_conv.weight = self.k_conv.weight
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        bs, ch, nums = x.size()
        x_q = self.q_conv(x).reshape(bs, self.gp, ch // self.gp, nums).permute(0, 1, 3, 2)
        x_k = self.k_conv(x).reshape(bs, self.gp, ch // self.gp, nums)
        x_v = self.v_conv(x)
        energy = torch.matmul(x_q, x_k)
        energy = torch.sum(energy, dim=1)
        attn = self.softmax(energy)
        attn = attn / (1e-9 + attn.sum(dim=1, keepdim=True))
        x_r = torch.matmul(x_v, attn)
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        return x + x_r


class _PointNet2SAModuleBase(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, xyz, features=None):
        new_features_list = []
        xyz_trans = xyz.transpose(1, 2).contiguous()
        center_idx = pointops.furthestsampling(xyz, self.npoint)
        new_xyz = pointops.gathering(xyz_trans, center_idx).transpose(1,2).contiguous() if self.npoint else None
        center_features = pointops.gathering(features, center_idx)
        for i in range(len(self.groupers)):
            new_features = self.groupers[i](xyz, new_xyz, features, center_features)
            new_features = self.mlps[i](new_features)
            new_features = F.max_pool2d(new_features, kernel_size=[1, new_features.size(3)]).squeeze(-1)
            new_features = self.sas[i](new_features)
            new_features_list.append(new_features)
        return new_xyz, torch.cat(new_features_list, dim=1)


class PointNet2SAModuleMSG(_PointNet2SAModuleBase):
    def __init__(self, *, npoint, radii, nsamples, mlps, gp, bn=True, use_xyz=True):
        super().__init__()
        self.npoint = npoint
        self.groupers, self.mlps, self.sas = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for i in range(len(radii)):
            radius, nsample = radii[i], nsamples[i]
            self.groupers.append(
                pointops.QueryAndGroup_Edge(radius, nsample, use_xyz=use_xyz) if npoint else pointops.GroupAll(use_xyz)
            )
            mlp_spec = mlps[i]
            if use_xyz: mlp_spec[0] += 3
            self.mlps.append(pt_util.SharedMLP(mlp_spec, bn=bn))
            self.sas.append(SA_Layer(mlp_spec[-1], gp))


class PointNet2SAModule(PointNet2SAModuleMSG):
    def __init__(self, *, mlp, npoint=None, radius=None, nsample=None, gp=None, bn=True, use_xyz=True):
        super().__init__(mlps=[mlp], npoint=npoint, radii=[radius], nsamples=[nsample], gp=gp, bn=bn, use_xyz=use_xyz)


class PointNet2FPModule(nn.Module):
    def __init__(self, *, mlp, bn=True): super().__init__(); self.mlp = pt_util.SharedMLP(mlp, bn=bn)
    def forward(self, unknown, known, unknow_feats, known_feats):
        if known is not None:
            dist, idx = pointops.nearestneighbor(unknown, known)
            dist_recip = 1.0 / (dist + 1e-8)
            weight = dist_recip / torch.sum(dist_recip, dim=2, keepdim=True)
            interpolated_feats = pointops.interpolation(known_feats, idx, weight)
        else:
            interpolated_feats = known_feats.expand(*known_feats.size()[0:2], unknown.size(1))
        new_features = torch.cat([interpolated_feats, unknow_feats], dim=1) if unknow_feats is not None else interpolated_feats
        return self.mlp(new_features.unsqueeze(-1)).squeeze(-1)


class PointNet2(nn.Module):
    def __init__(self, param=None):
        super().__init__()
        c, use_xyz, sap, knn, fs, gp = 3, True, param['SAMPLING'], param['KNN'], param['FEATURE_SIZE'], param['GROUP']
        self.SA_modules = nn.ModuleList([
            PointNet2SAModule(npoint=sap[0], nsample=knn[0], gp=gp, mlp=[c, 32, 32, 64], use_xyz=use_xyz),
            PointNet2SAModule(npoint=sap[1], nsample=knn[1], gp=gp, mlp=[64, 64, 64, 128], use_xyz=use_xyz),
            PointNet2SAModule(npoint=sap[2], nsample=knn[2], gp=gp, mlp=[128, 128, 128, 256], use_xyz=use_xyz),
            PointNet2SAModule(npoint=sap[3], nsample=knn[3], gp=gp, mlp=[256, 256, 256, 512], use_xyz=use_xyz)
        ])
        self.FP_modules = nn.ModuleList([
            PointNet2FPModule(mlp=[fs[1] + c, 256, 256, fs[0]]),
            PointNet2FPModule(mlp=[fs[2] + 64, 256, fs[1]]),
            PointNet2FPModule(mlp=[fs[3] + 128, 256, fs[2]]),
            PointNet2FPModule(mlp=[512 + 256, 256, fs[3]])
        ])
    def forward(self, pointcloud):
        l_xyz, l_features = [pointcloud], [pointcloud.transpose(1, 2).contiguous()]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz); l_features.append(li_features)
        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i])
        return l_features[3].unsqueeze(-1), l_features[2].unsqueeze(-1), l_features[1].unsqueeze(-1), l_features[0].unsqueeze(-1)


# ----------------------------
# Wrapper Network
class PPTNetVectorRegressor(nn.Module):
    def __init__(self, param=None, output_dim=3):
        super().__init__()
        self.backbone = PointNet2(param)
        self.aggregation = lp.SpatialPyramidNetVLAD(
            feature_size=[512, 256, 128, 64],
            cluster_size=param["CLUSTER_SIZE"],
            output_dim=param["OUTPUT_DIM"],
            gating=param["GATING"],
            add_batch_norm=True,
            final_output_dim=256,
            debug=False
        )
        self.regressor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        if x.dim() == 4 and x.size(1) == 1:
            x = x.squeeze(1)

        f0, f1, f2, f3 = self.backbone(x)
        feat = self.aggregation(f0, f1, f2, f3)
        out = self.regressor(feat)
        return out