import torch
import torch.nn as nn
import torch.nn.functional as F 
from timm.models.layers import trunc_normal_



def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)


class GroupSpatialAttention(nn.Module):
    def __init__(self, channels, groups=1, kernel_size=7):
        super().__init__()
        self.groups = groups 
        self.group_size = channels // groups
        self.normalization = nn.InstanceNorm2d(channels)
        self.conv = nn.Conv2d(2*self.groups, self.groups, kernel_size, padding=kernel_size//2, bias=False, groups=self.groups)
        self.sigmoid = nn.Sigmoid()

    def calc_attn(self, x):
        b, c, h, w = x.shape
        norm_x = self.normalization(x).view(b*self.groups, self.group_size, h, w)
        avg_out = torch.mean(norm_x, dim=1, keepdim=True)
        max_out, _ = torch.max(norm_x, dim=1, keepdim=True)
        pool_x = torch.cat((avg_out, max_out), dim=1)
        pool_x = pool_x.view(b, 2 * self.groups, h, w)
        attn = self.sigmoid(self.conv(pool_x))
        attn = attn.repeat_interleave(self.group_size, dim=1) 
        return attn
    
    def forward(self, x):
        attn = self.calc_attn(x)
        out = x * attn
        return out


class LabelCorrelationHead(nn.Module):
    def __init__(self, num_features, num_classes, projection_dim=0, projection_depth=1, attn_dim=1):
        super().__init__()
        self.num_classes = num_classes
        self.dim = attn_dim

        if projection_dim == 0: 
            projection_dim = num_features
        
        projection_layers = []
        for i in range(projection_depth):
            if i == 0:
                projection_layers.append(nn.Linear(num_features, projection_dim))
            else:
                projection_layers.append(nn.Linear(projection_dim, projection_dim))
            projection_layers.append(nn.BatchNorm1d(projection_dim))
            projection_layers.append(nn.ReLU(True))
        self.projection = nn.Sequential(*projection_layers)

        self.qk = nn.Linear(projection_dim, num_classes * 2 * attn_dim, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.gamma = nn.Parameter(torch.zeros(num_classes), requires_grad=True)

        self.fc = nn.Linear(num_features, num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        
    def calc_correlated_logits(self, features, logits):
        b, c = features.shape
        projected_features = self.projection(features) # (b, d)
        q, k = self.qk(projected_features).chunk(2, dim=-1) # q, k : (b, c*d)
        q, k = q.view(b, self.num_classes, self.dim), k.view(b, self.num_classes, self.dim) # (b, c, d) * (b, c, d)
        attn = torch.bmm(q, k.transpose(1, 2)) # (b, c, c)

        attn = attn - torch.diagonal(attn, dim1=-2, dim2=-1).unsqueeze(-1) # remove diagonal values
        attn = attn / attn.abs().max(dim=-1, keepdim=True)[0] # normalize

        logits = self.sigmoid(logits)
        out = torch.bmm(attn, logits.unsqueeze(-1)).squeeze(-1)
        return out

    def forward(self, features):
        logits = self.fc(features)
        logits = logits + self.gamma * self.calc_correlated_logits(features, logits)
        return logits
