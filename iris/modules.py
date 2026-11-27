from dataclasses import dataclass
import torch
from torch import nn

# Table 7; overlapping shoulder/hip anchors are an explicit implementation choice.
PARTS = ((0,1,2,3,4,5,6,7,14,15), (6,8,10,12), (7,9,11,13),
         (14,16,18,20,22), (15,17,19,21,23))

@dataclass
class PoseConfig:
    width: int = 512  # not specified by paper
    heads: int = 8
    layers: int = 8
    clip: int = 20
    steps: int = 8
    persons: int = 3
    bottleneck: int = 1024
    binding_heads: int = 8
    delta: int = 100

class PoseEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.convs = nn.ModuleList([nn.Conv1d(len(p)*9, cfg.width, cfg.clip, stride=cfg.clip) for p in PARTS])
        self.time = nn.Parameter(torch.randn(cfg.steps, cfg.width)*0.02)
        self.person = nn.Parameter(torch.randn(cfg.persons, cfg.width)*0.02)
        self.part = nn.Parameter(torch.randn(5, cfg.width)*0.02)
        layer = nn.TransformerEncoderLayer(cfg.width, cfg.heads, cfg.width*4,
                                          dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, cfg.layers, enable_nested_tensor=False)

    def forward(self, pose):
        # B,T,P,24,9; missing frames zero-filled by preprocessing.
        b,t,p,j,c = pose.shape
        if j != 24 or c != 9 or t != self.cfg.steps*self.cfg.clip or not 1 <= p <= self.cfg.persons:
            raise ValueError("Expected [B,steps*clip,P<=3,24,9]")
        tokens = []
        for indices, conv in zip(PARTS, self.convs):
            x = pose[:,:,:,indices,:].permute(0,2,3,4,1).reshape(b*p,len(indices)*9,t)
            tokens.append(conv(x).transpose(1,2).reshape(b,p,self.cfg.steps,-1).transpose(1,2))
        x = torch.stack(tokens, dim=3)
        x = x + self.time[None,:,None,None] + self.person[None,None,:p,None] + self.part[None,None,None]
        valid = pose[...,8].bool().any(dim=(1,3))
        mask = valid[:,None,:,None].expand(b,self.cfg.steps,p,5).flatten(1)
        if not valid.any(-1).all():
            raise ValueError("Empty pose samples must use RGB-only fallback")
        out = self.encoder(x.flatten(1,3),src_key_padding_mask=~mask)
        return (out*mask[...,None]).reshape(b,self.cfg.steps,p,5,-1)

class BindingLayer(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ffn = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim,4*dim), nn.GELU(), nn.Linear(4*dim,dim))
    def forward(self, q, kv, allowed):
        # Empty boxes: use a dummy zero key to avoid all-masked softmax NaNs.
        nonempty = allowed.any(-1)
        allowed = allowed.clone()
        allowed[~nonempty,0] = True
        kv = kv * nonempty[:,None,None]
        q = q + self.attn(self.norm(q),kv,kv,key_padding_mask=~allowed,need_weights=False)[0]
        return (q + self.ffn(q)) * nonempty[:,None,None]

class SpatialBinding(nn.Module):
    def __init__(self, hidden, cfg):
        super().__init__()
        self.q = nn.Linear(hidden,cfg.bottleneck)
        self.kv = nn.Linear(hidden,cfg.bottleneck)
        self.layers = nn.ModuleList([BindingLayer(cfg.bottleneck,cfg.binding_heads) for _ in range(2)])
        self.out = nn.Linear(cfg.bottleneck, hidden)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
    def forward(self, pose, visual, allowed):
        # pose [B,T,P,5,D], visual [B,T,HW,D], mask [B,T,P,HW]
        b,t,p,n,d = pose.shape
        q = self.q(pose).reshape(b*t*p,n,-1)
        kv = self.kv(visual)[:,:,None].expand(-1,-1,p,-1,-1).reshape(b*t*p,visual.shape[2],-1)
        mask = allowed.reshape(b*t*p,-1)
        for layer in self.layers:
            q = layer(q,kv,mask)
        out = self.out(q) * mask.any(-1)[:,None,None]
        return out.reshape(b,t,p,n,d)

def region_mask(pose, height, width, clip):
    """Union of visible joints within each dense temporal window; patch-center inclusion."""
    b,tp,p,_,_ = pose.shape
    xy = pose[...,:2].reshape(b,tp//clip,clip,p,24,2)
    valid = pose[...,8].bool().reshape(b,tp//clip,clip,p,24)
    low = xy.masked_fill(~valid[...,None],float('inf')).amin(dim=(2,4))
    high = xy.masked_fill(~valid[...,None],float('-inf')).amax(dim=(2,4))
    y,x = torch.meshgrid((torch.arange(height,device=pose.device)+.5)/height,
                         (torch.arange(width,device=pose.device)+.5)/width,indexing='ij')
    centers = torch.stack((x,y),-1).reshape(-1,2)
    return ((centers >= low[...,None,:]) & (centers <= high[...,None,:])).all(-1)
