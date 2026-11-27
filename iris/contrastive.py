import torch
from torch import nn
from torch.nn import functional as F
from .modules import PoseEncoder

class ContrastivePose(nn.Module):
    def __init__(self,cfg,text_encoder,projection_dim=256,temperature=.07):
        super().__init__()
        self.pose_encoder = PoseEncoder(cfg)
        self.text_encoder = text_encoder
        self.pose_head = nn.Linear(cfg.width,projection_dim)
        self.text_head = nn.Linear(text_encoder.config.hidden_size,projection_dim)
        self.logit_scale = nn.Parameter(torch.tensor(1/temperature).log())
    def forward(self,pose,input_ids,attention_mask):
        encoded = self.pose_encoder(pose)
        count = pose[...,8].bool().any(dim=(1,3)).sum(-1).clamp_min(1)
        p = encoded.sum(dim=(1,2,3))/(count[:,None]*encoded.shape[1]*5)
        t = self.text_encoder(input_ids=input_ids,attention_mask=attention_mask).last_hidden_state[:,0]
        p,t = F.normalize(self.pose_head(p),dim=-1),F.normalize(self.text_head(t),dim=-1)
        logits = self.logit_scale.exp().clamp(max=100)*p@t.T
        labels = torch.arange(len(p),device=p.device)
        return {'loss':(F.cross_entropy(logits,labels)+F.cross_entropy(logits.T,labels))/2}
    def optimizer_groups(self, rates=None):
        rates = rates or {"pose":1e-4,"bert":1e-5,"heads":3e-4}
        return [{'params':self.pose_encoder.parameters(),'lr':rates['pose']},
                {'params':self.text_encoder.parameters(),'lr':rates['bert']},
                {'params':list(self.pose_head.parameters())+list(self.text_head.parameters())+[self.logit_scale],'lr':rates['heads']}]
