import torch
from torch import nn
from torch.nn import functional as F
from transformers.modeling_outputs import CausalLMOutput
from .modules import PoseConfig, PoseEncoder, SpatialBinding, region_mask

SPECIAL = ["<|pose_start|>", "<|pose_end|>", "<|pose_token|>"]

class Iris(nn.Module):
    """Qwen3-VL adapter, preserving its native vision encoder and DeepStack features.

    Each example contains one video. Variable-length examples are forwarded separately
    within a microbatch. This reference implementation prioritizes explicit alignment.
    """
    def __init__(self, backbone, token_ids, cfg=None):
        super().__init__()
        self.backbone, self.token_ids = backbone, token_ids
        self.cfg = cfg or PoseConfig()
        d = backbone.config.text_config.hidden_size
        self.pose_encoder = PoseEncoder(self.cfg)
        self.projector = nn.Sequential(nn.Linear(self.cfg.width,d),nn.GELU(),nn.Linear(d,d))
        self.binding = SpatialBinding(d,self.cfg)
        self._embedding_hook = None

    def forward(self, examples=None, **example):
        if examples is not None:
            return {"loss": torch.stack([self.forward(**x).loss for x in examples]).mean()}
        pose = example.pop('pose',None)
        if pose is None:
            return self.backbone(**example)
        ids = example['input_ids']
        if ids.shape[0] != 1 or example.get('pixel_values') is not None:
            raise ValueError('Pose path requires one video per example and no image inputs')
        if 'past_key_values' in example:
            raise ValueError('Use generate_greedy; cached pose decoding is not implemented')
        grid = example['video_grid_thw']
        if len(grid) != 1:
            raise ValueError('Exactly one video required')
        t,h,w = grid[0].tolist()
        merge = self.backbone.config.vision_config.spatial_merge_size
        h,w = h//merge,w//merge
        if t != self.cfg.steps:
            raise ValueError(f'Expected {self.cfg.steps} merged video times, got {t}')
        slots = ids[0] == self.token_ids[2]
        if slots.sum().item() != t*pose.shape[2]*10:
            raise ValueError('Pose placeholder count must equal T*P*10')
        emb = self.backbone.get_input_embeddings()(ids)
        visual, deep = self.backbone.model.get_video_features(example['pixel_values_videos'],grid)
        visual = torch.cat(visual).to(emb)
        vmask = ids == self.backbone.config.video_token_id
        if vmask.sum() != t*h*w:
            raise ValueError('Video token/grid mismatch')
        emb = emb.masked_scatter(vmask[...,None].expand_as(emb),visual)
        pose = pose.to(device=emb.device,dtype=next(self.pose_encoder.parameters()).dtype)
        xp = self.projector(self.pose_encoder(pose))
        xpv = self.binding(xp,visual.reshape(1,t,h*w,-1),region_mask(pose,h,w,self.cfg.clip))
        emb = emb.clone()
        emb[0,slots] = torch.cat((xp,xpv),dim=3).reshape(-1,emb.shape[-1]).to(emb)
        # Compute native Qwen positions without pose block; retain timestamp handling.
        block = torch.zeros_like(ids[0],dtype=torch.bool)
        for tok in self.token_ids:
            block |= ids[0] == tok
        base_ids = ids[:,~block]
        positions,_ = self.backbone.model.get_rope_index(base_ids,video_grid_thw=grid,
                                                       attention_mask=torch.ones_like(base_ids))
        pos = torch.zeros(3,1,ids.shape[1],dtype=torch.long,device=ids.device)
        pos[:,:,~block] = positions
        vt = positions[0,0,base_ids[0]==self.backbone.config.video_token_id].reshape(t,h*w)[:,0]
        offset = positions[:,0,base_ids[0]==self.backbone.config.video_token_id].amin()
        p = pose.shape[2]
        coords = torch.stack((vt[:,None,None].expand(t,p,10),
            (torch.arange(p,device=ids.device)+self.cfg.delta+offset)[None,:,None].expand(t,p,10),
            (torch.arange(10,device=ids.device)+self.cfg.delta+offset)[None,None,:].expand(t,p,10)))
        pos[:,0,slots] = coords.reshape(3,-1)
        start = (ids[0]==self.token_ids[0]).nonzero().flatten()
        end = (ids[0]==self.token_ids[1]).nonzero().flatten()
        if len(start)!=1 or len(end)!=1 or not start.item()<end.item():
            raise ValueError('Expected exactly one ordered pose block')
        pos[:,:,start] = pos[:,:,:start.item()].max()+1
        pos[:,:,end] = pos[:,:,:end.item()].max()+1
        # Text after pose begins beyond every preceding coordinate.
        n = ids.shape[1]-end.item()-1
        pos[:,:,end.item()+1:] = (torch.arange(n,device=ids.device)+pos[:,:,end].max()+1)[None,None]
        out = self.backbone.model.language_model(inputs_embeds=emb,position_ids=pos,
            attention_mask=example.get('attention_mask',torch.ones_like(ids)),
            visual_pos_masks=vmask,deepstack_visual_embeds=deep,use_cache=False)
        logits = self.backbone.lm_head(out.last_hidden_state)
        labels = example.get('labels')
        loss = None if labels is None else F.cross_entropy(logits[:,:-1].float().reshape(-1,logits.shape[-1]),
                                                           labels[:,1:].reshape(-1),ignore_index=-100)
        return CausalLMOutput(loss=loss,logits=logits)

    def configure_stage(self, stage):
        if stage not in (2,3):
            raise ValueError('Iris supports stage 2 or 3; stage 1 uses ContrastivePose')
        if self._embedding_hook is not None:
            self._embedding_hook.remove()
            self._embedding_hook = None
        self.requires_grad_(stage==3)
        self.backbone.visual.requires_grad_(False)
        self.projector.requires_grad_(True)
        self.binding.requires_grad_(True)
        if stage == 2:
            weight = self.backbone.get_input_embeddings().weight
            weight.requires_grad_(True)
            def mask_grad(grad):
                mask = torch.zeros(grad.shape[0],device=grad.device,dtype=grad.dtype)
                mask[self.token_ids] = 1
                return grad * mask[:,None]
            self._embedding_hook = weight.register_hook(mask_grad)

    def optimizer_groups(self, stage):
        groups = {}
        embedding = self.backbone.get_input_embeddings().weight
        for name,p in self.named_parameters():
            if not p.requires_grad:
                continue
            lr = 2e-4 if stage==2 else (3e-5 if name=='pose_encoder.person' else 2e-5 if name.startswith('pose_encoder') else
                 3e-5 if name.startswith(('binding','projector')) else 1e-5)
            # No AdamW decay on frozen vocabulary rows, even with zero gradient.
            decay = 0.0 if (stage==2 and p is embedding) else .01
            groups.setdefault((lr,decay),[]).append(p)
        return [{'params':v,'lr':k[0],'weight_decay':k[1]} for k,v in groups.items()]

    @torch.no_grad()
    def generate_greedy(self, max_new_tokens=128, **example):
        """Reference uncached decoding; no repeated model loading or training."""
        self.eval()
        example = dict(example)
        example.pop('labels',None)
        generated = []
        eos = self.backbone.generation_config.eos_token_id
        eos = eos if isinstance(eos,list) else [eos]
        for _ in range(max_new_tokens):
            token = self(**example).logits[:,-1].argmax(-1,keepdim=True)
            generated.append(token.item())
            if token.item() in eos:
                break
            example['input_ids'] = torch.cat((example['input_ids'],token),dim=1)
            example['attention_mask'] = torch.ones_like(example['input_ids'])
        return generated
