import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from .model import SPECIAL
from .modules import PoseConfig

class Manifest(Dataset):
    def __init__(self,path,processor=None,stage=3,inference=False,pose_config=None):
        self.cfg = pose_config or PoseConfig()
        self.inference = inference
        self.path = Path(path)
        self.rows = [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]
        self.processor,self.stage = processor,stage
        if not self.rows:
            raise ValueError('Empty manifest')
    def __len__(self):
        return len(self.rows)
    def resolve(self,path):
        return str((self.path.parent/path).resolve())
    def __getitem__(self,index):
        row = self.rows[index]
        pose = None
        if row.get('pose'):
            pose = torch.from_numpy(np.load(self.resolve(row['pose']),allow_pickle=False)).float()
            if pose.ndim!=4 or pose.shape[0]!=self.cfg.steps*self.cfg.clip or pose.shape[-2:]!=(24,9) or pose.shape[1]>self.cfg.persons or not torch.isfinite(pose).all():
                raise ValueError('Expected finite pose [160,P<=3,24,9]')
            if pose.shape[1]==0 or not pose[...,8].any():
                pose = None
        if self.stage==1:
            if pose is None:
                raise ValueError('Stage 1 requires valid poses')
            return {'pose':pose,'caption':row['caption']}
        if self.stage==2 and pose is None:
            raise ValueError('Stage 2 requires pose-paired data')
        content = [{'type':'video','video':self.resolve(row['video']),'nframes':self.cfg.steps*2},
                   {'type':'text','text':row['question']}]
        messages = [{'role':'user','content':content}]
        from qwen_vl_utils import process_vision_info
        # Pose block precedes question, after visual content.
        if pose is not None:
            block = SPECIAL[0]+SPECIAL[2]*(self.cfg.steps*pose.shape[1]*10)+SPECIAL[1]
            content.insert(1,{'type':'text','text':block})
        prefix = self.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        images,videos,kwargs = process_vision_info(messages,image_patch_size=16,return_video_kwargs=True,return_video_metadata=True)
        video_data,metadata = zip(*videos)
        encoded = self.processor(text=[prefix],images=images,videos=list(video_data),
            video_metadata=list(metadata),return_tensors='pt',**kwargs)
        # Append assistant answer with native end-of-turn token; never supervise prompt.
        answer = (torch.empty((1,0),dtype=torch.long) if self.inference else
                  self.processor.tokenizer(row['answer']+'<|im_end|>',add_special_tokens=False,return_tensors='pt')['input_ids'])
        labels = torch.cat((torch.full_like(encoded['input_ids'],-100),answer),dim=1)
        encoded['input_ids'] = torch.cat((encoded['input_ids'],answer),dim=1)
        encoded['attention_mask'] = torch.ones_like(encoded['input_ids'])
        encoded['labels'] = labels
        if pose is not None:
            encoded['pose'] = pose.unsqueeze(0)
        # Processor metadata is consumed during video tokenization, not model forward.
        return {k:v for k,v in encoded.items() if k in {'input_ids','attention_mask','labels','pixel_values_videos','video_grid_thw','pose'}}


def collate(examples):
    return {'examples':examples}
