import argparse
import json
from pathlib import Path
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from .model import Iris, SPECIAL
from .modules import PoseConfig
from .data import Manifest

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--manifest',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda')
    p.add_argument('--max-new-tokens',type=int,default=128)
    args=p.parse_args()
    path=Path(args.checkpoint)
    config=json.loads((path/'iris_config.json').read_text())
    processor=AutoProcessor.from_pretrained(path/'processor')
    backbone=Qwen3VLForConditionalGeneration.from_pretrained(config['base_model'])
    backbone.resize_token_embeddings(len(processor.tokenizer))
    model=Iris(backbone,processor.tokenizer.convert_tokens_to_ids(SPECIAL),PoseConfig(**config['pose']))
    model.load_state_dict(torch.load(path/'iris.pt',map_location='cpu',weights_only=True))
    dtype=torch.bfloat16 if args.device.startswith('cuda') else torch.float32
    model.to(device=args.device,dtype=dtype).eval()
    dataset=Manifest(args.manifest,processor,inference=True,pose_config=model.cfg)
    with open(args.output,'w') as f:
        for i in range(len(dataset)):
            example={k:v.to(args.device) for k,v in dataset[i].items()}
            example.pop('labels',None)
            result=model.generate_greedy(max_new_tokens=args.max_new_tokens,**example)
            f.write(json.dumps({'id':dataset.rows[i].get('id',i),'prediction':processor.tokenizer.decode(result,skip_special_tokens=True)},ensure_ascii=False)+'\n')
            f.flush()

if __name__=='__main__':
    main()
