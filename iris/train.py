"""Three-stage training entry point. Never runs on import."""
import argparse
import json
from dataclasses import asdict
from pathlib import Path
import torch
from transformers import (AutoModel, AutoTokenizer, AutoProcessor, Qwen3VLForConditionalGeneration,
                          Trainer, TrainingArguments, set_seed)
from .model import Iris, SPECIAL
from .modules import PoseConfig
from .contrastive import ContrastivePose
from .data import Manifest, collate


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage',type=int,choices=[1,2,3],required=True)
    p.add_argument('--manifest',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--model',default='Qwen/Qwen3-VL-8B-Instruct')
    p.add_argument('--bert',default='bert-base-uncased')
    p.add_argument('--previous',help='Stage 1 output for stage 2; stage 2 output for stage 3')
    p.add_argument('--deepspeed')
    p.add_argument('--batch-size',type=int)
    p.add_argument('--gradient-accumulation',type=int,default=1)
    p.add_argument('--bf16',action='store_true')
    p.add_argument('--resume')
    args = p.parse_args()
    if args.stage>1 and not args.previous:
        p.error('--previous is required for progressive training')
    set_seed(42)
    cfg = PoseConfig()
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    if args.stage==1:
        tokenizer = AutoTokenizer.from_pretrained(args.bert)
        model = ContrastivePose(cfg,AutoModel.from_pretrained(args.bert))
        def data_collator(rows):
            # Different person counts are forwarded as a single padded batch.
            poses = torch.zeros(len(rows),160,3,24,9)
            for i,row in enumerate(rows):
                poses[i,:,:row['pose'].shape[1]] = row['pose']
            return dict(pose=poses,**tokenizer([r['caption'] for r in rows],padding=True,truncation=True,return_tensors='pt')) | {}
        # BERT token_type_ids are unused by our forward contract.
        base_collator = data_collator
        def data_collator(rows):
            return {k:v for k,v in base_collator(rows).items() if k!='token_type_ids'}
        dataset = Manifest(args.manifest,stage=1)
        groups = model.optimizer_groups()
        processor = None
    else:
        processor = AutoProcessor.from_pretrained(args.model)
        processor.tokenizer.add_special_tokens({'additional_special_tokens':SPECIAL})
        backbone = Qwen3VLForConditionalGeneration.from_pretrained(args.model,torch_dtype=dtype)
        backbone.resize_token_embeddings(len(processor.tokenizer))
        model = Iris(backbone,processor.tokenizer.convert_tokens_to_ids(SPECIAL),cfg)
        previous = Path(args.previous)
        if args.stage==2:
            model.pose_encoder.load_state_dict(torch.load(previous/'pose_encoder.pt',map_location='cpu',weights_only=True))
        else:
            model.load_state_dict(torch.load(previous/'iris.pt',map_location='cpu',weights_only=True))
        model.configure_stage(args.stage)
        groups = model.optimizer_groups(args.stage)
        dataset = Manifest(args.manifest,processor,args.stage)
        data_collator = collate
    optimizer = torch.optim.AdamW(groups,weight_decay=.01)
    training = TrainingArguments(output_dir=args.output,num_train_epochs={1:150,2:2,3:1}[args.stage],
        per_device_train_batch_size=args.batch_size or (512 if args.stage==1 else 12),
        gradient_accumulation_steps=args.gradient_accumulation,learning_rate=1e-5,lr_scheduler_type='cosine',
        warmup_ratio=.05,weight_decay=.01,max_grad_norm=1.,bf16=args.bf16,
        deepspeed=args.deepspeed,remove_unused_columns=False,save_safetensors=False,
        report_to='none',save_strategy='epoch',dataloader_num_workers=0,ddp_find_unused_parameters=True)
    trainer = Trainer(model=model,args=training,train_dataset=dataset,data_collator=data_collator,optimizers=(optimizer,None))
    trainer.train(resume_from_checkpoint=args.resume)
    # Accelerate gathers full weights for ZeRO; all ranks must participate.
    state = trainer.accelerator.get_state_dict(trainer.model_wrapped)
    if trainer.is_world_process_zero():
        out = Path(args.output)
        out.mkdir(parents=True,exist_ok=True)
        torch.save({k.removeprefix('pose_encoder.'):v.cpu() for k,v in state.items() if k.startswith('pose_encoder.')},out/'pose_encoder.pt')
        if args.stage>1:
            torch.save({k:v.cpu() for k,v in state.items()},out/'iris.pt')
            processor.save_pretrained(out/'processor')
        (out/'iris_config.json').write_text(json.dumps({'pose':asdict(cfg),'base_model':args.model,'stage':args.stage},indent=2))

if __name__=='__main__':
    main()
