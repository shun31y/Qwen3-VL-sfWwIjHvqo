"""Config-driven three-stage training; --validate-only never loads model weights."""
import argparse
import json
from dataclasses import asdict
from pathlib import Path
import torch
from transformers import (AutoModel, AutoTokenizer, AutoProcessor, Qwen3VLForConditionalGeneration,
                          Trainer, TrainerCallback, TrainingArguments, set_seed)
from .model import Iris, SPECIAL
from .modules import PoseConfig
from .contrastive import ContrastivePose
from .data import Manifest, collate
from .experiment import load_config, validate_runtime, validate_previous, manifest_digest

class CaptionCollator:
    def __init__(self,tokenizer,cfg,max_length):
        self.tokenizer,self.cfg,self.max_length = tokenizer,cfg,max_length
    def __call__(self,rows):
        poses = torch.zeros(len(rows),self.cfg.steps*self.cfg.clip,self.cfg.persons,24,9)
        for i,row in enumerate(rows):
            poses[i,:,:row['pose'].shape[1]] = row['pose']
        text = self.tokenizer([r['caption'] for r in rows],padding=True,truncation=True,
                              max_length=self.max_length,return_tensors='pt')
        return {'pose':poses, 'input_ids':text['input_ids'],'attention_mask':text['attention_mask']}


def build(config):
    stage = config['stage']
    cfg = PoseConfig(**config['pose'])
    dtype = torch.bfloat16 if config['training'].get('bf16',False) else torch.float32
    if stage==1:
        processor = AutoTokenizer.from_pretrained(config['bert'])
        text = AutoModel.from_pretrained(config['bert'])
        if text.config.model_type != 'bert':
            raise ValueError('Stage 1 requires a BERT text encoder')
        model = ContrastivePose(cfg,text,config['contrastive']['projection_dim'],config['contrastive']['temperature'])
        data_collator = CaptionCollator(processor,cfg,config['contrastive']['max_text_length'])
        groups = model.optimizer_groups(config['learning_rates'])
    else:
        previous = validate_previous(config)
        processor = AutoProcessor.from_pretrained(previous/'processor' if stage==3 else config['model'])
        if stage==2:
            processor.tokenizer.add_special_tokens({'additional_special_tokens':SPECIAL})
        backbone = Qwen3VLForConditionalGeneration.from_pretrained(config['model'],torch_dtype=dtype)
        backbone.resize_token_embeddings(len(processor.tokenizer))
        model = Iris(backbone,processor.tokenizer.convert_tokens_to_ids(SPECIAL),cfg)
        if stage==2:
            model.pose_encoder.load_state_dict(torch.load(previous/'pose_encoder.pt',map_location='cpu',weights_only=True))
        else:
            meta = json.loads((previous/'iris_config.json').read_text())
            if meta['token_ids'] != model.token_ids:
                raise ValueError('Special-token IDs changed across stages')
            model.load_state_dict(torch.load(previous/'iris.pt',map_location='cpu',weights_only=True))
        model.configure_stage(stage)
        groups = model.optimizer_groups(stage,config['learning_rates'],config['training']['weight_decay'])
        data_collator = collate
    dataset = Manifest(config['manifest'],processor,stage,pose_config=cfg)
    if stage==1 and len(dataset)<config['training']['per_device_train_batch_size']:
        raise ValueError('Stage 1 dataset is smaller than its contrastive batch')
    return model,processor,dataset,data_collator,groups


class RecipeCheckpoint(TrainerCallback):
    def __init__(self, config, digest):
        self.record = {'config': config, 'manifest_sha256': digest}
    def on_save(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            path = Path(args.output_dir)/f'checkpoint-{state.global_step}'/'experiment.json'
            path.write_text(json.dumps(self.record,indent=2))


def validate_resume(config, resume, digest):
    saved = json.loads((Path(resume)/'experiment.json').read_text())
    current = {k:v for k,v in config.items() if k!='output'}
    previous = {k:v for k,v in saved['config'].items() if k!='output'}
    if current != previous or saved['manifest_sha256'] != digest:
        raise ValueError('Resume recipe or manifest changed; resume requires the same experiment')


def run(config,resume=None):
    validate_runtime(config)
    digest = manifest_digest(config['manifest'])
    if resume:
        validate_resume(config,resume,digest)
    set_seed(config['seed'])
    model,processor,dataset,data_collator,groups = build(config)
    training = TrainingArguments(output_dir=config['output'],seed=config['seed'],data_seed=config['seed'],
        remove_unused_columns=False,save_safetensors=False,report_to='none',**config['training'])
    optimizer = torch.optim.AdamW(groups,weight_decay=training.weight_decay,
                                  betas=(training.adam_beta1,training.adam_beta2),eps=training.adam_epsilon)
    trainer = Trainer(model=model,args=training,train_dataset=dataset,data_collator=data_collator,optimizers=(optimizer,None),
                      callbacks=[RecipeCheckpoint(config,digest)])
    out = Path(config['output'])
    if trainer.is_world_process_zero():
        out.mkdir(parents=True,exist_ok=True)
        (out/'resolved_config.json').write_text(json.dumps(config,indent=2))
    trainer.train(resume_from_checkpoint=resume)
    state = trainer.accelerator.get_state_dict(trainer.model_wrapped)
    if trainer.is_world_process_zero():
        state = {k:v.cpu() for k,v in state.items()}
        torch.save({k.removeprefix('pose_encoder.'):v for k,v in state.items() if k.startswith('pose_encoder.')},out/'pose_encoder.pt')
        torch.save(state,out/('contrastive.pt' if config['stage']==1 else 'iris.pt'))
        processor.save_pretrained(out/('tokenizer' if config['stage']==1 else 'processor'))
        meta = {'pose':asdict(model.pose_encoder.cfg),'base_model':config['model'],'bert':config['bert'],
                'stage':config['stage'],'manifest_sha256':digest,
                'token_ids':None if config['stage']==1 else model.token_ids}
        (out/'iris_config.json').write_text(json.dumps(meta,indent=2))
        trainer.save_state()
    return trainer


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config',required=True)
    p.add_argument('--resume',help='Trainer checkpoint directory, including optimizer/scheduler/RNG states')
    p.add_argument('--validate-only',action='store_true',help='Validate schema and print resolved recipe without loading weights')
    args = p.parse_args()
    config = load_config(args.config)
    if args.validate_only:
        print(json.dumps(config,indent=2))
        return
    run(config,args.resume)

if __name__=='__main__':
    main()
