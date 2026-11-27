"""Strict, serializable experiment recipes. Paths are relative to the config file."""
import hashlib
import json
import os
from pathlib import Path
from .modules import PoseConfig

REQUIRED = {'stage','model','bert','pose','contrastive','manifest','output','previous',
            'seed','expected_world_size','expected_global_batch','learning_rates','training'}

def load_config(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text())
    missing, unknown = REQUIRED-set(config), set(config)-REQUIRED
    if missing or unknown:
        raise ValueError(f'Config keys: missing={missing}, unknown={unknown}')
    if config['stage'] not in (1,2,3):
        raise ValueError('stage must be 1, 2 or 3')
    pose = PoseConfig(**config['pose'])
    if min(pose.width,pose.heads,pose.layers,pose.clip,pose.steps,pose.persons,pose.bottleneck,pose.binding_heads)<=0:
        raise ValueError('Pose dimensions must be positive')
    if pose.width%pose.heads or pose.bottleneck%pose.binding_heads:
        raise ValueError('Attention widths must be divisible by heads')
    from dataclasses import fields
    from transformers import TrainingArguments
    supported = {f.name for f in fields(TrainingArguments)}
    reserved = {'output_dir','seed','data_seed','remove_unused_columns','save_safetensors','report_to'}
    if set(config['training'])-supported or set(config['training']) & reserved:
        raise ValueError('Unsupported or reserved training argument')
    contrastive = config['contrastive']
    if set(contrastive) != {'projection_dim','temperature','max_text_length'} or min(contrastive.values())<=0:
        raise ValueError('Invalid contrastive settings')
    for key in ('manifest','output','previous'):
        if config[key] is not None:
            config[key] = str((path.parent/config[key]).resolve())
    if config['stage']>1 and not config['previous']:
        raise ValueError('Stage 2/3 requires previous-stage output')
    if config['training'].get('deepspeed'):
        config['training']['deepspeed'] = str((path.parent/config['training']['deepspeed']).resolve())
    stage = config['stage']
    expected_lrs = ({'pose','bert','heads'} if stage==1 else
                    {'alignment'} if stage==2 else {'pose','person','binding','llm'})
    if set(config['learning_rates']) != expected_lrs or any(v<=0 for v in config['learning_rates'].values()):
        raise ValueError(f'Expected positive learning rates for {expected_lrs}')
    return config


def validate_runtime(config):
    world = int(os.environ.get('WORLD_SIZE','1'))
    if world != config['expected_world_size']:
        raise ValueError(f"Expected {config['expected_world_size']} processes, got {world}; use torchrun or an explicitly revised config")
    training = config['training']
    batch = world*training['per_device_train_batch_size']*training['gradient_accumulation_steps']
    if batch != config['expected_global_batch']:
        raise ValueError('Effective global batch does not match expected_global_batch')
    if config['stage']==1:
        if world!=1 or training['gradient_accumulation_steps']!=1:
            raise ValueError('Stage 1 requires one process and no accumulation: negatives are local to a batch')
        if training['per_device_train_batch_size']<2 or not training.get('dataloader_drop_last'):
            raise ValueError('Stage 1 requires batch >= 2 and drop_last to avoid singleton InfoNCE')


def manifest_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_previous(config):
    previous = Path(config['previous'])
    meta = json.loads((previous/'iris_config.json').read_text())
    if meta['stage'] != config['stage']-1:
        raise ValueError('Previous checkpoint must come from the immediately preceding stage')
    if meta['pose'] != config['pose']:
        raise ValueError('Pose configuration mismatch across stages')
    if config['stage']==3 and meta['base_model'] != config['model']:
        raise ValueError('Qwen backbone mismatch across stages')
    return previous
