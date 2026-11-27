import copy
import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pytest
import torch
from transformers import BertConfig, BertModel, BertTokenizerFast
from iris import train
from iris.experiment import load_config, validate_runtime, validate_previous
from test_iris import fixture


def test_paper_recipes(monkeypatch):
    for stage in (1,2,3):
        c=load_config(f'configs/iris/stage{stage}.json')
        monkeypatch.setenv('WORLD_SIZE',str(c['expected_world_size']))
        validate_runtime(c)
        assert c['training']['num_train_epochs']=={1:150,2:2,3:1}[stage]
        assert c['training']['warmup_ratio']==.05
        assert c['pose']['layers']==8
    monkeypatch.setenv('WORLD_SIZE','1')
    with pytest.raises(ValueError,match='processes'):
        validate_runtime(load_config('configs/iris/stage2.json'))


def test_three_stage_training_and_bert_resume(tmp_path,monkeypatch):
    """Real HF Trainer/AdamW/scheduler/checkpoints, offline tiny BERT and Qwen.

    Stage 1 reads real JSONL/NPY through Manifest. Stages 2/3 use already
    tokenized synthetic video samples, replacing only processor/data loading.
    """
    monkeypatch.setenv('WORLD_SIZE','1')
    torch.set_num_threads(2)
    bert_dir=tmp_path/'bert'; bert_dir.mkdir()
    (bert_dir/'vocab.txt').write_text('[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nperson\nwalks\nruns\n')
    tokenizer=BertTokenizerFast(vocab_file=str(bert_dir/'vocab.txt'))
    tokenizer.save_pretrained(bert_dir)
    bert=BertModel(BertConfig(vocab_size=8,hidden_size=16,num_hidden_layers=1,num_attention_heads=2,intermediate_size=32))
    before=bert.embeddings.word_embeddings.weight.detach().clone()
    bert.save_pretrained(bert_dir)
    iris,example=fixture()
    qwen_dir=tmp_path/'qwen'; iris.backbone.save_pretrained(qwen_dir)
    manifest=tmp_path/'stage1.jsonl'
    rows=[]
    for i in range(4):
        pose=example['pose'][0].numpy().copy(); pose[...,:8]+=i*.1
        np.save(tmp_path/f'{i}.npy',pose)
        rows.append(json.dumps({'pose':f'{i}.npy','caption':'person walks' if i%2 else 'person runs'}))
    manifest.write_text('\n'.join(rows))
    c=load_config('configs/iris/stage1.json')
    c.update(bert=str(bert_dir),model=str(qwen_dir),pose=asdict(iris.cfg),manifest=str(manifest),
             output=str(tmp_path/'stage1'),expected_global_batch=2)
    c['contrastive']['projection_dim']=8
    c['training'].update(bf16=False,use_cpu=True,max_steps=2,per_device_train_batch_size=2,
        save_strategy='steps',save_steps=1,logging_steps=1,disable_tqdm=True)
    train.set_seed(c["seed"])
    initial_model,_,_,_,_=train.build(c)
    initial_pose=initial_model.pose_encoder.convs[0].weight.detach().clone()
    trainer=train.run(c)
    assert trainer.state.global_step==2
    assert not torch.equal(before,trainer.model.text_encoder.embeddings.word_embeddings.weight)
    assert not torch.equal(initial_pose,trainer.model.pose_encoder.convs[0].weight)
    out=Path(c['output'])
    checkpoint=out/'checkpoint-1'
    for filename in ('optimizer.pt','scheduler.pt','rng_state.pth','trainer_state.json','pytorch_model.bin'):
        assert (checkpoint/filename).exists()
    changed=copy.deepcopy(c); changed['learning_rates']['bert']=.001
    with pytest.raises(ValueError,match='Resume recipe'):
        train.run(changed,str(checkpoint))
    state=torch.load(out/'contrastive.pt',weights_only=True)
    assert any(k.startswith('text_encoder.') for k in state)
    # Resume exactly the same recipe from step 1; final weights must match.
    resumed=copy.deepcopy(c); resumed['output']=str(tmp_path/'resumed')
    resumed_trainer=train.run(resumed,str(checkpoint))
    assert resumed_trainer.state.global_step==2
    for key,value in state.items():
        assert torch.allclose(value,resumed_trainer.model.state_dict()[key],atol=1e-7),key

    class FakeTokenizer:
        def __len__(self): return 64
        def add_special_tokens(self,*a,**k): return 0
        def convert_tokens_to_ids(self,tokens): return [55,56,57]
    class FakeProcessor:
        tokenizer=FakeTokenizer()
        def save_pretrained(self,path): Path(path).mkdir(parents=True,exist_ok=True)
    monkeypatch.setattr(train.AutoProcessor,'from_pretrained',lambda *a,**kw:FakeProcessor())
    class Prepared:
        def __len__(self): return 4
        def __getitem__(self,index): return {k:v.clone() for k,v in example.items()}
    monkeypatch.setattr(train,'Manifest',lambda *a,**kw:Prepared())
    previous=str(out)
    for stage in (2,3):
        next_c=copy.deepcopy(c)
        recipe=load_config(f'configs/iris/stage{stage}.json')
        next_c.update(stage=stage,previous=previous,output=str(tmp_path/f'stage{stage}'),learning_rates=recipe['learning_rates'])
        model,_,_,_,groups=train.build(next_c)
        if stage==2:
            saved_pose=torch.load(out/'pose_encoder.pt',weights_only=True)
            for key,value in saved_pose.items():
                assert torch.equal(value,model.pose_encoder.state_dict()[key])
        else:
            prior=torch.load(Path(previous)/'iris.pt',weights_only=True)
            for key,value in prior.items(): assert torch.equal(value,model.state_dict()[key])
        initial={k:v.clone() for k,v in model.state_dict().items()}
        trained=train.run(next_c)
        final=trained.model.state_dict()
        assert not torch.equal(initial['binding.out.weight'],final['binding.out.weight'])
        for key in initial:
            if key.startswith('backbone.model.visual.') or (stage==2 and key.startswith('pose_encoder.')):
                assert torch.equal(initial[key],final[key]),key
        if stage==3:
            assert not torch.equal(initial['pose_encoder.convs.0.weight'],final['pose_encoder.convs.0.weight'])
            assert not torch.equal(initial['backbone.lm_head.weight'],final['backbone.lm_head.weight'])
        assert (Path(next_c['output'])/'iris.pt').exists()
        previous=next_c['output']
    bad=copy.deepcopy(next_c); bad['pose']['width']=32
    with pytest.raises(ValueError,match='configuration mismatch'): validate_previous(bad)


def test_config_learning_rates_reach_optimizer_groups():
    from iris.contrastive import ContrastivePose
    model,_=fixture()
    rates={'pose':.001,'person':.002,'binding':.003,'llm':.004}
    model.configure_stage(3)
    groups=model.optimizer_groups(3,rates,weight_decay=.02)
    by_id={id(p):g for g in groups for p in g['params']}
    assert by_id[id(model.pose_encoder.person)]['lr']==rates['person']
    assert by_id[id(model.pose_encoder.convs[0].weight)]['lr']==rates['pose']
    assert by_id[id(model.binding.out.weight)]['lr']==rates['binding']
    assert by_id[id(model.backbone.lm_head.weight)]['lr']==rates['llm']
    assert all(g['weight_decay']==.02 for g in groups)
    bert=BertModel(BertConfig(vocab_size=8,hidden_size=16,num_hidden_layers=1,num_attention_heads=2,intermediate_size=32))
    contrastive=ContrastivePose(model.cfg,bert)
    assert [g['lr'] for g in contrastive.optimizer_groups({'pose':.1,'bert':.2,'heads':.3})]==[.1,.2,.3]
