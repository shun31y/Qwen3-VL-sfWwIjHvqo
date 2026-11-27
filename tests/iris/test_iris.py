import numpy as np
import torch
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration, BertConfig, BertModel
from iris.modules import PoseConfig, PoseEncoder, SpatialBinding, region_mask
from iris.model import Iris
from iris.pose import clean, descriptor
from iris.contrastive import ContrastivePose

torch.set_num_threads(2)

def fixture():
    config = Qwen3VLConfig(text_config=dict(vocab_size=64,hidden_size=32,intermediate_size=64,
        num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=8,
        rope_scaling={'rope_type':'default','mrope_section':[1,1,2]}),
        vision_config=dict(depth=2,hidden_size=32,intermediate_size=64,num_heads=4,
            patch_size=2,spatial_merge_size=2,temporal_patch_size=2,out_hidden_size=32,
            num_position_embeddings=16,deepstack_visual_indexes=[0,1]),
        image_token_id=60,video_token_id=61,vision_start_token_id=62,vision_end_token_id=63)
    backbone = Qwen3VLForConditionalGeneration(config)
    cfg = PoseConfig(width=16,heads=2,layers=2,clip=2,steps=2,bottleneck=16,binding_heads=2)
    model = Iris(backbone,[55,56,57],cfg)
    # Qwen3 video is timestamp + separately wrapped effective frames.
    ids = torch.tensor([[1,62,61,63,2,62,61,63,55]+[57]*20+[56,3,4]])
    pose = torch.rand(1,4,1,24,9)
    pose[...,8] = 1
    labels = ids.clone(); labels[:,:-2] = -100
    example = dict(input_ids=ids,attention_mask=torch.ones_like(ids),labels=labels,pose=pose,
        video_grid_thw=torch.tensor([[2,2,2]]),pixel_values_videos=torch.randn(8,24))
    return model,example

def test_actual_qwen_forward_backward_and_positions():
    model,x = fixture()
    captured = {}
    def hook(module,args,kwargs):
        captured.update(kwargs)
    handle = model.backbone.model.language_model.register_forward_pre_hook(hook,with_kwargs=True)
    out = model(**x)
    assert torch.isfinite(out.loss)
    out.loss.backward()
    assert model.binding.out.weight.grad.abs().sum()>0
    assert model.projector[0].weight.grad.abs().sum()>0
    assert model.pose_encoder.convs[0].weight.grad.abs().sum()>0
    pos = captured['position_ids'][:,0]
    assert (pos[0,9:19]==pos[0,2]).all()
    assert (pos[0,19:29]==pos[0,6]).all()
    assert pos[:,-1].min()>pos[:,:30].max()
    assert captured['deepstack_visual_embeds'] is not None
    handle.remove()

def test_stage2_adamw_keeps_old_embeddings_exactly_frozen():
    model,x = fixture(); model.configure_stage(2)
    old = model.backbone.get_input_embeddings().weight.detach().clone()
    optimizer = torch.optim.AdamW(model.optimizer_groups(2))
    model(**x).loss.backward(); optimizer.step()
    new = model.backbone.get_input_embeddings().weight.detach()
    keep = torch.ones(len(old),dtype=torch.bool); keep[model.token_ids]=False
    assert torch.equal(old[keep],new[keep])
    assert not torch.equal(old[55],new[55])
    assert all(not p.requires_grad for p in model.pose_encoder.parameters())
    model.configure_stage(3)
    assert all(not p.requires_grad for p in model.backbone.visual.parameters())
    assert all(p.requires_grad for p in model.pose_encoder.parameters())

def test_rgb_fallback_identical():
    model,x=fixture(); model.eval()
    x.pop('pose'); x.pop('labels')
    with torch.no_grad():
        assert torch.equal(model(**x).logits,model.backbone(**x).logits)

def test_empty_binding_and_mask_isolation():
    cfg=PoseConfig(bottleneck=16,binding_heads=2)
    layer=SpatialBinding(16,cfg)
    q=torch.randn(1,1,1,5,16); v=torch.randn(1,1,4,16)
    empty=torch.zeros(1,1,1,4,dtype=torch.bool)
    assert torch.equal(layer(q,v,empty),torch.zeros_like(q))
    torch.nn.init.normal_(layer.out.weight)
    allowed=empty.clone(); allowed[...,0]=True
    out=layer(q,v,allowed)
    changed=v.clone(); changed[:,:,1:]+=100
    assert torch.allclose(out,layer(q,changed,allowed))
    layer(q,v,empty).sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in layer.parameters() if p.grad is not None)

def test_descriptor_and_short_gap():
    uv=np.ones((8,1,24,2))*2; xyz=np.ones((8,1,24,3))
    x=descriptor(uv,xyz,np.zeros((8,1,3)),np.ones((8,1,4,4),bool),4,4,1,1)
    assert np.all(x[...,2:5]==0) and np.all(x[...,5:8]==1)
    x[3:5]=0
    y=clean(x)
    assert y[3:5,...,8].all()
    assert np.isfinite(y).all()

def test_contrastive_backward():
    cfg=PoseConfig(width=16,heads=2,layers=1,clip=2,steps=2)
    bert=BertModel(BertConfig(vocab_size=32,hidden_size=16,num_hidden_layers=1,num_attention_heads=2,intermediate_size=32))
    model=ContrastivePose(cfg,bert,8)
    pose=torch.rand(3,4,1,24,9); pose[...,8]=1
    out=model(pose,torch.randint(0,32,(3,5)),torch.ones(3,5,dtype=torch.long))
    out['loss'].backward()
    assert model.pose_encoder.convs[0].weight.grad.abs().sum()>0
    assert model.text_head.weight.grad.abs().sum()>0

def test_invalid_slot_count():
    import pytest
    model,x=fixture(); x['input_ids'][0,10]=3
    with pytest.raises(ValueError,match='placeholder'):
        model(**x)

def test_padding_does_not_change_pose_features():
    cfg=PoseConfig(width=16,heads=2,layers=1,clip=2,steps=2)
    encoder=PoseEncoder(cfg).eval()
    one=torch.rand(1,4,1,24,9); one[...,8]=1
    padded=torch.cat((one,torch.zeros(1,4,2,24,9)),dim=2)
    assert torch.allclose(encoder(one),encoder(padded)[:,:,:1],atol=1e-6)

def test_multi_person_and_checkpoint(tmp_path):
    model,x=fixture()
    x['pose']=x['pose'].expand(-1,-1,3,-1,-1).clone()
    ids=torch.tensor([[1,62,61,63,2,62,61,63,55]+[57]*60+[56,3,4]])
    x.update(input_ids=ids,attention_mask=torch.ones_like(ids),labels=ids.clone())
    x['labels'][:,:-2]=-100
    model.eval()
    expected=model(**x).logits
    torch.save(model.state_dict(),tmp_path/'state.pt')
    copy,_=fixture(); copy.load_state_dict(torch.load(tmp_path/'state.pt',weights_only=True)); copy.eval()
    assert torch.equal(expected,copy(**x).logits)

def test_inference_manifest_never_includes_answer(tmp_path,monkeypatch):
    import sys
    from types import SimpleNamespace
    from iris.data import Manifest
    class Tokenizer:
        def __call__(self,*args,**kwargs):
            raise AssertionError('Inference must not tokenize gold answer')
    class Processor:
        tokenizer=Tokenizer()
        def apply_chat_template(self,messages,**kwargs):
            assert 'SECRET' not in str(messages)
            return 'prompt'
        def __call__(self,**kwargs):
            assert kwargs['do_sample_frames'] is False
            return {'input_ids':torch.tensor([[1,2]]),'attention_mask':torch.ones(1,2,dtype=torch.long)}
    monkeypatch.setitem(sys.modules,'qwen_vl_utils',SimpleNamespace(process_vision_info=lambda *a,**k:(None,[(torch.zeros(16,3,4,4),{})],{'do_sample_frames':False})))
    path=tmp_path/'manifest.jsonl'
    path.write_text('{"video":"x.mp4","question":"Q?","answer":"SECRET"}\n')
    item=Manifest(path,Processor(),inference=True)[0]
    assert item['input_ids'].tolist()==[[1,2]]
