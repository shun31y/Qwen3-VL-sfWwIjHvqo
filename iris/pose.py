"""Deterministic processing of estimator exports in Table-7 joint order."""
import argparse
import numpy as np


def descriptor(uv, xyz, translation, masks, width, height, local_scale, global_scale):
    """uv [T,P,24,2] pixels; xyz local camera-axis 3D; translation [T,P,3].

    masks [T,P,H,W] bool. xyz + translation is camera-space 3D.
    Scales must be fitted on TRAINING data only; the paper does not supply them.
    """
    if local_scale <= 0 or global_scale <= 0:
        raise ValueError('Normalization scales must be positive')
    if xyz.shape[-2:] != (24,3) or uv.shape != xyz.shape[:-1]+(2,):
        raise ValueError('Expected Table-7 ordered 24 joints')
    pelvis = xyz[...,[14,15],:].mean(-2)
    out = np.zeros(xyz.shape[:-1]+(9,),dtype=np.float32)
    out[...,:2] = uv / [width,height]
    out[...,2:5] = (xyz-pelvis[...,None,:])/local_scale
    out[...,5:8] = ((pelvis+translation)/global_scale)[...,None,:]
    for t in range(len(uv)):
        for p in range(uv.shape[1]):
            finite = np.isfinite(uv[t,p]).all(-1) & np.isfinite(xyz[t,p]).all(-1)
            px = np.nan_to_num(uv[t,p]).astype(int)
            inside = finite & (uv[t,p,:,0]>=0) & (uv[t,p,:,0]<width) & (uv[t,p,:,1]>=0) & (uv[t,p,:,1]<height)
            out[t,p,inside,8] = masks[t,p,px[inside,1],px[inside,0]]
    out[~np.isfinite(out)] = 0
    return out


def clean(pose, max_gap=3, alpha=.7, jitter_threshold=None):
    x = np.asarray(pose,dtype=np.float32).copy()
    if x.ndim != 4 or x.shape[-2:] != (24,9) or not np.isfinite(x).all():
        raise ValueError('Pose must be finite [T,P,24,9]')
    if not np.isin(x[...,8],[0,1]).all():
        raise ValueError('Visibility must be binary')
    for p in range(x.shape[1]):
        valid = x[:,p,:,8].any(-1)
        neighbors = np.r_[False,valid[:-1]] | np.r_[valid[1:],False]
        valid &= neighbors
        x[~valid,p] = 0
        indices = np.flatnonzero(valid)
        for a,b in zip(indices[:-1],indices[1:]):
            if 1 < b-a <= max_gap+1:
                for t in range(a+1,b):
                    ratio = (t-a)/(b-a)
                    x[t,p,:,:8] = (1-ratio)*x[a,p,:,:8]+ratio*x[b,p,:,:8]
                    x[t,p,:,8] = x[a,p,:,8]*x[b,p,:,8]
        if jitter_threshold is not None:
            visible = (x[1:,p,:,8]*x[:-1,p,:,8]).astype(bool)
            speed = np.linalg.norm(np.diff(x[:,p,:,2:5],axis=0),axis=-1)
            if visible.any() and speed[visible].max() > jitter_threshold:
                x[:,p] = 0
                continue
        for t in range(1,len(x)):
            visible = (x[t,p,:,8]*x[t-1,p,:,8]).astype(bool)
            x[t,p,visible,:8] = alpha*x[t,p,visible,:8]+(1-alpha)*x[t-1,p,visible,:8]
    return x[:,x[...,8].any(axis=(0,2))]


def sample(pose, frames=160):
    if len(pose)==0:
        raise ValueError('Empty temporal sequence')
    return pose[np.linspace(0,len(pose)-1,frames).round().astype(int)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('input',help='NPZ with uv, xyz, translation, masks in Table-7 order')
    p.add_argument('output')
    p.add_argument('--local-scale',type=float,required=True)
    p.add_argument('--global-scale',type=float,required=True)
    p.add_argument('--jitter-threshold',type=float,required=True)
    args = p.parse_args()
    raw = np.load(args.input,allow_pickle=False)
    h,w = raw['masks'].shape[-2:]
    x = descriptor(raw['uv'],raw['xyz'],raw['translation'],raw['masks'],w,h,args.local_scale,args.global_scale)
    x = clean(x,jitter_threshold=args.jitter_threshold)
    np.save(args.output,sample(x)[:,:3])

if __name__ == '__main__':
    main()
