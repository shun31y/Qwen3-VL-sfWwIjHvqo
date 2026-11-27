"""Score already normalized benchmark outputs; never silently parse free-form answers."""
import argparse
import json
import numpy as np

def score(rows,task):
    if not rows:
        raise ValueError('No predictions')
    if task=='choice':
        return {'accuracy':float(np.mean([r['prediction']==r['target'] for r in rows])),'n':len(rows)}
    predicted=np.array([float(r['prediction']) for r in rows])
    target=np.array([float(r['target']) for r in rows])
    if not np.isfinite(predicted).all() or not np.isfinite(target).all() or (target<=0).any():
        raise ValueError('Counting targets must be positive; predictions and targets must be finite')
    return {'normalized_mae':float(np.mean(np.abs(predicted-target)/target)),
            'off_by_one':float(np.mean(np.abs(predicted-target)<=1)),'n':len(rows)}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('predictions',help='JSONL with normalized prediction and target fields')
    p.add_argument('--task',choices=['choice','count'],required=True)
    args=p.parse_args()
    with open(args.predictions) as f:
        rows=[json.loads(s) for s in f if s.strip()]
    print(json.dumps(score(rows,args.task),indent=2))

if __name__=='__main__':
    main()
