"""Post-hoc diagnostic with identical observed BDA feedback for all methods."""
from pathlib import Path
import json
import numpy as np
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits
from biased import fit_biased
ROOT=Path(__file__).resolve().parent;OUT=ROOT.parent/'results/round2'
f=np.load(OUT/'corel1k_features.npz');x,y=f['features'],f['labels']
res=json.loads((OUT/'heldout_retrieval.json').read_text());splits=np.load(OUT/'heldout_retrieval_split.npz')
rows=[]
with threadpool_limits(limits=1):
    for r in res['rows']:
        if r['method']!='BDA' or 'training_ids' not in r:continue
        q=r['query'];ids=np.asarray(r['training_ids']);mask=y[ids]==y[q]
        evaluation=splits['evaluation_pool'];evaluation=evaluation[evaluation!=q]
        for name,method,gamma in [('BDA','bda',None),('MBA','mba',None),('KBDA','bda',res['gamma']),('KMBA','mba',res['gamma'])]:
            m=fit_biased(x[ids],mask,2,min(3,int(mask.sum())-1),min(20,int(mask.sum()*(~mask).sum())),method=method,gamma=gamma,ridge=1e-3)
            z=m.transform(x[evaluation]);qz=m.transform(x[[q]])
            nearest=np.argsort(cdist(qz,z,'sqeuclidean')[0],kind='stable')[:20]
            rows.append(dict(method=name,query=q,round=r['round']+1,p20=float(np.mean(y[evaluation[nearest]]==y[q])),training_ids=ids.tolist()))
summary=[]
for name in ['BDA','MBA','KBDA','KMBA']:
 for step in range(1,6):
    vals=[r['p20'] for r in rows if r['method']==name and r['round']==step]
    summary.append(dict(method=name,round=step,mean=float(np.mean(vals)),sd=float(np.std(vals,ddof=1)),queries=len(vals)))
(OUT/'matched_feedback.json').write_text(json.dumps(dict(scope='post-hoc diagnostic; identical BDA feedback, not parameter selection',summary=summary,rows=rows),indent=2)+'\n')
print(json.dumps([r for r in summary if r['round']==5]),flush=True)
