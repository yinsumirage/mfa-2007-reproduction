"""Corel1K task-level substitute experiment; no result-dependent tuning."""
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).parent/'.mpl-cache'))
import hashlib,json,time
import numpy as np
from scipy.spatial.distance import pdist
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from retrieval_features import descriptor
from retrieval import run_query
ROOT=Path(__file__).resolve().parent;OUT=ROOT.parent/'results/round2';DATA=ROOT/'.data-cache/corel1k-mirror/img'
CONFIG=dict(query_seed=16,queries_per_class=10,rounds=9,output_dim=2,ridge=1e-3,k1_cap=3,k2_cap=20,
            feedback_top=40,feedback_positive=5,feedback_negative=5,descriptor='HSV256+Haar24+edge5; block L2 normalization',
            gamma_rule='inverse median squared distance among 200 fixed unlabeled samples',main_metric='precision at 20 excluding query AND all seen feedback',original_dataset=False)


def main():
    files=sorted(DATA.glob('*.jpg'));names=[p.name.split(' (')[0] for p in files];classes=sorted(set(names));y=np.array([classes.index(n) for n in names])
    assert len(files)==1000
    cached=OUT/'corel1k_features.npz'
    if cached.exists():x=np.load(cached)['features']
    else:
        x=np.array([descriptor(p) for p in files]);np.savez(cached,features=x,labels=y,names=np.array([p.name for p in files]))
    rng=np.random.default_rng(16);queries=np.concatenate([rng.choice(np.flatnonzero(y==c),10,replace=False) for c in range(10)])
    gammaids=np.random.default_rng(1616).choice(len(x),200,replace=False);gamma=float(1/np.median(pdist(x[gammaids],'sqeuclidean')))
    (OUT/'retrieval_plan.json').write_text(json.dumps(dict(config=CONFIG,gamma=gamma,queries=queries.tolist(),query_filenames=[files[i].name for i in queries],feature_sha256=hashlib.sha256(x.tobytes()).hexdigest()),indent=2)+'\n')
    rows=[];start=time.perf_counter()
    with threadpool_limits(limits=1):
        for method,kern in [('BDA',None),('MBA',None),('KBDA',gamma),('KMBA',gamma)]:
            methodstart=time.perf_counter()
            for number,query in enumerate(queries):
                history=run_query(x,y,int(query),'mba' if method.endswith('MBA') else 'bda',rounds=9,gamma=kern,ridge=1e-3)
                for row in history:row['method']=method
                rows.extend(history)
                if (number+1)%20==0:print(json.dumps(dict(method=method,queries=number+1,seconds=time.perf_counter()-methodstart)),flush=True)
    summary=[]
    for method in ['BDA','MBA','KBDA','KMBA']:
        for step in range(10):
            items=[r for r in rows if r['method']==method and r['round']==step]
            row=dict(method=method,round=step,count=len(items))
            for metric in ['p20_all','p20_unseen','p40_all','p40_unseen','p60_all','p60_unseen']:
                vals=[r[metric] for r in items if r[metric] is not None]
                row[metric+'_mean']=float(np.mean(vals));row[metric+'_sd']=float(np.std(vals,ddof=1))
            row['shortfall_count']=sum(r.get('feedback_shortfall',False) for r in items)
            summary.append(row)
    result=dict(config=CONFIG,gamma=gamma,elapsed_seconds=time.perf_counter()-start,summary=summary,rows=rows,
                scope='Corel1K third-party mirror, task-level adaptation; not original Corel22K curves')
    (OUT/'retrieval.json').write_text(json.dumps(result,indent=2)+'\n')
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for method,color in zip(['BDA','MBA','KBDA','KMBA'],['#899aa9','#156e78','#b8791f','#477a9d']):
        rr=[r for r in summary if r['method']==method]
        for ax,metric in zip(axs,['p20_all_mean','p20_unseen_mean']):ax.plot(range(10),[r[metric]*100 for r in rr],marker='o',label=method,color=color)
    axs[0].set_title('Exclude query only');axs[1].set_title('Exclude query + seen feedback')
    for ax in axs:ax.set(xlabel='Feedback round',ylabel='Mean precision@20 (%)',ylim=(0,100));ax.legend()
    fig.savefig(OUT/'retrieval_precision.png',dpi=180);plt.close(fig)
    print('FINISHED',json.dumps([r for r in summary if r['round']==9]),flush=True)

if __name__=='__main__':main()
