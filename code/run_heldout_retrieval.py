"""Fixed evaluation gallery, disjoint feedback pool, same queries for all methods."""
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).parent/'.mpl-cache'))
import json
import numpy as np
from threadpoolctl import threadpool_limits
from retrieval import run_heldout_query,run_query
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent;OUT=ROOT.parent/'results/round2'


def summarize(rows,methods,rounds,metric='p20'):
    summary=[]
    for method in methods:
        for step in range(rounds+1):
            items=[r for r in rows if r['method']==method and r['round']==step]
            vals=[r[metric] for r in items]
            summary.append(dict(method=method,round=step,count=len(vals),mean=float(np.mean(vals)),sd=float(np.std(vals,ddof=1)),
                                feedback_shortfalls=sum(r.get('feedback_shortfall',False) for r in items)))
    return summary


def main():
    f=np.load(OUT/'corel1k_features.npz');x,y=f['features'],f['labels'];plan=json.loads((OUT/'retrieval_plan.json').read_text());gamma=plan['gamma']
    rng=np.random.default_rng(1616);pool=[];evaluation=[];queries=[]
    for c in np.unique(y):
        ids=rng.permutation(np.flatnonzero(y==c));pool.extend(ids[:40]);evaluation.extend(ids[40:]);queries.extend(ids[40:50])
    pool,evaluation,queries=map(np.array,[pool,evaluation,queries])
    np.savez(OUT/'heldout_retrieval_split.npz',feedback_pool=pool,evaluation_pool=evaluation,queries=queries)
    methods=['BDA','MBA','KBDA','KMBA'];rows=[];diagnostic=[]
    with threadpool_limits(limits=1):
        for method,kern in zip(methods,[None,None,gamma,gamma]):
            for q in queries:
                rr=run_heldout_query(x,y,int(q),pool,evaluation,'mba' if method.endswith('MBA') else 'bda',gamma=kern,rounds=5)
                for row in rr:row['method']=method
                rows.extend(rr)
            # Original top40-including-seen diagnostic is retained with correct
            # method names. It is not used to choose parameters.
            for q in plan['queries']:
                rr=run_query(x,y,int(q),'mba' if method.endswith('MBA') else 'bda',rounds=9,gamma=kern,display_unseen=False)
                for row in rr:row['method']=method
                diagnostic.extend(rr)
            print(method,'complete',flush=True)
    summary=summarize(rows,methods,5)
    result=dict(protocol='fixed held-out gallery; only disjoint feedback-pool labels enter fitting',rounds=5,queries=100,feedback_pool=400,
                evaluation_pool=600,evaluation_count_per_query=599,gamma=gamma,summary=summary,rows=rows,
                adaptation='Corel1K mirror; explicit ridge and descriptor conventions; not original paper curves')
    (OUT/'heldout_retrieval.json').write_text(json.dumps(result,indent=2)+'\n')
    diag_summary=summarize(diagnostic,methods,9,'p20_unseen')
    (OUT/'retrieval_including_seen_diagnostic.json').write_text(json.dumps(dict(protocol='take top40, then remove already seen; not primary',summary=diag_summary,rows=diagnostic),indent=2)+'\n')
    fig,ax=plt.subplots(figsize=(7,4),constrained_layout=True)
    for method in methods:
        rr=[r for r in summary if r['method']==method];ax.plot(range(6),[r['mean']*100 for r in rr],marker='o',label=method)
    ax.set(xlabel='Feedback round',ylabel='Mean precision@20 (%)',title='Corel1K: fixed held-out gallery, 100 queries',ylim=(0,100));ax.legend()
    fig.savefig(OUT/'heldout_retrieval.png',dpi=180);plt.close(fig)
    print(json.dumps([r for r in summary if r['round']==5]),flush=True)

if __name__=='__main__':main()
