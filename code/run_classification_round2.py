"""Three predeclared splits, two image datasets, no test-set selection."""
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).parent/'.mpl-cache'))
import json,time,hashlib
import numpy as np
from sklearn.datasets import load_digits,fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mfa import fit_mfa,fit_mmfa
ROOT=Path(__file__).resolve().parent;OUT=ROOT.parent/'results/round2';OUT.mkdir(exist_ok=True)
CONFIG=dict(seeds=[16,17,18],k1=3,k2=50,ndim=16,matrix_shape=[4,4],iterations=10,pca_energy=.98,blas_threads=1,
            selection='fixed in round2-plan.md before results; no hyperparameter sweep')


def main():
    digits=load_digits();faces=fetch_olivetti_faces(data_home=str(ROOT/'.data-cache'),download_if_missing=False)
    # Exact 2x2 block means, preserving the spatial matrix for MMFA.
    fi=faces.images.reshape(400,32,2,32,2).mean(axis=(2,4)).astype(float)
    datasets=[('Digits',digits.data.astype(float)/16,digits.target,(8,8)),
              ('Olivetti',fi.reshape(400,-1),faces.target,(32,32))]
    rows=[];splits={};predictions={}
    with threadpool_limits(limits=1):
        for dataset,x,y,shape in datasets:
            for seed in CONFIG['seeds']:
                if dataset=='Digits':
                    pool,test=train_test_split(np.arange(len(x)),test_size=.3,random_state=seed,stratify=y)
                    train,_=train_test_split(pool,train_size=600,random_state=seed,stratify=y[pool])
                else:
                    rng=np.random.default_rng(seed)
                    perclass=[rng.permutation(np.flatnonzero(y==c)) for c in np.unique(y)]
                    train=np.concatenate([ii[:6] for ii in perclass]);test=np.concatenate([ii[6:] for ii in perclass])
                assert not np.intersect1d(train,test).size
                key=f'{dataset}_{seed}';splits[key+'_train']=train;splits[key+'_test']=test
                trainx,testx=x[train],x[test]
                jobs=[('Raw pixels',None),('PCA',lambda:PCA(16,svd_solver='full').fit(trainx)),
                      ('LDA',lambda:LinearDiscriminantAnalysis(n_components=min(16,len(np.unique(y))-1),solver='svd').fit(trainx,y[train])),
                      ('MFA-1',lambda:fit_mfa(trainx,y[train],16,3,50,solver='mfa1')),
                      ('MFA-2',lambda:fit_mfa(trainx,y[train],16,3,50,solver='mfa2',seed=seed)),
                      ('MMFA',lambda:fit_mmfa(trainx.reshape(-1,*shape),y[train],(4,4),3,50,10)),
                      ('MFA-2 shared PCA',lambda:fit_mfa(trainx,y[train],16,3,50,solver='mfa2',seed=seed,mfa2_pca_energy=.98))]
                for name,job in jobs:
                    start=time.perf_counter();model=job() if job else None;seconds=time.perf_counter()-start
                    ti=trainx.reshape(-1,*shape) if name=='MMFA' else trainx
                    vi=testx.reshape(-1,*shape) if name=='MMFA' else testx
                    ztr=model.transform(ti) if model else ti;zte=model.transform(vi) if model else vi
                    pred=KNeighborsClassifier(1).fit(ztr,y[train]).predict(zte)
                    row=dict(dataset=dataset,seed=seed,method=name,ndim=ztr.shape[1],train_count=len(train),test_count=len(test),
                             correct=int(np.sum(pred==y[test])),accuracy=float(np.mean(pred==y[test])),fit_seconds=seconds,
                             diagnostic_only=name=='MFA-2 shared PCA')
                    if name.startswith('MFA-2'):
                        sol=model.trace_solution
                        row.update(converged=sol.converged,spectral_residual=sol.spectral_residual,iterations=len(sol.history)-1,retained_rank=sol.retained_rank)
                        assert sol.converged
                    rows.append(row);predictions[key+'_'+name.replace(' ','_')]=pred
                    print(json.dumps(row),flush=True)
    summary=[]
    for dataset,*_ in datasets:
        for name,_ in jobs:
            items=[v for v in rows if v['dataset']==dataset and v['method']==name]
            summary.append(dict(dataset=dataset,method=name,mean_accuracy=float(np.mean([v['accuracy'] for v in items])),
                                sd_accuracy=float(np.std([v['accuracy'] for v in items],ddof=1)),
                                mean_fit_seconds=float(np.mean([v['fit_seconds'] for v in items])),seeds=CONFIG['seeds'],ndim=items[0]['ndim']))
    np.savez(OUT/'classification_splits.npz',**splits);np.savez(OUT/'classification_predictions.npz',**predictions)
    result=dict(config=CONFIG,rows=rows,summary=summary,scope='replacement image data; not USF benchmark reproduction',
                dataset_sha256={name:hashlib.sha256(x.tobytes()+y.tobytes()).hexdigest() for name,x,y,_ in datasets})
    (OUT/'classification.json').write_text(json.dumps(result,indent=2)+'\n')
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    main_methods=['Raw pixels','PCA','LDA','MFA-1','MFA-2','MMFA']
    for ax,(dataset,*_) in zip(axs,datasets):
        data=[s for s in summary if s['dataset']==dataset and s['method'] in main_methods]
        ax.bar(np.arange(6),[s['mean_accuracy']*100 for s in data],yerr=[s['sd_accuracy']*100 for s in data],
               capsize=3,color=['#899aa9','#899aa9','#477a9d','#156e78','#156e78','#b8791f'])
        ax.set(xticks=np.arange(6),xticklabels=['Raw','PCA','LDA','MFA-1','MFA-2','MMFA'],ylim=(0,105),ylabel='1-NN accuracy (%)',title=dataset+' · mean ± SD across 3 splits')
    fig.savefig(OUT/'classification.png',dpi=180);plt.close(fig)
    print('FINISHED',json.dumps(summary),flush=True)

if __name__=='__main__':main()
