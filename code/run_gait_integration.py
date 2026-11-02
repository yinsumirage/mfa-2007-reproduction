"""Synthetic binary silhouettes: integration check of Eqs.8-9, not a benchmark."""
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).parent/'.mpl-cache'))
import json
import numpy as np
from threadpoolctl import threadpool_limits
from sklearn.decomposition import PCA
from mfa import fit_mfa,fit_mmfa
from gait import cycle_average,identify
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent;OUT=ROOT.parent/'results/round2'


def frames(parameters,seed):
    rng=np.random.default_rng(seed);yy,xx=np.mgrid[:32,:24];result=[]
    width,height,stride=parameters
    for t in range(48):
        phase=2*np.pi*(t%12)/12
        head=((xx-12)/2.3)**2+((yy-(9-height))/2.5)**2<1
        torso=(abs(xx-12)<width)&(yy>=10-height)&(yy<21)
        legs=np.zeros_like(head)
        for sign in [-1,1]:
            center=12+sign*stride*np.sin(phase)*(yy-20)/10
            legs|=(np.abs(xx-center)<1.1)&(yy>=20)&(yy<30)
        silhouette=head|torso|legs
        # Small independent bit noise ensures this is a numerical pipeline
        # fixture, not a claim to model real silhouette extraction errors.
        silhouette=np.logical_xor(silhouette,rng.random(silhouette.shape)<.01)
        result.append(silhouette)
    return np.asarray(result,float)


def main():
    rng=np.random.default_rng(16);params=np.column_stack([rng.uniform(2.3,4.5,12),rng.uniform(0,3,12),rng.uniform(2,5,12)])
    galleries=[];probes=[];glabels=[];plabels=[];train=[];labels=[]
    for i,p in enumerate(params):
        for sequence in range(2):
            avg=cycle_average(frames(p,16000+i*100+sequence),12)
            galleries.append(avg);glabels.append(i);train.extend(avg);labels.extend([i]*len(avg))
        avg=cycle_average(frames(p,18000+i*100),12);probes.append(avg);plabels.append(i)
    train=np.asarray(train);labels=np.asarray(labels);flat=train.reshape(len(train),-1);rows=[]
    with threadpool_limits(limits=1):
        for name,job in [('Raw',None),('PCA',lambda:PCA(8,svd_solver='full').fit(flat)),
                         ('MFA-1',lambda:fit_mfa(flat,labels,8,3,20,solver='mfa1')),
                         ('MFA-2',lambda:fit_mfa(flat,labels,8,3,20,solver='mfa2')),
                         ('MMFA',lambda:fit_mmfa(train,labels,(2,4),3,20,5,ridge=1e-6))]:
            model=job() if job else None
            def transform(s):
                if model is None:return s.reshape(len(s),-1)
                return model.transform(s if name=='MMFA' else s.reshape(len(s),-1))
            top,dist=identify([transform(s) for s in probes],[transform(s) for s in galleries],glabels)
            assert dist.shape==(12,24) and np.isfinite(dist).all()
            rows.append(dict(method=name,rank1_count=int(np.sum(top[:,0]==plabels)),probe_sequences=12))
    result=dict(scope='synthetic integration fixture only, not measured gait performance',period=12,cycles_per_sequence=4,
                gallery_sequences=24,probe_sequences=12,training_cycle_images=96,seed=16,
                mmfa_ridge=1e-6,rows=rows)
    (OUT/'gait_integration.json').write_text(json.dumps(result,indent=2)+'\n')
    f=frames(params[0],16000);fig,axs=plt.subplots(1,5,figsize=(9,3),constrained_layout=True)
    for ax,i in zip(axs[:4],[0,3,6,9]):ax.imshow(f[i],cmap='gray_r',vmin=0,vmax=1);ax.set_title(f'Frame {i}');ax.axis('off')
    axs[4].imshow(cycle_average(f,12)[0],cmap='gray_r',vmin=0,vmax=1);axs[4].set_title('Cycle average');axs[4].axis('off')
    fig.suptitle('Synthetic silhouette fixture; supplied period = 12')
    fig.savefig(OUT/'gait_pipeline.png',dpi=180);plt.close(fig);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
