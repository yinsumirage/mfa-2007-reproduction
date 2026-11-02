"""Cross-language GED check for Eq.16/17/18 matrices, including singular ridge."""
from pathlib import Path
import os,sys,subprocess,json
import numpy as np
from scipy.io import savemat,loadmat
from scipy.spatial.distance import cdist,pdist
from scipy.linalg import subspace_angles
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from mfa import laplacian
from biased import fit_biased
OUT=ROOT.parent/'results/round2/reference_checks';ENV=ROOT/'.reference-env'
rng=np.random.default_rng(322);x=rng.normal(size=(14,5));pos=np.arange(14)<7
p,n=pos.astype(float),(~pos).astype(float);P,N=p.sum(),n.sum()
lwa=np.diag(p)-np.outer(p,p)/P
lba=np.diag(n)+N/P**2*np.outer(p,p)-(np.outer(p,n)+np.outer(n,p))/P
fixture={};models={};code=[f"addpath('{ROOT}/third_party/mfa-matlab-reference/feature_extraction/MFA');",f"load('{OUT}/biased_fixture.mat');"]
for name,method,gamma in [('mba','mba',None),('bda','bda',None),('kmba','mba',.2),('kbda','bda',.2)]:
    m=fit_biased(x,pos,2,3,12,method=method,gamma=gamma,ridge=1e-3);models[name]=m
    features=x if gamma is None else np.exp(-gamma*cdist(x,x,'sqeuclidean'))
    lw,lb=(lwa,lba) if method=='bda' else (laplacian(m.graphs.within),laplacian(m.graphs.between))
    fixture.update({name+'_x':features,name+'_lw':lw,name+'_lb':lb,name+'_alpha':m.alpha})
    code.extend([f"options=struct('ReducedDim',2,'Regu',1,'ReguAlpha',{name}_alpha,'bEigs',0);",f"[w_{name},v_{name}]=LGE({name}_lb,{name}_lw,options,{name}_x);"])
code.append("save('-mat7-binary','"+str(OUT/'biased_octave.mat')+"','w_mba','w_bda','w_kmba','w_kbda');")
savemat(OUT/'biased_fixture.mat',fixture);(OUT/'biased_check.m').write_text('\n'.join(code))
env=os.environ.copy();env.update(OCTAVE_HOME=str(ENV),OPENBLAS_NUM_THREADS='1')
proc=subprocess.run([str(ENV/'bin/octave-cli'),'--quiet',str(OUT/'biased_check.m')],env=env,capture_output=True,text=True,timeout=90)
(OUT/'biased_octave.log').write_text(proc.stdout+'\n'+proc.stderr)
if proc.returncode:raise RuntimeError(proc.stderr)
o=loadmat(OUT/'biased_octave.mat');result={}
for name,m in models.items():
    w=o['w_'+name]
    if m.gamma is not None:
        k=fixture[name+'_x'];w/=np.sqrt(np.einsum('ij,ik,kj->j',w,k,w))
    angle=float(subspace_angles(w,m.projection).max());err=float(np.abs(pdist(fixture[name+'_x']@w)-pdist(m.transform(x))).max())
    assert angle<1e-6 and err<1e-7,(name,angle,err)
    result[name]=dict(subspace_angle_radians=angle,max_pair_distance_abs_error=err,alpha=m.alpha)
(OUT/'biased_comparison.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
