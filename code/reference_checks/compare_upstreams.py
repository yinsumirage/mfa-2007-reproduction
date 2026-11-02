"""Execute preserved upstream code and compare like-for-like definitions."""
from pathlib import Path
import hashlib, importlib.util, json, os, subprocess, sys
import numpy as np
from scipy.io import savemat, loadmat
from scipy.linalg import subspace_angles
from scipy.spatial.distance import cdist, pdist

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from mfa import build_graphs, Graphs, laplacian, ged_projection
OUT=ROOT.parent/'results/round2/reference_checks';OUT.mkdir(parents=True,exist_ok=True)
ENV=ROOT/'.reference-env'


def run(args, env=None):
    p=subprocess.run(list(map(str,args)),env=env,capture_output=True,text=True,timeout=90)
    (OUT/(Path(args[0]).name+'.log')).write_text(p.stdout+'\n'+p.stderr)
    if p.returncode:raise RuntimeError(p.stderr[-2500:])
    return p


def global_pairs(x,y,k1,k2):
    n=len(x);d=cdist(x,x,'sqeuclidean');a=np.zeros((n,n));b=np.zeros((n,n))
    for i in range(n):
        candidates=np.flatnonzero((y==y[i])&(np.arange(n)!=i))
        ids=candidates[np.argsort(d[i,candidates],kind='stable')[:k1]]
        a[i,ids]=a[ids,i]=1
    d[y[:,None]==y[None,:]]=d.max()+100
    ids=np.argsort(d.ravel(order='F'),kind='stable')[:k2]
    i,j=np.unravel_index(ids,(n,n),order='F');b[i,j]=b[j,i]=1
    return Graphs(a,b)


def compare(w1,w2,x):
    err=float(np.max(np.abs(pdist(x@w1)-pdist(x@w2))))
    angle=float(np.max(subspace_angles(w1,w2)))
    assert angle<1e-6 and err<1e-7,(angle,err)
    return dict(max_subspace_angle_radians=angle,max_pair_distance_abs_error=err)


def main():
    rng=np.random.default_rng(1616);x=rng.normal(size=(60,6));y=np.repeat(np.arange(3),20)
    x-=x.mean(axis=0)
    g=global_pairs(x,y,3,80);paper=build_graphs(x,y,3,20)
    sw,sp=laplacian(g.within),laplacian(g.between)
    savemat(OUT/'fixture.mat',dict(x=x,y=y[:,None],sw=sw,sp=sp,
                                  paper_sw=laplacian(paper.within),paper_sp=laplacian(paper.between)))
    matlab=ROOT/'third_party/mfa-matlab-reference/feature_extraction/MFA'
    script=OUT/'run_reference.m'
    script.write_text(f"""addpath('{matlab}');
load('{OUT}/fixture.mat');
options=struct('intraK',3,'interK',80,'ReducedDim',3,'Regu',1,'ReguAlpha',0,'bEigs',0,'keepMean',1);
[wfull,evfull]=MFA(y,options,x);
[wmatched,evmatched]=LGE(paper_sp,paper_sw,options,x);
options.Regu=0;options.PCARatio=.98;
[wpca,evpca]=LGE(paper_sp,paper_sw,options,x);
save('-mat7-binary','{OUT}/octave_results.mat','wfull','evfull','wmatched','evmatched','wpca','evpca');
""")
    env=os.environ.copy();env['OCTAVE_HOME']=str(ENV);env['OPENBLAS_NUM_THREADS']='1'
    run([ENV/'bin/octave-cli','--quiet',script],env)
    o=loadmat(OUT/'octave_results.mat')
    ours=ged_projection(*g.scatters(x),3)
    result={'matlab_full_mfa':compare(o['wfull'],ours,x),
            'matlab_lge_paper_graphs':compare(o['wmatched'],ged_projection(*paper.scatters(x),3),x)}
    _,s,vt=np.linalg.svd(x,full_matrices=False);r=int(np.searchsorted(np.cumsum(s*s)/sum(s*s),.98)+1)
    q=vt[:r].T;result['matlab_lge_pca98']=compare(o['wpca'],q@ged_projection(*paper.scatters(x@q),3),x)
    np.savetxt(OUT/'x.csv',x,delimiter=',');np.savetxt(OUT/'y.csv',y,delimiter=',',fmt='%d')
    rscript=OUT/'run_reference.R'
    rscript.write_text(f"""source('{ROOT}/third_party/rdimtools/R/auxfuncs.R')
x=as.matrix(read.csv('{OUT}/x.csv',header=FALSE));y=scan('{OUT}/y.csv',quiet=TRUE)
g=aux.nbdlogical(x,y,3,4)
w=(g$hom | t(g$hom))*1;wp=(g$het | t(g$het))*1;diag(w)=0;diag(wp)=0
write.table(w,'{OUT}/r_within.csv',sep=',',row.names=FALSE,col.names=FALSE)
write.table(wp,'{OUT}/r_between.csv',sep=',',row.names=FALSE,col.names=FALSE)
""")
    run([ENV/'bin/Rscript',rscript])
    expected=np.zeros((len(x),len(x)));d=cdist(x,x,'sqeuclidean')
    for i in range(len(x)):
        candidates=np.flatnonzero(y!=y[i]);ids=candidates[np.argsort(d[i,candidates],kind='stable')[:4]]
        expected[i,ids]=expected[ids,i]=1
    rw=np.loadtxt(OUT/'r_within.csv',delimiter=',');rb=np.loadtxt(OUT/'r_between.csv',delimiter=',')
    assert np.array_equal(rw,paper.within) and np.array_equal(rb,expected)
    result['rdimtools_graph_helper']=dict(within_exact_equal=True,between_exact_equal=True,
           scope='original aux.nbdlogical executed; full do.mfa/maotai not executed')
    spec=importlib.util.spec_from_file_location('upstream_multiple_mfa',ROOT/'third_party/multiple-mfa-python/mmfa.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    # Only a removed NumPy name is bridged; original source remains untouched.
    old=getattr(np,'mat',None);np.mat=np.asmatrix
    x2=rng.normal(size=(45,6));y2=np.repeat(np.arange(3),15);captured={}
    def trace(frame,event,arg):
        if frame.f_code is module.MMFA.__code__ and event=='return':
            for k in ['W','W_','X_1','X_2']:captured[k]=np.asarray(frame.f_locals[k]).copy()
        return trace
    sys.settrace(trace)
    try:wref=module.MMFA(x2,y2,3,14,pca_op=0)
    finally:
        sys.settrace(None)
        if old is None:del np.mat
        else:np.mat=old
    expected=np.zeros((45,45));dist=cdist(x2,x2,'sqeuclidean')
    for c in range(3):
        for e in range(c+1,3):
            ii,jj=np.flatnonzero(y2==c),np.flatnonzero(y2==e)
            for _,i,j in sorted((dist[i,j],i,j) for i in ii for j in jj)[:14]:expected[i,j]=expected[j,i]=1
    assert np.array_equal(captured['W_'],expected)
    our=ged_projection(captured['X_1'],captured['X_2'],wref.shape[1])
    result['python_multiple_mfa']=compare(wref,our,x2)
    result['python_multiple_mfa'].update(cross_class_graph_exact_equal=True,compatibility='runtime np.mat=np.asmatrix; pca_op=0',scope='Multiple MFA, not matrix-based MFA')
    result['k2_semantics_edge_counts_on_fixture']={
        'paper_class_vs_rest_k2_20':int(np.sum(paper.between)/2),
        'matlab_global_ordered_entries_80':int(np.sum(g.between)/2),
        'r_per_sample_k2_4':int(np.sum(rb)/2)}
    result['source_hashes']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [matlab/'MFA.m',matlab/'LGE.m',ROOT/'third_party/rdimtools/R/auxfuncs.R',ROOT/'third_party/multiple-mfa-python/mmfa.py']}
    result['versions']={'octave':'10.3.0','R':'4.5.2','numpy':np.__version__}
    (OUT/'comparison.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):main()
