"""One fixed CPU experiment. No hyperparameter search or original-table claims."""
import os
os.environ.setdefault('MPLCONFIGDIR', os.path.join(os.path.dirname(__file__), '.mpl-cache'))
import hashlib
import json
import platform
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy
import sklearn
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from threadpoolctl import threadpool_limits

from mfa import fit_mfa, fit_mmfa, build_graphs, solve_trace_ratio, trace_ratio

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / 'results'
OUT.mkdir(exist_ok=True)
CONFIG = dict(seed=16, train_size=600, test_fraction=0.3, ndim=16, k1=3, k2=50,
              matrix_shape=[4, 4], matrix_iterations=10, mfa2_max_iter=100,
              mfa2_tol=1e-10, pca_energy=0.98, ged_ridge=0.0, blas_threads=1)


def main():
    digits = load_digits()
    x, y = digits.data.astype(float) / 16., digits.target
    pool, test = train_test_split(np.arange(len(x)), test_size=CONFIG['test_fraction'],
                                  random_state=16, stratify=y)
    train, _ = train_test_split(pool, train_size=600, random_state=16, stratify=y[pool])
    assert not np.intersect1d(train, test).size
    np.savez(OUT/'split_indices.npz', train=train, test=test)
    models = {}
    rows = []

    def evaluate(name, model, train_input, test_input, fit_call=None):
        start = time.perf_counter()
        model = fit_call() if fit_call else model
        fit_seconds = time.perf_counter()-start
        ztrain = model.transform(train_input) if model else train_input
        ztest = model.transform(test_input) if model else test_input
        knn = KNeighborsClassifier(n_neighbors=1).fit(ztrain, y[train])
        pred = knn.predict(ztest)
        row = dict(method=name, output_dim=ztrain.shape[1], test_accuracy=float(np.mean(pred==y[test])),
                   correct=int(np.sum(pred==y[test])), test_count=len(test),
                   fit_seconds=fit_seconds, train_objective=None)
        if name.startswith('MFA-'):
            a,b = model.graphs.scatters(x[train]-model.mean)
            row['train_objective'] = trace_ratio(a,b,model.projection)
        if name=='MFA-1': row['pca_rank'] = model.pca_rank
        if name=='MFA-2':
            sol=model.trace_solution
            row.update(iterations=len(sol.history)-1, converged=sol.converged,
                       spectral_residual=sol.spectral_residual, retained_rank=sol.retained_rank)
            if not sol.converged: raise RuntimeError('MFA-2 did not converge')
        if name=='MMFA': row['train_objective']=model.history[-1]['within_between']
        rows.append(row)
        models[name]=model
        print(json.dumps(row), flush=True)

    with threadpool_limits(limits=1):
        evaluate('Raw pixels',None,x[train],x[test])
        evaluate('PCA',None,x[train],x[test], lambda:PCA(n_components=16,svd_solver='full').fit(x[train]))
        for solver,name in [('mfa1','MFA-1'),('mfa2','MFA-2')]:
            evaluate(name,None,x[train],x[test],lambda solver=solver:fit_mfa(
                x[train], y[train],ndim=16,k1=3,k2=50,solver=solver,ridge=0.0))
        evaluate('MMFA',None,x[train].reshape(-1,8,8),x[test].reshape(-1,8,8),
                 lambda:fit_mmfa(x[train].reshape(-1,8,8),y[train],(4,4),3,50,10,ridge=0.0))
        # Solver-only comparison: SAME graph, SAME positive subspace, orthonormal
        # GED span as the initialization. Does not change classification results.
        a,b=models['MFA-1'].graphs.scatters(x[train]-models['MFA-1'].mean)
        warm=solve_trace_ratio(a,b,16,initial=models['MFA-1'].projection)
        diagnostics=dict(initial_lambda=warm.history[0]['lambda'],final_lambda=warm.history[-1]['lambda'],
                         initial_objective=warm.history[0]['within_between'],
                         final_objective=warm.history[-1]['within_between'],
                         converged=warm.converged,spectral_residual=warm.spectral_residual)

    np.savez(OUT/'fitted_parameters.npz',
             mfa1_mean=models['MFA-1'].mean, mfa1_projection=models['MFA-1'].projection,
             mfa2_mean=models['MFA-2'].mean, mfa2_projection=models['MFA-2'].projection,
             mmfa_mean=models['MMFA'].mean, mmfa_left=models['MMFA'].left, mmfa_right=models['MMFA'].right,
             pca_mean=models['PCA'].mean_, pca_components=models['PCA'].components_)

    # Illustration with different scale from the real experiment: no test set,
    # no accuracy claims. Two coordinates make the training edges visible.
    rng=np.random.default_rng(16)
    toy=np.concatenate([rng.normal(loc=(i*2, i%2),scale=.5,size=(12,2)) for i in range(3)])
    labels=np.repeat(np.arange(3),12)
    g=build_graphs(toy,labels,2,5)
    plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for ax,s,title in zip(axs,[g.within,g.between],['Within-class neighbors','Class-versus-rest margin pairs']):
        for i,j in np.argwhere(np.triu(s,1)):
            ax.plot(toy[[i,j],0],toy[[i,j],1],color='#8a9ba8',lw=.8,zorder=1)
        ax.scatter(toy[:,0],toy[:,1],c=labels,cmap='viridis',s=35,zorder=2)
        ax.set(title=title,xlabel='Feature 1',ylabel='Feature 2')
    fig.savefig(OUT/'training_graphs.png',dpi=180);plt.close(fig)
    hist=models['MFA-2'].trace_solution.history
    fig,ax=plt.subplots(figsize=(7,4),constrained_layout=True)
    ax.plot([h['iteration'] for h in hist],[h['lambda'] for h in hist],marker='o',color='#156e78')
    ax.set(xlabel='Iteration',ylabel='Between / total trace ratio',title='MFA-2 on training scatter matrices')
    fig.savefig(OUT/'solver_convergence.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4),constrained_layout=True)
    bars=ax.bar([r['method'] for r in rows],[100*r['test_accuracy'] for r in rows],color=['#abb3bd','#899aa9','#477a9d','#156e78','#d59a3b'])
    ax.bar_label(bars,fmt='%.2f%%',padding=3)
    ax.set(ylim=(0,105),ylabel='Test accuracy (%)',title='Digits: one fixed split, 1-nearest neighbor')
    fig.savefig(OUT/'digits_accuracy.png',dpi=180);plt.close(fig)
    result=dict(config=CONFIG,dataset='sklearn load_digits; replacement dataset, not USF/Corel',
                dataset_shape=list(x.shape), dataset_sha256=hashlib.sha256(digits.data.tobytes()+y.tobytes()).hexdigest(),
                train_indices_sha256=hashlib.sha256(train.tobytes()).hexdigest(),
                test_indices_sha256=hashlib.sha256(test.tobytes()).hexdigest(),rows=rows,
                solver_only_diagnostic=diagnostics,mfa2_history=hist,
                mmfa_history=models['MMFA'].history,
                environment=dict(python=platform.python_version(),platform=platform.platform(),
                                 numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__),
                limitations=['Single split; no significance estimate','Digits replaces original datasets',
                             'MFA-1/2 preprocessing differs as documented','No GPU used',
                             'No selection using test accuracy'])
    (OUT/'digits_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print('SOLVER DIAGNOSTIC',json.dumps(diagnostics),flush=True)


if __name__=='__main__':
    main()
