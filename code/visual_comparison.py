"""Controlled 2D Digits subspace comparison for the presentation.

This is a NEW teaching diagnostic, not a rerun/replacement of round2. Classes,
query, splits, graph counts and shared preprocessing are fixed before results.
Samples are rows. Native learned directions are saved, and the main views use
Euclidean-orthonormal bases of the SAME learned subspaces (QR). This removes
arbitrary directional scaling for the common projected-Euclidean 1-NN comparison;
it differs from each method's native metric normalization, notably LDA/LPP/GED.
"""
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).parent/'.mpl-cache'))
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import base64,hashlib,io,json
import numpy as np
from scipy.linalg import eigh
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import connected_components
from sklearn.datasets import load_digits
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from mfa import build_graphs,laplacian,ged_projection

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'results/visual_comparison'
ASSETS=ROOT/'figures'
CONFIG=dict(date='2026-11-02',dataset='scikit-learn Digits',classes=[0,1,2],
            split_seed=16,split_source='results/round2/classification_splits.npz',
            train_key='Digits_16_train',test_key='Digits_16_test',
            image_scaling='pixels / 16',shared_preprocessing='train mean, PCA retaining 98% variance',
            graph_input='original 64D training pixels / 16, before PCA; no test samples',
            scatter_input='shared centered PCA98 training features',
            output_dimensions=2,lpp_neighbors=5,mfa_k1=3,mfa_k2=50,
            ged_ridge=0.0,numerical_rank_relative_tolerance=1e-12,
            primary_geometry='Euclidean-orthonormal learned 2D subspace via QR in shared PCA coordinates',
            native_geometry='method-specific native coordinate normalization, saved separately',
            query_rule='smallest original test ID with label 1, fixed before fitting',
            selection='no parameter sweep; no test-based method/query/class selection',
            scope='controlled three-class 2D teaching diagnostic, not original-paper or prior round2 benchmark')
COLORS={0:'#152B3A',1:'#007F89',2:'#D45A47'}
MARKERS={0:'o',1:'^',2:'s'}
NAMES={'pca':'PCA','lda':'LDA','lpp':'LPP','mfa':'MFA-1'}
FORMULAS={
 'pca':{'formula':'maximize tr(W^T S_T W), W^T W = I','meaning':'Retain overall variance; labels do not enter fitting.'},
 'lda':{'formula':'maximize tr[(W^T S_W W)^(-1) W^T S_B W]','meaning':'Separate global class means relative to within-class scatter.'},
 'lpp':{'formula':'minimize tr(W^T X^T L X W), W^T X^T D X W = I','meaning':'Preserve unlabeled neighboring samples.'},
 'mfa':{'formula':'maximize tr[(W^T A W)^(-1) W^T B W]','meaning':'Use labeled within-class neighbors and class-versus-rest margin pairs; MFA-1 GED surrogate.'}}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def sym(x):return (x+x.T)/2

def build_lpp_graph(x,k=5):
    """Unlabeled binary kNN graph, OR symmetrized, deterministic row-order ties."""
    if x.ndim!=2 or not np.isfinite(x).all() or len(x)<=k:raise ValueError('Invalid LPP input')
    d=cdist(x,x,'sqeuclidean');np.fill_diagonal(d,np.inf)
    g=np.zeros((len(x),len(x)))
    for i in range(len(x)):
        j=np.argsort(d[i],kind='stable')[:k];g[i,j]=g[j,i]=1
    return g


def canonical_qr(w):
    q=np.linalg.qr(w,mode='reduced')[0]
    for j in range(q.shape[1]):
        k=np.argmax(np.abs(q[:,j]))
        if q[k,j]<0:q[:,j]*=-1
    return q


def pair_scatter(x,g):
    # Independent unordered-edge sum, equal to X^T L X for symmetric adjacency.
    out=np.zeros((x.shape[1],x.shape[1]))
    for i,j in np.argwhere(np.triu(g,1)):
        d=x[i]-x[j];out+=g[i,j]*np.outer(d,d)
    return out


def normalized_residual(a,b,w,lam):
    delta=a@w-(b@w)*lam[None,:]
    scale=(np.linalg.norm(a,2)+np.linalg.norm(b,2)*np.max(np.abs(lam)))*np.linalg.norm(w)
    return float(np.linalg.norm(delta)/max(scale,np.finfo(float).tiny))


def classify(train_z,test_z,train_y,test_y):
    ds=cdist(test_z,train_z,'sqeuclidean');ix=ds.argmin(1);pred=train_y[ix]
    confusion=np.zeros((3,3),dtype=int)
    for truth,guess in zip(test_y,pred):confusion[truth,guess]+=1
    per_class={str(c):{'correct':int(np.sum(pred[test_y==c]==c)),
                       'total':int(np.sum(test_y==c)),
                       'accuracy':float(np.mean(pred[test_y==c]==c))} for c in CONFIG['classes']}
    return dict(predictions=pred.tolist(),correct=int(np.sum(pred==test_y)),total=len(pred),
                accuracy=float(np.mean(pred==test_y)),confusion=confusion.tolist(),per_class=per_class)


def thumbnail(pixels):
    pic=Image.fromarray(np.round(pixels*255/16).astype(np.uint8)).resize((32,32),Image.Resampling.NEAREST)
    buf=io.BytesIO();pic.save(buf,format='PNG')
    return 'data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode('ascii')


def graph_stats(g,x):
    a=sym(x.T@laplacian(g)@x)
    return dict(edges=int(np.count_nonzero(np.triu(g,1))),connected_components=int(connected_components(g,directed=False,return_labels=False)),
                min_scatter_eigenvalue=float(eigh(a,eigvals_only=True)[0]),
                scatter_pair_identity_relative_error=float(np.linalg.norm(a-pair_scatter(x,g))/max(np.linalg.norm(a),1e-300)))


def draw_scatter(ax,m,result,legend=False):
    tz=np.array(m['train_coordinates']);vz=np.array(m['test_coordinates']);ty=np.array(result['train_labels']);vy=np.array(result['test_labels'])
    for c in CONFIG['classes']:
        ax.scatter(*tz[ty==c].T,s=32,c=COLORS[c],marker=MARKERS[c],alpha=.24,linewidths=0)
        ax.scatter(*vz[vy==c].T,s=56,facecolors='none',edgecolors=COLORS[c],marker=MARKERS[c],linewidths=1.25,label=f'Digit {c}')
    q=np.array(m['query_coordinates']);ax.scatter(*q,s=185,marker='*',c='#B48C36',edgecolors='#152B3A',linewidths=1.1,zorder=6)
    ax.set_title(f"{m['name']}   test 1-NN: {100*m['accuracy']:.1f}%",fontweight='bold',fontsize=17,pad=14)
    ax.set(xlabel='Projection coordinate 1 (a.u.)',ylabel='Projection coordinate 2 (a.u.)',
           xlim=result['plot_bounds']['x'],ylim=result['plot_bounds']['y'])
    ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.13);ax.set_axisbelow(True)
    ax.spines[['top','right']].set_visible(False)
    if legend:ax.legend(frameon=False,loc='upper right',fontsize=13)


def figures(result,images):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':13,'axes.labelcolor':'#152B3A','text.color':'#152B3A','axes.edgecolor':'#B8C4C8','figure.facecolor':'#F8F7F3','axes.facecolor':'#F8F7F3'})
    fig,axs=plt.subplots(2,2,figsize=(13.2,10.3),layout='constrained')
    for ax,m in zip(axs.flat,result['methods']):
        draw_scatter(ax,m,result)
        ax.legend(frameon=False,loc='upper right',fontsize=12,markerscale=.8)
    fig.suptitle('Same Digits images, four learned 2D subspaces',fontsize=21,fontweight='bold')
    fig.supxlabel('Filled: training. Hollow: held-out test. Gold star: fixed query ID 56 (digit 1).',fontsize=13)
    fig.savefig(ASSETS/'comparison.png',dpi=170);plt.close(fig)
    for m in result['methods']:
        fig,ax=plt.subplots(figsize=(7.6,5.5),layout='constrained');draw_scatter(ax,m,result,legend=True)
        fig.savefig(ASSETS/f"{m['id']}.png",dpi=180);plt.close(fig)
    # Same fixed query, 6 training neighbors under each learned metric.
    fig,axs=plt.subplots(4,7,figsize=(12.8,6.5),layout='constrained')
    qid=result['query']['id']
    for row,m in enumerate(result['methods']):
        for col,(idx,label) in enumerate([(qid,result['query']['label'])]+[(n['id'],n['label']) for n in m['neighbors']]):
            ax=axs[row,col];ax.imshow(images[idx],cmap='gray',vmin=0,vmax=16,interpolation='nearest');ax.set_xticks([]);ax.set_yticks([])
            if col==0:
                ax.set_ylabel(m['name'],rotation=0,ha='right',va='center',labelpad=18,fontweight='bold',fontsize=15)
                ax.set_title('Fixed query' if row==0 else '',fontsize=12)
            else:ax.set_title(f"#{col}   digit {label}",fontsize=11,color=COLORS[label])
            ax.set_xlabel(f'ID {idx}',fontsize=10)
            for sp in ax.spines.values():sp.set_visible(False)
    fig.suptitle('The same query retrieves different training neighbors',fontsize=19,fontweight='bold')
    fig.savefig(ASSETS/'neighbors.png',dpi=180);plt.close(fig)
    fig=plt.figure(figsize=(11.5,3.7));gs=fig.add_gridspec(1,2,width_ratios=[1,3],left=.035,right=.985,bottom=.06,top=.73);ax=fig.add_subplot(gs[0,0]);ax.imshow(images[qid],cmap='gray',vmin=0,vmax=16,interpolation='nearest');ax.axis('off');ax.set_title('Query ID 56, digit 1',fontsize=14,pad=12)
    ax=fig.add_subplot(gs[0,1]);ax.axis('off')
    rows=[[m['name'],f"{m['query_coordinates'][0]:.3f}",f"{m['query_coordinates'][1]:.3f}"] for m in result['methods']]
    table=ax.table(cellText=rows,colLabels=['Method','Coordinate 1','Coordinate 2'],cellLoc='center',loc='center');table.auto_set_font_size(False);table.set_fontsize(17);table.scale(1,2)
    for (row,col),cell in table.get_celld().items():cell.set_edgecolor('#D7E0E1');cell.set_facecolor('#E4EDED' if row==0 else '#F8F7F3')
    fig.suptitle('The image becomes two coordinates in a learned feature space',fontsize=20,fontweight='bold',y=.98)
    fig.savefig(ASSETS/'image_to_coordinates.png',dpi=170);plt.close(fig)


def main():
    OUT.mkdir(parents=True,exist_ok=True);ASSETS.mkdir(parents=True,exist_ok=True)
    source=ROOT/CONFIG['split_source'];d=load_digits();x=d.data.astype(float)/16;y=d.target
    split=np.load(source)
    train=split[CONFIG['train_key']];test=split[CONFIG['test_key']]
    train=train[np.isin(y[train],CONFIG['classes'])];test=test[np.isin(y[test],CONFIG['classes'])]
    query_id=int(min(test[y[test]==1]));query_index=int(np.flatnonzero(test==query_id)[0])
    assert query_id==56 and not np.intersect1d(train,test).size
    plan={**CONFIG,'train_count':len(train),'test_count':len(test),'query_id':query_id,
          'source_split_sha256':sha(source),'code_sha256':sha(__file__),
          'input_data_sha256':hashlib.sha256(x.tobytes()+y.tobytes()).hexdigest()}
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    # No method fitting or test metrics occurs before the fixed plan is saved.
    mean=x[train].mean(0);center=x[train]-mean
    _,ss,vt=np.linalg.svd(center,full_matrices=False)
    rank=min(int(np.searchsorted(np.cumsum(ss**2)/np.sum(ss**2),.98)+1),int(np.sum(ss>ss[0]*1e-12)))
    common=vt[:rank].T;tr=center@common;te=(x[test]-mean)@common;ty=y[train];vy=y[test]
    native={'pca':np.eye(rank)[:,:2]};checks={}
    checks['common_pca_rank']=rank;checks['retained_training_variance']=float(np.sum(ss[:rank]**2)/np.sum(ss**2))
    st=tr.T@tr;sw=np.zeros_like(st);sb=np.zeros_like(st)
    for c in CONFIG['classes']:
        block=tr[ty==c];mu=block.mean(0);sw+=(block-mu).T@(block-mu);sb+=len(block)*np.outer(mu,mu)
    checks['lda_total_scatter_identity_error']=float(np.linalg.norm(st-sw-sb)/np.linalg.norm(st))
    lda=LinearDiscriminantAnalysis(solver='svd',n_components=2).fit(tr,ty)
    native['lda']=lda.scalings_[:,:2]
    # Standard unlabeled LPP on original input distances, with shared PCA scatter.
    g=build_lpp_graph(x[train],CONFIG['lpp_neighbors']);a=sym(tr.T@laplacian(g)@tr);b=sym(tr.T@np.diag(g.sum(1))@tr)
    bv=eigh(b,eigvals_only=True);assert bv[0]>1e-12*bv[-1]
    val,vec=eigh(a,b)
    threshold=max(abs(val[-1]),1)*1e-12
    projected=tr@vec
    variations=np.var(projected,axis=0)
    # LPP does NOT automatically drop its first direction. Nonconstant near-zero
    # modes from graph components remain valid. Only zero-variance coordinates
    # would be trivial. Full-rank centered PCA has none here.
    valid=np.flatnonzero((val>=-threshold)&(variations>1e-12*np.max(variations)))
    if len(valid)<2:raise ValueError('Too few nonconstant LPP directions')
    selected=valid[:2];native['lpp']=vec[:,selected]
    checks['lpp']=dict(**graph_stats(g,tr),denominator_eigenvalues=bv.tolist(),
        all_generalized_eigenvalues=val.tolist(),selected_indices=selected.tolist(),selected_eigenvalues=val[selected].tolist(),
        selected_coordinate_variances=variations[selected].tolist(),
        near_zero_modes=[{'index':int(i),'eigenvalue':float(val[i]),'coordinate_variance':float(variations[i])} for i in np.flatnonzero(abs(val)<=threshold)],
        generalized_eigen_residual=normalized_residual(a,b,native['lpp'],val[selected]),
        selection='smallest eigenvalues with nonconstant coordinates; no unconditional first-vector skip')
    mf=build_graphs(x[train],ty,CONFIG['mfa_k1'],CONFIG['mfa_k2']);wa,ba=mf.scatters(tr)
    av=eigh(wa,eigvals_only=True);assert av[0]>1e-12*av[-1]
    native['mfa']=ged_projection(wa,ba,2,ridge=0.0)
    mval,mvec=eigh(ba,wa);mw=mvec[:,-2:][:,::-1]
    checks['mfa']=dict(within=graph_stats(mf.within,tr),between=graph_stats(mf.between,tr),
        denominator_eigenvalues=av.tolist(),selected_eigenvalues=mval[-2:][::-1].tolist(),
        generalized_eigen_residual=normalized_residual(ba,wa,mw,mval[-2:][::-1]),
        equivalent_core_ged_subspace_error=float(np.linalg.norm(canonical_qr(mw)@canonical_qr(mw).T-canonical_qr(native['mfa'])@canonical_qr(native['mfa']).T)))
    result={'config':plan,'common_preprocessing':{'rank':rank,'mean':mean.tolist(),'pca_basis':common.tolist(),'retained_variance':checks['retained_training_variance']},
        'train_ids':train.tolist(),'train_labels':ty.tolist(),'test_ids':test.tolist(),'test_labels':vy.tolist(),
        'query':{'id':query_id,'label':int(y[query_id]),'test_index':query_index},'methods':[]}
    for key in ['pca','lda','lpp','mfa']:
        w=canonical_qr(native[key]);trz=tr@w;tez=te@w
        metrics=classify(trz,tez,ty,vy);native_metrics=classify(tr@native[key],te@native[key],ty,vy)
        ds=cdist(tez[[query_index]],trz,'sqeuclidean')[0];nearest=np.argsort(ds,kind='stable')[:6]
        n=[{'id':int(train[j]),'label':int(ty[j]),'distance':float(np.sqrt(ds[j])),'is_same_class':bool(ty[j]==y[query_id])} for j in nearest]
        checks[key+'_orthonormal_error']=float(np.linalg.norm(w.T@w-np.eye(2)))
        result['methods'].append({'id':key,'name':NAMES[key],**FORMULAS[key],**metrics,
             'train_coordinates':trz.tolist(),'test_coordinates':tez.tolist(),'query_coordinates':tez[query_index].tolist(),
             'neighbors':n,'projection_original_64d':(common@w).tolist(),
             'native_train_coordinates':(tr@native[key]).tolist(),'native_test_coordinates':(te@native[key]).tolist(),
             'native_metrics':native_metrics,'geometry_note':'QR changes axis scaling/geometry relative to native normalization. It preserves the learned 2D subspace.'})
    # An explicitly posthoc example for explaining model disagreement. It never
    # affects the fixed query, classes, parameters, fitting or aggregate metrics.
    disagree=[i for i in range(len(test)) if len({m['predictions'][i] for m in result['methods']})>1]
    if disagree:
        ii=min(disagree,key=lambda i:int(test[i]));detail=[]
        for m in result['methods']:
            ds=cdist(np.array(m['test_coordinates'])[[ii]],np.array(m['train_coordinates']),'sqeuclidean')[0]
            j=int(np.argmin(ds))
            detail.append({'id':m['id'],'prediction':int(m['predictions'][ii]),
                           'neighbor_1':{'id':int(train[j]),'label':int(ty[j]),'distance':float(np.sqrt(ds[j]))},
                           'coordinates':m['test_coordinates'][ii]})
        result['diagnostic_example']={'id':int(test[ii]),'label':int(vy[ii]),'test_index':ii,
            'selection_rule':'smallest original test ID where at least two method predictions differ',
            'selected_after_fit':True,'purpose':'illustration only, not model/query/parameter selection or aggregate evaluation',
            'methods':detail}
    else:result['diagnostic_example']=None
    allcoords=np.concatenate([np.array(m['train_coordinates']) for m in result['methods']]+[np.array(m['test_coordinates']) for m in result['methods']])
    lo=allcoords.min(0);hi=allcoords.max(0);pad=.08*(hi-lo);result['plot_bounds']={'x':[float(lo[0]-pad[0]),float(hi[0]+pad[0])],'y':[float(lo[1]-pad[1]),float(hi[1]+pad[1])]}
    assert max(checks[k+'_orthonormal_error'] for k in native)<1e-12
    assert checks['lda_total_scatter_identity_error']<1e-12
    assert checks['lpp']['scatter_pair_identity_relative_error']<1e-12
    assert checks['mfa']['within']['scatter_pair_identity_relative_error']<1e-12
    assert checks['mfa']['between']['scatter_pair_identity_relative_error']<1e-12
    assert checks['lpp']['generalized_eigen_residual']<1e-10 and checks['mfa']['generalized_eigen_residual']<1e-10
    checks.update(train_test_disjoint=True,query_fixed_before_fit=True,train_only_preprocessing=True,graph_uses_train_only=True,
                  labels_used_for_fitting={'pca':False,'lda':True,'lpp':False,'mfa':True},
                  no_test_based_selection=True,all_coordinate_values_finite=bool(np.isfinite(allcoords).all()))
    result['checks']=checks
    (OUT/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    thumbs={str(int(i)):thumbnail(d.images[i]) for i in np.concatenate([train,test])}
    (OUT/'thumbnails.json').write_text(json.dumps(thumbs)+'\n')
    np.savez(OUT/'projections.npz',mean=mean,pca_basis=common,**{k:common@canonical_qr(w) for k,w in native.items()})
    figures(result,d.images)
    validation={'date':'2026-11-02','train_count':len(train),'test_count':len(test),'query_id':query_id,'checks':'passed',
        'comparison_sha256':sha(OUT/'comparison.json'),'thumbnails_sha256':sha(OUT/'thumbnails.json'),
        'figure_sha256':{p.name:sha(p) for p in ASSETS.glob('*.png')},
        'summary':[{k:m[k] for k in ['id','name','accuracy','correct','total']} for m in result['methods']],
        'scope':CONFIG['scope'],'metric_note':CONFIG['primary_geometry']}
    (OUT/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    print(json.dumps(validation,indent=2))

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
