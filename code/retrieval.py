"""Feedback protocol with a strict boundary between oracle and fitting code."""
import numpy as np
from scipy.spatial.distance import cdist
from biased import fit_biased


def ranked(x,query,model=None):
    z=x if model is None else model.transform(x)
    d=cdist(z[[query]],z,'sqeuclidean')[0]
    order=np.argsort(d,kind='stable')
    return order[order!=query]


def feedback_oracle(order,labels,query_label,seen,top=40,per_class=5,display_unseen=True):
    """Reveal labels only in the displayed top40; never fill elsewhere.

    The primary protocol displays the first 40 unseen items. False retains the
    original diagnostic rule: take 40 first, then remove already seen items.
    """
    candidates=([int(i) for i in order if int(i) not in seen][:top] if display_unseen
                else [int(i) for i in order[:top] if int(i) not in seen])
    positive=[i for i in candidates if labels[i]==query_label][:per_class]
    negative=[i for i in candidates if labels[i]!=query_label][:per_class]
    return positive,negative


def run_query(x,labels,query,method,*,rounds=9,gamma=None,ridge=1e-3,callback=None,
              display_unseen=True):
    # Only this oracle/evaluation wrapper can see complete labels. fit_biased
    # receives an explicit subset and the labels the simulated user revealed.
    seen={int(query):True};model=None;history=[]
    for step in range(rounds+1):
        order=ranked(x,query,model)
        unseen=np.array([i for i in order if int(i) not in seen])
        row=dict(round=step,query=int(query),method=method,feedback_total=len(seen),
                 feedback_positive=sum(seen.values()),feedback_negative=len(seen)-sum(seen.values()))
        for k in [20,40,60]:
            row[f'p{k}_all']=float(np.mean(labels[order[:k]]==labels[query]))
            row[f'p{k}_unseen']=float(np.mean(labels[unseen[:k]]==labels[query])) if len(unseen)>=k else None
        history.append(row)
        if callback:callback(row)
        if step==rounds:break
        pos,neg=feedback_oracle(order,labels,labels[query],seen,display_unseen=display_unseen)
        row.update(selected_positive=pos,selected_negative=neg,feedback_shortfall=len(pos)<5 or len(neg)<5)
        for i in pos:seen[i]=True
        for i in neg:seen[i]=False
        ids=np.array(list(seen));mask=np.array(list(seen.values()),dtype=bool)
        if mask.sum()<2 or (~mask).sum()<1:
            row['fit_status']='retained previous model: insufficient positive/negative feedback'
            continue
        # Feasibility clipping is declared in the plan. It uses revealed counts,
        # not test accuracy and not labels outside the feedback list.
        k1=min(3,int(mask.sum())-1);k2=min(20,int(mask.sum()*(~mask).sum()))
        model=fit_biased(x[ids],mask,ndim=2,k1=k1,k2=k2,method=method,
                         gamma=gamma,ridge=ridge)
        row.update(fit_status='ok',k1=k1,k2=k2,alpha=model.alpha,training_ids=ids.tolist())
    return history


def run_heldout_query(x,labels,query,feedback_pool,evaluation_pool,method,*,
                      rounds=5,gamma=None,ridge=1e-3):
    """Only feedback-pool labels enter fitting; evaluation pool stays fixed."""
    feedback_pool=np.asarray(feedback_pool,dtype=int)
    evaluation_pool=np.asarray(evaluation_pool,dtype=int)
    if np.intersect1d(feedback_pool,evaluation_pool).size or query in feedback_pool:
        raise ValueError('Feedback and evaluation sets must be disjoint')
    evaluation_pool=evaluation_pool[evaluation_pool!=query]
    seen={int(query):True};model=None;history=[]
    for step in range(rounds+1):
        order=ranked(x,query,model)
        evaluation=order[np.isin(order,evaluation_pool)]
        row=dict(round=step,query=int(query),method=method,feedback_total=len(seen),
                 evaluation_count=len(evaluation),feedback_positive=sum(seen.values()))
        for k in [20,40,60]:row[f'p{k}']=float(np.mean(labels[evaluation[:k]]==labels[query]))
        history.append(row)
        if step==rounds:break
        available=order[np.isin(order,feedback_pool)]
        pos,neg=feedback_oracle(available,labels,labels[query],seen)
        row.update(selected_positive=pos,selected_negative=neg,feedback_shortfall=len(pos)<5 or len(neg)<5)
        for i in pos:seen[i]=True
        for i in neg:seen[i]=False
        ids=np.array(list(seen));mask=np.array(list(seen.values()),bool)
        assert not np.intersect1d(ids,evaluation_pool).size
        row['training_ids']=ids.tolist()
        if mask.sum()<2 or (~mask).sum()<1:
            row['fit_status']='retained previous model: insufficient feedback'
            continue
        model=fit_biased(x[ids],mask,ndim=2,k1=min(3,int(mask.sum())-1),
                         k2=min(20,int(mask.sum()*(~mask).sum())),method=method,
                         gamma=gamma,ridge=ridge)
        row.update(fit_status='ok',alpha=model.alpha)
    return history
