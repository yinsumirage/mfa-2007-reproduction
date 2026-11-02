"""Feedback isolation and descriptor edge cases, independent of accuracy."""
import tempfile,unittest
from pathlib import Path
import numpy as np
from PIL import Image
from retrieval import feedback_oracle,run_heldout_query
from retrieval_features import descriptor
from biased import fit_biased


class RetrievalChecks(unittest.TestCase):
    def test_top40_visibility_and_no_repeated_feedback(self):
        order=np.arange(60);labels=np.arange(60)%2;seen={i:True for i in range(30)}
        p,n=feedback_oracle(order,labels,0,seen,top=10,per_class=5)
        self.assertEqual(p,[30,32,34,36,38]);self.assertEqual(n,[31,33,35,37,39])
        p,n=feedback_oracle(order,labels,0,seen,top=10,per_class=5,display_unseen=False)
        self.assertEqual(p,[]);self.assertEqual(n,[])

    def test_feedback_shortfall_never_uses_hidden_outside_top40(self):
        order=np.arange(80);labels=np.r_[np.ones(40),np.zeros(40)]
        p,n=feedback_oracle(order,labels,0,{},top=40)
        self.assertEqual(p,[]);self.assertEqual(n,[0,1,2,3,4])

    def test_heldout_training_never_contains_evaluation_candidates(self):
        rng=np.random.default_rng(3);x=rng.normal(size=(36,5));labels=np.repeat([0,1,2],12)
        pool=np.r_[0:8,12:20,24:32];evaluation=np.setdiff1d(np.arange(36),pool);q=8
        rr=run_heldout_query(x,labels,q,pool,evaluation,'mba',rounds=2)
        for row in rr:
            self.assertEqual(row['evaluation_count'],11)
            if 'training_ids' in row:
                ids=np.array(row['training_ids']);self.assertFalse(np.intersect1d(ids,evaluation[evaluation!=q]).size)
                self.assertIn(q,ids)
                self.assertTrue(set(ids)-{q} <= set(pool))

    def test_heldout_label_changes_do_not_change_feedback_or_model(self):
        rng=np.random.default_rng(5);x=rng.normal(size=(40,5));labels=np.repeat([0,1],20)
        pool=np.r_[0:12,20:32];evaluation=np.setdiff1d(np.arange(40),pool);q=12
        altered=labels.copy();ids=evaluation[evaluation!=q];altered[ids]=1-altered[ids]
        r1=run_heldout_query(x,labels,q,pool,evaluation,'mba',rounds=2)
        r2=run_heldout_query(x,altered,q,pool,evaluation,'mba',rounds=2)
        for a,b in zip(r1[:-1],r2[:-1]):
            self.assertEqual(a['selected_positive'],b['selected_positive'])
            self.assertEqual(a['selected_negative'],b['selected_negative'])
            self.assertEqual(a.get('alpha'),b.get('alpha'))

    def test_constant_image_features_remain_finite(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'white.png';Image.fromarray(np.full((16,16,3),255,np.uint8)).save(p)
            f=descriptor(p)
        self.assertEqual(f.shape,(285,));self.assertTrue(np.isfinite(f).all())
        self.assertEqual(np.count_nonzero(f[:256]),1);self.assertTrue(np.array_equal(f[-5:],np.zeros(5)))

    def test_explicit_ridge_required_and_kernel_projection_ged_residual(self):
        rng=np.random.default_rng(7);x=rng.normal(size=(12,4));p=np.arange(12)<6
        with self.assertRaises(ValueError):fit_biased(x,p,ridge=0)
        m=fit_biased(x,p,2,2,8,gamma=.3,ridge=1e-3)
        k=np.exp(-.3*((x[:,None]-x[None,:])**2).sum(axis=2));a,b=m.graphs.scatters(k)
        metric=a+m.alpha*np.eye(len(a))
        for w in m.projection.T:
            val=w@b@w/(w@metric@w)
            self.assertLess(np.linalg.norm(b@w-val*metric@w)/max(np.linalg.norm(b@w),1e-15),1e-8)

if __name__=='__main__':unittest.main(verbosity=2)
