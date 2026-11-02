"""Additional checks with independent constructions and cross-coordinate cases."""
import unittest
import numpy as np
from scipy.linalg import eigvalsh, subspace_angles
from scipy.spatial.distance import pdist
from mfa import (build_graphs, solve_trace_ratio, trace_ratio, fit_mfa, fit_mmfa,
                 laplacian, ged_projection)
from biased import build_biased_graphs, fit_biased, bda_scatters
from gait import cycle_average, sequence_distance, identify


def root_oracle(a,b,d):
    lo,hi=0.,1.
    for _ in range(70):
        mid=(lo+hi)/2
        if eigvalsh(b-mid*(a+b))[-d:].sum()>0:lo=mid
        else:hi=mid
    return (lo+hi)/2


class StrengthenedChecks(unittest.TestCase):
    def test_random_psd_global_roots_and_scales(self):
        for seed in range(12):
            rng=np.random.default_rng(seed)
            x,y=rng.normal(size=(7,7)),rng.normal(size=(7,7))
            a,b=x.T@x+np.eye(7)*.1,y.T@y+np.eye(7)*.1
            d=1+seed%5;expected=root_oracle(a,b,d)
            for scale in [1e-8,1,1e8]:
                sol=solve_trace_ratio(a*scale,b*scale,d,seed=seed)
                self.assertTrue(sol.converged)
                self.assertAlmostEqual(sol.history[-1]['lambda'],expected,places=9)

    def test_repeated_eigenvalues_and_null_directions(self):
        a,b=np.diag([1.,1.,2.,0.]),np.diag([2.,2.,1.,0.])
        for seed in [1,2,3]:
            sol=solve_trace_ratio(a,b,1,seed=seed)
            self.assertAlmostEqual(sol.history[-1]['lambda'],2/3,places=10)
            self.assertAlmostEqual(np.linalg.norm(sol.projection),1.)
            # A particular eigenvector need not be unique for a repeated root.
            self.assertLess(np.linalg.norm(sol.projection[2:]),1e-9)

    def test_graph_permutation_and_class_label_invariance(self):
        rng=np.random.default_rng(8);x=rng.normal(size=(30,4));y=np.repeat([0,1,2],10)
        p=rng.permutation(len(x));g=build_graphs(x,y,3,6);h=build_graphs(x[p],(y[p]+5)*7,3,6)
        np.testing.assert_array_equal(h.within,g.within[np.ix_(p,p)])
        np.testing.assert_array_equal(h.between,g.between[np.ix_(p,p)])

    def test_mfa_coordinate_translation_rotation(self):
        rng=np.random.default_rng(9);x=rng.normal(size=(45,7));y=np.repeat([0,1,2],15)
        q=np.linalg.qr(rng.normal(size=(7,7)))[0];offset=rng.normal(size=7)
        for solver in ['mfa1','mfa2']:
            m=fit_mfa(x,y,ndim=3,k1=3,k2=8,solver=solver,pca_energy=1.)
            t=fit_mfa(x@q+offset,y,ndim=3,k1=3,k2=8,solver=solver,pca_energy=1.)
            np.testing.assert_allclose(pdist(m.transform(x)),pdist(t.transform(x@q+offset)),atol=1e-8)

    def test_thin_mfa2_agrees_with_dense_scatter(self):
        rng=np.random.default_rng(10);x=rng.normal(size=(36,65));y=np.repeat([0,1,2],12)
        m=fit_mfa(x,y,ndim=4,k1=3,k2=10,solver='mfa2')
        a,b=m.graphs.scatters(x-x.mean(axis=0));s=solve_trace_ratio(a,b,4)
        self.assertLess(float(subspace_angles(m.projection,s.projection).max()),1e-7)
        self.assertAlmostEqual(trace_ratio(a,b,m.projection),trace_ratio(a,b,s.projection),places=9)

    def test_shared_pca_diagnostic_matches_literal_pipeline(self):
        rng=np.random.default_rng(11);x=rng.normal(size=(60,10));y=np.repeat([0,1,2],20)
        m=fit_mfa(x,y,3,3,10,solver='mfa2',mfa2_pca_energy=.98)
        centered=x-x.mean(axis=0);_,s,vt=np.linalg.svd(centered,full_matrices=False)
        k=int(np.searchsorted(np.cumsum(s*s)/sum(s*s),.98)+1);q=vt[:k].T
        a,b=m.graphs.scatters(centered@q);oracle=solve_trace_ratio(a,b,3)
        np.testing.assert_allclose(pdist(centered@m.projection),pdist(centered@q@oracle.projection),atol=1e-8)

    def test_matrix_projection_equals_kronecker_map(self):
        rng=np.random.default_rng(12);x=rng.normal(size=(30,4,5));y=np.repeat([0,1,2],10)
        m=fit_mmfa(x,y,(2,2),2,5,iterations=3)
        flat=(x-m.mean).reshape(len(x),-1)@np.kron(m.left,m.right)
        np.testing.assert_allclose(m.transform(x),flat,atol=1e-10)

    def test_biased_graph_hand_calculation(self):
        g=build_biased_graphs(np.array([0,1,3,7,8])[:,None],np.array([1,1,1,0,0],bool),1,2)
        edges=lambda w:set(map(tuple,np.argwhere(np.triu(w,1))))
        self.assertEqual(edges(g.within),{(0,1),(1,2)})
        self.assertEqual(edges(g.between),{(2,3),(2,4)})
        self.assertFalse(g.within[3:].any())

    def test_biased_linear_and_kernel_scatter_pair_sums(self):
        rng=np.random.default_rng(13);x=rng.normal(size=(12,4));pos=np.arange(12)<6
        for gamma in [None,.3]:
            m=fit_biased(x,pos,2,3,8,method='mba',gamma=gamma,ridge=1e-3)
            z=m.transform(x)
            k=x if gamma is None else np.exp(-gamma*((x[:,None]-x[None,:])**2).sum(axis=2))
            a,b=m.graphs.scatters(k)
            for scatter,graph in [(a,m.graphs.within),(b,m.graphs.between)]:
                literal=sum(graph[i,j]*np.sum((z[i]-z[j])**2) for i in range(12) for j in range(12))
                self.assertAlmostEqual(literal,2*np.trace(m.projection.T@scatter@m.projection),places=9)
            if gamma is not None:
                values,vectors=np.linalg.eigh(k);phi=vectors@np.diag(np.sqrt(np.maximum(values,0)))
                w=phi.T@m.projection
                np.testing.assert_allclose(phi@w,z,atol=1e-10)
                np.testing.assert_allclose(np.linalg.norm(w,axis=0),1,atol=1e-10)

    def test_bda_centroid_and_sample_order(self):
        rng=np.random.default_rng(14);x=rng.normal(size=(10,4));pos=np.arange(10)<5
        a,b=bda_scatters(x,pos);mean=x[:5].mean(axis=0)
        aw=sum(np.outer(z-mean,z-mean) for z in x[:5]);bw=sum(np.outer(z-mean,z-mean) for z in x[5:])
        np.testing.assert_allclose(a,aw);np.testing.assert_allclose(b,bw)
        for method in ['mba','bda']:
            m=fit_biased(x,pos,2,2,6,method=method,gamma=.2,ridge=1e-3)
            p=rng.permutation(10);t=fit_biased(x[p],pos[p],2,2,6,method=method,gamma=.2,ridge=1e-3)
            np.testing.assert_allclose(pdist(m.transform(x)),pdist(t.transform(x)),atol=1e-8)

    def test_cycle_averages_discard_incomplete_tail(self):
        f=np.array([[[0,1]],[[1,1]],[[1,0]],[[1,1]],[[0,0]]])
        np.testing.assert_array_equal(cycle_average(f,2),[[[.5,1]],[[1,.5]]])
        with self.assertRaises(ValueError):cycle_average(f,6)
        with self.assertRaises(ValueError):cycle_average(f*.5,2)

    def test_directed_robust_sequence_matching(self):
        probe=np.array([0.,2.,100.])[:,None];gallery=np.array([0.,1.])[:,None]
        self.assertEqual(sequence_distance(probe,gallery),1.)
        self.assertEqual(sequence_distance(gallery,probe),.5)
        top,dist=identify([probe],[gallery,np.array([50.,100.])[:,None]],['A','B'])
        self.assertEqual(top[0,0],'A')
        self.assertEqual(dist.shape,(1,2))

if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):unittest.main(verbosity=2)
