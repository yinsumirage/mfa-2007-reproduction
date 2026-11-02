"""Independent mathematical checks; no accuracy-based tuning."""
import unittest
import numpy as np
from scipy.linalg import eigvalsh

from mfa import (build_graphs, laplacian, solve_trace_ratio, trace_ratio,
                 fit_mfa, fit_mmfa, matrix_scatter)


class MathematicalChecks(unittest.TestCase):
    def test_hand_calculated_class_margin_pairs(self):
        x = np.array([0, 1, 4, 5, 10, 11], float)[:, None]
        g = build_graphs(x, [0, 0, 1, 1, 2, 2], 1, 1)
        edges = lambda s: set(map(tuple, np.argwhere(np.triu(s, 1))))
        self.assertEqual(edges(g.within), {(0, 1), (2, 3), (4, 5)})
        self.assertEqual(edges(g.between), {(1, 2), (3, 4)})

    def test_ties_are_deterministic(self):
        g = build_graphs(np.array([0, 0, 1, 1])[:, None], [0, 0, 1, 1], 1, 1)
        self.assertEqual(set(map(tuple, np.argwhere(np.triu(g.between, 1)))), {(0, 2)})

    def test_pairwise_laplacian_identity(self):
        rng = np.random.default_rng(5)
        x = rng.normal(size=(12, 4))
        g = build_graphs(x, np.repeat([0, 1, 2], 4), 2, 3)
        w = rng.normal(size=(4, 2))
        z = x @ w
        for s in [g.within, g.between]:
            literal = sum(s[i, j] * np.sum((z[i] - z[j]) ** 2)
                          for i in range(12) for j in range(12))
            matrix = 2 * np.trace(w.T @ x.T @ laplacian(s) @ x @ w)
            self.assertAlmostEqual(literal, matrix, places=10)
            self.assertGreaterEqual(eigvalsh(laplacian(s))[0], -1e-10)

    def test_scalar_solution_against_angle_search(self):
        a = np.array([[4., 1.], [1., 2.]])
        b = np.array([[2., -.3], [-.3, 5.]])
        sol = solve_trace_ratio(a, b, 1)
        angles = np.linspace(0, np.pi, 50001)
        directions = np.stack([np.cos(angles), np.sin(angles)])
        ratios = np.einsum('in,ij,jn->n', directions, a, directions) / np.einsum(
            'in,ij,jn->n', directions, b, directions)
        self.assertLessEqual(abs(trace_ratio(a, b, sol.projection) - ratios.min()), 1e-7)

    def test_multidimensional_against_independent_root_bisection(self):
        rng = np.random.default_rng(7)
        x, y = rng.normal(size=(6, 6)), rng.normal(size=(6, 6))
        a, b = x.T @ x + np.eye(6), y.T @ y + np.eye(6)
        lo, hi, d = 0., 1., 3
        for _ in range(100):
            mid = (lo + hi) / 2
            if eigvalsh(b - mid * (a + b))[-d:].sum() > 0:
                lo = mid
            else:
                hi = mid
        for seed in [3, 19]:
            sol = solve_trace_ratio(a, b, d, seed=seed)
            self.assertTrue(sol.converged)
            self.assertAlmostEqual(sol.history[-1]['lambda'], (lo + hi) / 2, places=9)
            np.testing.assert_allclose(sol.projection.T @ sol.projection, np.eye(d), atol=1e-10)
            self.assertGreaterEqual(np.diff([h['lambda'] for h in sol.history]).min(), -1e-10)
            self.assertLess(sol.spectral_residual, 1e-9)

    def test_singular_total_scatter_and_invalid_dimensions(self):
        a, b = np.diag([2., 1., 0., 0.]), np.diag([1., 3., 0., 0.])
        sol = solve_trace_ratio(a, b, 1)
        self.assertEqual(sol.retained_rank, 2)
        self.assertLess(np.linalg.norm(sol.projection[2:]), 1e-12)
        with self.assertRaises(ValueError):
            solve_trace_ratio(a, b, 3)
        with self.assertRaises(ValueError):
            solve_trace_ratio(a * 0, b * 0, 1)

    def test_input_guards(self):
        for labels in [[0]*6, [0, 0, 1, 1, 2]]:
            with self.assertRaises(ValueError):
                build_graphs(np.arange(6)[:, None], labels, 1, 1)
        with self.assertRaises(ValueError):
            build_graphs(np.array([[np.nan], [1], [2], [3]]), [0, 0, 1, 1], 1, 1)
        with self.assertRaises(ValueError):
            build_graphs(np.arange(6)[:, None], [0, 0, 1, 1, 2, 2], 2, 1)

    def test_out_of_sample_transformation(self):
        rng = np.random.default_rng(22)
        train = rng.normal(size=(36, 5))
        labels = np.repeat([0, 1, 2], 12)
        test = rng.normal(size=(9, 5))
        for solver in ['mfa1', 'mfa2']:
            model = fit_mfa(train, labels, ndim=2, k1=2, k2=8, solver=solver)
            old_mean = model.mean.copy()
            np.testing.assert_allclose(model.transform(test), (test-old_mean) @ model.projection)
            self.assertEqual(model.graphs.within.shape, (len(train), len(train)))
            model.transform(test * 1000)
            np.testing.assert_array_equal(model.mean, old_mean)

    def test_matrix_scatter_against_literal_pair_sum(self):
        rng = np.random.default_rng(2)
        images = rng.normal(size=(9, 3, 4))
        g = build_graphs(images.reshape(9, -1), np.repeat([0, 1, 2], 3), 1, 2)
        u, v = rng.normal(size=(3, 2)), rng.normal(size=(4, 2))
        for graph in [g.within, g.between]:
            literal_r, literal_l = np.zeros((4, 4)), np.zeros((3, 3))
            for i in range(9):
                for j in range(9):
                    diff = images[i] - images[j]
                    literal_r += graph[i, j] * (u.T @ diff).T @ (u.T @ diff)
                    literal_l += graph[i, j] * (diff @ v) @ (diff @ v).T
            np.testing.assert_allclose(matrix_scatter(images, graph, u, 'right'), literal_r, atol=1e-10)
            np.testing.assert_allclose(matrix_scatter(images, graph, v, 'left'), literal_l, atol=1e-10)

    def test_matrix_projection_on_new_images(self):
        rng = np.random.default_rng(4)
        images = rng.normal(size=(30, 4, 5))
        model = fit_mmfa(images, np.repeat([0, 1, 2], 10), (2, 2), 2, 5, iterations=3)
        new = rng.normal(size=(3, 4, 5))
        literal = np.stack([model.left.T @ (im-model.mean) @ model.right for im in new]).reshape(3, -1)
        np.testing.assert_allclose(model.transform(new), literal, atol=1e-10)


if __name__ == '__main__':
    unittest.main(verbosity=2)
