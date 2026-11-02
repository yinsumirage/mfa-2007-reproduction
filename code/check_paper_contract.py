"""Check the fixed paper definition and saved settings without retuning models."""
from pathlib import Path
import hashlib
import json

import numpy as np

from mfa import build_graphs, fit_mfa


ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    original = json.loads((ROOT / 'reproduction-manifest.json').read_text())
    for name, digest in original['algorithm_files_sha256'].items():
        assert sha(ROOT / name) == digest, name
    for name, digest in original['result_files_sha256'].items():
        assert sha(ROOT / name) == digest, name

    # Literal hand calculation: class 0 picks (1,2), class 1 picks (2,1),
    # and class 2 picks (4,3). The undirected union contains two edges.
    x = np.array([0, 1, 4, 5, 10, 11], dtype=float)[:, None]
    g = build_graphs(x, [0, 0, 1, 1, 2, 2], k1=1, k2=1)
    edges = lambda a: set(map(tuple, np.argwhere(np.triu(a, 1))))
    assert edges(g.within) == {(0, 1), (2, 3), (4, 5)}
    assert edges(g.between) == {(1, 2), (3, 4)}

    # Both solver routes must construct exactly the same two graphs.
    rng = np.random.default_rng(1607)
    labels = np.repeat([0, 1, 2], 12)
    train = rng.normal(size=(36, 5)) + labels[:, None] * 0.3
    one = fit_mfa(train, labels, ndim=2, k1=3, k2=50,
                  solver='mfa1', pca_energy=1.0)
    two = fit_mfa(train, labels, ndim=2, k1=3, k2=50, solver='mfa2')
    np.testing.assert_array_equal(one.graphs.within, two.graphs.within)
    np.testing.assert_array_equal(one.graphs.between, two.graphs.between)
    assert two.trace_solution.converged

    classification = json.loads((ROOT / 'results/round2/classification.json').read_text())
    visual = json.loads((ROOT / 'results/visual_comparison/comparison.json').read_text())
    assert classification['config']['k1'] == visual['config']['mfa_k1'] == 3
    assert classification['config']['k2'] == visual['config']['mfa_k2'] == 50
    assert classification['config']['seeds'] == [16, 17, 18]

    report = {
        'date': '2026-11-02',
        'status': 'passed',
        'source': {'doi': '10.1109/TIP.2007.906769',
                   'graph_definition_pdf_page': 3,
                   'graph_equations': [5, 6],
                   'solver_names_and_procedure_pdf_page': 7},
        'canonical_graph_rule': 'Each class selects k2 closest pairs against all other classes together; take the undirected union.',
        'classification_profile': {'k1': 3, 'k2': 50, 'seeds': [16, 17, 18],
                                   'vector_output_dim': 16, 'matrix_output_shape': [4, 4]},
        'manual_pair_fixture': 'passed',
        'mfa1_and_mfa2_use_identical_graphs': True,
        'mfa2_fixture_converged': True,
        'prior_core_code_and_result_hashes_unchanged': True,
        'original_benchmarks_reproduced': False,
        'engineering_conventions': ['stable distance ties', 'numerical rank threshold',
                                    'eigenvector normalization', 'residual-based stopping'],
    }
    print(json.dumps({'status': report['status'], 'k1': 3, 'k2': 50,
                      'same_graphs_for_both_solvers': True, 'core_and_results_unchanged': True}))


if __name__ == '__main__':
    main()
