"""Graph metrics of a walk over hidden-state clusters (used by exp30 / exp31)."""
import numpy as np
from collections import Counter


def graph_feats(lab, C):
    """lab: cluster label per step; C: [K, d] unit-norm centroids of the model"""
    T = len(lab); out = {}
    nodes = set(lab); V = len(nodes)
    tr = list(zip(lab[:-1], lab[1:])); moves = [(a, b) for a, b in tr if a != b]
    E = set(tuple(sorted(e)) for e in moves)
    out["nodes_per_step"] = V / T
    out["self_loop_share"] = 1 - len(moves) / len(tr)
    out["edge_density"] = len(E) / max(1, V * (V - 1) / 2)
    out["cycle_rank_per_step"] = (len(E) - V + 1) / T                       # independent cycles in the undirected graph
    vc = np.array(sorted(Counter(lab).values(), reverse=True)) / T
    out["top3_visit_share"] = vc[:3].sum()                                  # hub concentration
    first = {}
    for t, l in enumerate(lab): first.setdefault(l, t)
    out["return_rate"] = np.mean([first[lab[t]] < t - 1 and lab[t] != lab[t - 1] for t in range(1, T)])   # comes back to an old state
    out["new_state_rate_late"] = np.mean([first[lab[t]] == t for t in range(T // 2, T)])                   # still exploring late on
    # entropy rate of the empirical Markov chain (how predictable the next state is)
    cnt = Counter(tr); outdeg = Counter(a for a, _ in tr)
    out["entropy_rate"] = -sum(c / len(tr) * np.log(c / outdeg[a]) for (a, b), c in cnt.items())
    # jump lengths between centroids (logic jumps), relative to this model's typical centroid distance
    if moves:
        d = np.array([1 - C[a] @ C[b] for a, b in moves])
        out["mean_jump"] = d.mean(); out["max_jump"] = d.max()
    # repeated short cycles in the label sequence (loops)
    g3 = [tuple(lab[t:t + 3]) for t in range(T - 2) if len(set(lab[t:t + 3])) > 1]
    out["repeated_3gram_share"] = (np.mean([c > 1 for c in Counter(g3).values()]) if g3 else 0.0)
    # algebraic connectivity of the visited, weighted, undirected graph (low = two weakly linked communities)
    idx = {n: k for k, n in enumerate(sorted(nodes))}; A = np.zeros((V, V))
    for a, b in moves: A[idx[a], idx[b]] += 1; A[idx[b], idx[a]] += 1
    if V >= 3:
        dg = A.sum(1); ok = dg > 0
        if ok.sum() >= 3:
            A2 = A[np.ix_(ok, ok)]; dg = A2.sum(1); L = np.eye(len(dg)) - A2 / np.sqrt(np.outer(dg, dg))
            out["spectral_gap"] = np.sort(np.linalg.eigvalsh(L))[1]
    return out
