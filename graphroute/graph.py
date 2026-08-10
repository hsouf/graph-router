"""Sparse graph primitives: pruning, community detection, frontier PPR.

Everything operates on scipy CSR matrices and stays sparse. The traversal
routine never materialises a dense vector over the whole node set, which is
what keeps query latency flat as the corpus grows.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import scipy.sparse as sp


def _row_ids(mat: sp.csr_matrix) -> np.ndarray:
    return np.repeat(np.arange(mat.shape[0], dtype=np.int64), np.diff(mat.indptr))


def prune_topk(mat: sp.csr_matrix, k: int) -> sp.csr_matrix:
    """Keep only the k largest entries in each row.

    The single most important knob for query speed: it bounds the branching
    factor of every traversal step regardless of how hubby a node is.
    """
    mat = mat.tocsr()
    mat.sort_indices()
    indptr, indices, data = mat.indptr, mat.indices, mat.data
    if data.size == 0:
        return mat

    keep = np.ones(data.shape[0], dtype=bool)
    for row in range(mat.shape[0]):
        start, end = indptr[row], indptr[row + 1]
        n = end - start
        if n <= k:
            continue
        drop = np.argpartition(data[start:end], n - k)[: n - k]
        keep[start + drop] = False

    rows = _row_ids(mat)[keep]
    out = sp.csr_matrix(
        (data[keep], (rows, indices[keep])), shape=mat.shape
    )
    out.eliminate_zeros()
    return out


SUPPORT_SATURATION = 8.0
"""Joint count at which the NPMI support discount reaches 1.0."""


def cooccurrence(
    doc_term: sp.csr_matrix,
    weight: str = "npmi",
    min_pair: int = 2,
    support_saturation: float = SUPPORT_SATURATION,
) -> sp.csr_matrix:
    """Term-term graph from a document-term matrix.

    `npmi` normalises away raw frequency, so a rare term co-occurring with
    another rare term outranks anything that merely shares space with a hub.
    That is what lets the graph carry *topical* rather than *popular* signal.

    NPMI alone has a sharp edge: it is exactly 1.0 for *any* pair that always
    co-occurs, including a pair seen together twice in the whole corpus. Left
    uncorrected, the least-supported edges in the graph are also the strongest,
    and a degree-capped neighbour list fills up with coincidences before it
    reaches the terms a topic is actually built from. Scaling by
    `log1p(joint) / log1p(support_saturation)` — capped at 1.0 — keeps NPMI's
    shape while requiring a pair to be seen `support_saturation` times before
    it can claim full strength. Two co-occurrences buy half the weight of eight.
    """
    binary = doc_term.copy().tocsr()
    binary.data = np.ones_like(binary.data, dtype=np.float32)

    co = (binary.T @ binary).tocoo()
    n_docs = binary.shape[0]
    df = np.asarray(binary.sum(axis=0)).ravel()

    mask = (co.row != co.col) & (co.data >= min_pair)
    rows, cols, joint = co.row[mask], co.col[mask], co.data[mask]
    if rows.size == 0:
        return sp.csr_matrix((binary.shape[1],) * 2, dtype=np.float32)

    if weight == "count":
        vals = joint.astype(np.float32)
    else:
        p_xy = joint / n_docs
        p_x = df[rows] / n_docs
        p_y = df[cols] / n_docs
        with np.errstate(divide="ignore", invalid="ignore"):
            pmi = np.log(p_xy / (p_x * p_y))
            npmi = pmi / -np.log(p_xy)
        vals = np.nan_to_num(npmi, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        vals = np.clip(vals, 0.0, None)
        if support_saturation > 1:
            support = np.minimum(
                1.0, np.log1p(joint) / np.log1p(support_saturation)
            ).astype(np.float32)
            vals = vals * support

    keep = vals > 0
    return sp.csr_matrix(
        (vals[keep], (rows[keep], cols[keep])), shape=(binary.shape[1],) * 2
    )


def row_normalize(mat: sp.csr_matrix) -> sp.csr_matrix:
    sums = np.asarray(mat.sum(axis=1)).ravel()
    sums[sums == 0] = 1.0
    return (sp.diags(1.0 / sums) @ mat).tocsr()


def label_propagation(
    adj: sp.csr_matrix, max_iter: int = 10, inertia: float = 0.0
) -> np.ndarray:
    """Vectorised weighted label propagation -> one community id per node.

    Synchronous updates via a one-hot label matrix: `adj @ L` gives every node
    the weighted mass of every candidate label in one sparse product, so a pass
    costs O(edges) with no Python loop over nodes.

    Ties break toward the lowest label id (scipy's `argmax` returns the first
    maximum), which is both deterministic and what breaks the symmetry of a
    fresh all-distinct labelling. `inertia` optionally biases nodes toward
    their current label; leave it at 0 unless a graph is oscillating, since a
    positive value can freeze symmetric cliques in place.

    Synchronous updates can settle into a two-cycle instead of a fixed point,
    so we watch the previous two states and stop on either.
    """
    n = adj.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.int32)
    adj = adj.tocsr()
    labels = np.arange(n, dtype=np.int64)
    has_edges = np.diff(adj.indptr) > 0
    prev = None

    for _ in range(max_iter):
        uniq, comp = np.unique(labels, return_inverse=True)
        onehot = sp.csr_matrix(
            (np.ones(n, dtype=np.float32), (np.arange(n), comp)),
            shape=(n, uniq.size),
        )
        mass = (adj @ onehot).tocsr()
        if inertia:
            mass = (mass + onehot * np.float32(inertia)).tocsr()
        best = np.asarray(mass.argmax(axis=1)).ravel()
        new_labels = np.where(has_edges, uniq[best], labels)

        if np.array_equal(new_labels, labels) or (
            prev is not None and np.array_equal(new_labels, prev)
        ):
            labels = new_labels
            break
        prev, labels = labels, new_labels

    _, compact = np.unique(labels, return_inverse=True)
    return compact.astype(np.int32)


def frontier_ppr(
    adj: sp.csr_matrix,
    seeds: Dict[int, float],
    alpha: float = 0.45,
    hops: int = 3,
    beam: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    """Personalised PageRank restricted to a beam-limited frontier.

    Push-style approximation: each hop keeps only the `beam` heaviest residual
    nodes. Work per query is O(hops * beam * avg_degree) and is independent of
    the size of the graph, which is the whole point.

    Returns (node_ids, scores) sorted by descending score.
    """
    if not seeds:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)

    idx = np.fromiter(seeds.keys(), dtype=np.int64, count=len(seeds))
    val = np.fromiter(seeds.values(), dtype=np.float64, count=len(seeds))
    total = val.sum()
    if total <= 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
    val = val / total

    scores: Dict[int, float] = {}

    for _ in range(hops):
        for node, weight in zip(idx.tolist(), val.tolist()):
            scores[node] = scores.get(node, 0.0) + alpha * weight

        sub = adj[idx].tocoo()
        if sub.nnz == 0:
            idx, val = np.empty(0, dtype=np.int64), np.empty(0)
            break
        contrib = sub.data * val[sub.row] * (1.0 - alpha)
        uniq, inv = np.unique(sub.col, return_inverse=True)
        agg = np.zeros(uniq.size, dtype=np.float64)
        np.add.at(agg, inv, contrib)

        if uniq.size > beam:
            top = np.argpartition(agg, uniq.size - beam)[uniq.size - beam :]
            uniq, agg = uniq[top], agg[top]
        idx, val = uniq.astype(np.int64), agg
        if val.sum() <= 1e-12:
            break

    # The frontier left over after the final hop never got its turn at the top
    # of the loop. Accumulate it at the same alpha weighting as every other
    # hop; giving it full weight instead would let the outermost ring outscore
    # nodes one step from the seed.
    for node, weight in zip(idx.tolist(), val.tolist()):
        scores[node] = scores.get(node, 0.0) + alpha * weight

    nodes = np.fromiter(scores.keys(), dtype=np.int64, count=len(scores))
    vals = np.fromiter(scores.values(), dtype=np.float32, count=len(scores))
    order = np.argsort(-vals)
    return nodes[order], vals[order]
