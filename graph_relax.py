import numpy as np
import networkx as nx
from collections import defaultdict

__all__ = ["relax"]


class Interaction:
    def __init__(self, forcefunc):
        self.func = forcefunc

    def force(self, pairs, positions, cell=None):
        forces = np.zeros_like(positions)
        for a, b in pairs:
            d = positions[a] - positions[b]
            if cell is not None:
                d -= np.floor(d + 0.5)
                d = d @ cell
            r = np.linalg.norm(d)
            assert r > 0, (a, b, d, positions[a], positions[b])
            f = self.func(r) * d / r
            forces[a] -= f
            forces[b] += f
        return forces


def relax(
    g: nx.Graph,
    node_pos: np.ndarray,
    cell: np.ndarray = None,
    dt: float = 0.01,
    iters: int = 10,
) -> np.ndarray:
    """Relax the shape of a graph.

    Args:
        g (nx.Graph): a graph describing the connectivity between nodes.
        node_pos (np.ndarray): Initial positions of the nodes. Should it be in fractional coordinates when cell is specified.
        cell (np.ndarray, optional): The cell matrix. Defaults to None.
        dt (float, optional): Time delta for energy minimization. Defaults to 0.01.
        iters (int, optional): Number of iterations. Defaults to 10.

    Returns:
        np.ndarray: Updated positions of the nodes.
    """
    if cell is not None:
        celli = np.linalg.inv(cell)
    D = dict(nx.all_pairs_shortest_path_length(g))

    # 距離ごとに対を仕分ける。
    distances = defaultdict(list)
    for i in D:
        for j in D[i]:
            d = D[i][j]
            if i < j:
                distances[d].append((i, j))

    # もし相互作用を間引くならここで。

    for _ in range(iters):
        # 距離ごとに相互作用を変える。
        total_forces = np.zeros_like(node_pos)
        for d, pairs in distances.items():
            interaction = Interaction(lambda r: r - d)
            forces = interaction.force(pairs, node_pos, cell=cell)
            total_forces += forces

        if cell is not None:
            total_forces = total_forces @ celli

        node_pos += dt * total_forces

    return node_pos


def test():
    g = nx.octahedral_graph()
    atoms = np.random.random([96, 3])
    cell = None

    atoms = relax(g, atoms, cell, iters=1000)
    for i, j in g.edges():
        d = atoms[i] - atoms[j]
        if cell is not None:
            d -= np.floor(d + 0.5)
            d = d @ cell
        print(np.linalg.norm(d))


if __name__ == "__main__":
    test()
