import numpy as np
import networkx as nx
from collections import defaultdict


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


def relax(g, node_pos, cell=None, dt=0.01, iters=10):
    # 隣接情報gを立体化する。
    # node_posを初期配置とする。
    # cell行列が与えられる場合はnode_posはセル相対でなければならない。
    # 今はcellの調節はしない。結合長さは1とする。
    # 隣接しない対の距離はネットワーク上の距離にする。ただし、max_walkよりも遠い節点間には力は働かない。

    # g から 距離行列(距離グラフ)を生成する。連結でない場合は打ち切ってよい。
    # 計算量を減らしたいなら、距離行列を間引けばいい。遠距離の対ほど間引くようにする。(というより、遠距離の対ほど増えるので、その増加を抑えるように間引けばいい)
    # まずは とりあえず全部計算する。
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
