from relucent import Complex, set_seeds, get_mlp_model
import torch
import networkx as nx


def test_complex_one():
    set_seeds(0)

    model = get_mlp_model(widths=[4, 8], add_last_relu=True)

    cplx1 = Complex(model)

    start_point1 = torch.rand((1, 4)).to(model.dtype)

    cplx1.bfs(start=start_point1)

    G1 = cplx1.get_dual_graph()

    cplx2 = Complex(model)

    start_point_2 = torch.rand((1, 4)).to(model.dtype)

    cplx2.bfs(start=start_point_2)

    G2 = cplx2.get_dual_graph()

    p1 = cplx1.point2bv(start_point1)

    assert p1 in cplx2
    assert nx.is_isomorphic(G1, G2)


def test_complex_two():
    set_seeds(0)

    model = get_mlp_model(widths=[5, 9], add_last_relu=True)

    cplx1 = Complex(model)

    start_point1 = torch.rand((1, 5)).to(model.dtype)

    cplx1.bfs(start=start_point1)

    G1 = cplx1.get_dual_graph()

    cplx2 = Complex(model)

    cplx2.recover_from_dual_graph(G1, initial_bv=cplx1.point2bv(start_point1), source=cplx1.point2poly(start_point1))

    G2 = cplx2.get_dual_graph()

    assert (G1.adj == G2.adj) and (G1.nodes == G2.nodes) and (G1.graph == G2.graph)


def test_search_one():
    set_seeds(0)

    model = get_mlp_model(widths=[16, 64, 64, 64, 10])

    cplx = Complex(model)

    start_point = torch.rand(16).to(model.dtype)

    p = cplx.point2poly(start_point)

    assert torch.allclose(start_point @ p.W + p.b, model(start_point))

    cplx.bfs(max_polys=100, start=start_point)

    assert p in cplx

    assert len(cplx) == 100

    assert len(set(cplx.index2poly)) == len(cplx)


def test_search_two():
    set_seeds(0)

    model = get_mlp_model(widths=[6, 8, 10])

    cplx = Complex(model)

    dfs_result = cplx.dfs(max_depth=2, nworkers=1, get_volumes=False)

    assert dfs_result["Search Depth"] == 2

    assert [p.shis is not None for p in cplx]


if __name__ == "__main__":
    test_complex_one()
    test_complex_two()
    test_search_one()
    test_search_two()
