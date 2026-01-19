from relucent import Complex, set_seeds, get_mlp_model


def test_bfs_one():
    set_seeds(0)

    widths = [16, 64, 64, 64, 10]

    model = get_mlp_model(widths=widths)

    print("Model:\n", model)

    cplx = Complex(model)

    # # breakpoint()

    # start = torch.rand(widths[0])
    # end = torch.rand(widths[0])

    # start_poly = cplx.point2poly(start)
    # end_poly = cplx.point2poly(end)

    # print("Hamming:", (start_poly.bv != end_poly.bv).sum())
    # path = cplx.hamming_astar(nworkers=64, start=start, end=end)
    # print("Path:", len(path) - 1)
    # print()
    # print(path)

    cplx.bfs(max_polys=100)


if __name__ == "__main__":
    test_bfs_one()
