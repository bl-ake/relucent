from relucent import Complex, set_seeds, get_mlp_model


def test_bfs_one():
    set_seeds(0)

    model = get_mlp_model(widths=[16, 64, 64, 64, 10])

    print("Model:\n", model)

    cplx = Complex(model)

    cplx.bfs(max_polys=100)


if __name__ == "__main__":
    test_bfs_one()
