"""Test that relucent public API is importable."""


def test_top_level_imports():
    from relucent import (
        Complex,
        Polyhedron,
        NN,
        SSManager,
        convert,
        get_env,
        get_mlp_model,
        set_seeds,
        split_sequential,
    )

    assert Complex is not None
    assert Polyhedron is not None
    assert NN is not None
    assert SSManager is not None
    assert callable(convert)
    assert callable(get_env)
    assert callable(get_mlp_model)
    assert callable(set_seeds)
    assert callable(split_sequential)
