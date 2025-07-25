import math

import jax.numpy as jnp
import pytest

from simplexity.utils.dyck_paths import catalan_number, get_dyck_paths, log_comb, unrank_dyck_path


def test_catalan_number():
    assert catalan_number(0) == 1
    assert catalan_number(1) == 1
    assert catalan_number(2) == 2
    assert catalan_number(3) == 5
    assert catalan_number(4) == 14
    assert catalan_number(5) == 42
    assert catalan_number(6) == 132
    assert catalan_number(7) == 429
    assert catalan_number(8) == 1430


@pytest.mark.parametrize(
    ("n", "k"),
    [
        (0, 0),
        (1, 0),
        (1, 1),
        (2, 0),
        (2, 1),
        (2, 2),
    ],
)
def test_log_comb(n, k):
    assert jnp.round(jnp.exp(log_comb(n, k))) == math.comb(n, k)


@pytest.mark.parametrize("n", [0, 1, 2, 3])
def test_get_dyck_paths(n):
    paths = get_dyck_paths(n)
    assert paths.shape == (catalan_number(n), 2 * n)
    assert jnp.all(jnp.sum(paths, axis=1) == 0)
    assert jnp.all(jnp.cumsum(paths, axis=1) >= 0)
    assert len(jnp.unique(paths, axis=0)) == paths.shape[0]


def test_unrank_dyck_path():
    path = unrank_dyck_path(jnp.array(0), 3)
    assert jnp.all(path == jnp.array([1, 1, 1, -1, -1, -1]))
    path = unrank_dyck_path(jnp.array(1), 3)
    assert jnp.all(path == jnp.array([1, 1, -1, 1, -1, -1]))
    path = unrank_dyck_path(jnp.array(2), 3)
    assert jnp.all(path == jnp.array([1, 1, -1, -1, 1, -1]))
    path = unrank_dyck_path(jnp.array(3), 3)
    assert jnp.all(path == jnp.array([1, -1, 1, 1, -1, -1]))
    path = unrank_dyck_path(jnp.array(4), 3)
    assert jnp.all(path == jnp.array([1, -1, 1, -1, 1, -1]))
