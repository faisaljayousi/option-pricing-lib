from oplib.pricers.bs_geometric_asian import (
    geometric_asian_call_bs,
    geometric_asian_put_bs,
)
from oplib.pricers.bs_vanilla import call_price, put_price


def test_geometric_call_reduces_to_vanilla_when_m_equals_1():
    s0, k, r, q, sigma, T, m = 100, 100, 0.02, 0.0, 0.2, 1.0, 1
    geo = geometric_asian_call_bs(s0, k, r, q, sigma, T, m)
    bs = call_price(s0, k, r, q, sigma, T)
    assert abs(geo - bs) < 1e-12


def test_geometric_put_reduces_to_vanilla_when_m_equals_1():
    s0, k, r, q, sigma, T, m = 100, 100, 0.02, 0.0, 0.2, 1.0, 1
    geo = geometric_asian_put_bs(s0, k, r, q, sigma, T, m)
    bs = put_price(s0, k, r, q, sigma, T)
    assert abs(geo - bs) < 1e-12
