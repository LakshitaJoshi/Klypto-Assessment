from py_vollib.black_scholes.greeks.analytical import (
    delta, gamma, theta, vega, rho
)

def compute_single_greek(option_type, S, K, T, r, iv):
    return delta(option_type, S, K, T, r, iv)
