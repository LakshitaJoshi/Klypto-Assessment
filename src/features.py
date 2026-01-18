import numpy as np
import pandas as pd

# -------------------------
# EMA FEATURES (Task 2.1)
# -------------------------
def compute_ema(df, span, price_col="spot_close"):
    return df[price_col].ewm(span=span, adjust=False).mean()


# -------------------------
# RETURNS
# -------------------------
def compute_returns(series):
    return series.pct_change()


# -------------------------
# FUTURES BASIS
# -------------------------
def futures_basis(fut_close, spot_close):
    return (fut_close - spot_close) / spot_close


# -------------------------
# PCR CALCULATIONS
# -------------------------
def pcr_oi(put_oi, call_oi):
    return put_oi / call_oi.replace(0, np.nan)


def pcr_volume(put_vol, call_vol):
    return put_vol / call_vol.replace(0, np.nan)


# -------------------------
# DERIVED IV FEATURES
# -------------------------
def avg_iv(call_iv, put_iv):
    return (call_iv + put_iv) / 2


def iv_spread(call_iv, put_iv):
    return call_iv - put_iv
