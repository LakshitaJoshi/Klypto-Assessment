import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

def prepare_hmm_features(df, feature_cols):
    """
    Select and standardize features for HMM.
    """
    X = df[feature_cols].copy()
    X = X.dropna()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X.index, X_scaled, scaler


def train_hmm(X, n_states=3, random_state=42):
    """
    Train Gaussian HMM.
    """
    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=500,
        random_state=random_state
    )
    model.fit(X)
    return model


def predict_regimes(model, X):
    """
    Predict hidden states.
    """
    return model.predict(X)


def map_regimes_to_trend(df, regime_col, return_col="spot_returns"):
    """
    Map raw HMM states to:
    +1 → Uptrend
    -1 → Downtrend
     0 → Sideways
    """
    mapping = {}

    for r in df[regime_col].unique():
        mean_ret = df.loc[df[regime_col] == r, return_col].mean()

        if mean_ret > 0:
            mapping[r] = 1
        elif mean_ret < 0:
            mapping[r] = -1
        else:
            mapping[r] = 0

    df["regime"] = df[regime_col].map(mapping)
    return df, mapping
