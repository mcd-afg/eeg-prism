
import pandas as pd
import numpy as np

def cat_p_factor(df):
    return pd.cut(
        df["p_factor"],
        bins=[-np.inf, 0.44, 0.946, np.inf],
        labels=[0, 1, 2]
    ).astype(int)

def bin_p_factor(df, bool):
    if bool==True:
        return pd.cut(
            df["p_factor"],
            bins=[-np.inf, 0.946, np.inf],
            labels=[0, 1]
        ).astype(int)
    else:
        return pd.cut(
            df["p_factor"],
            bins=[-np.inf, 0.44, np.inf],
            labels=[0, 1]
        ).astype(int)