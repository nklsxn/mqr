import numpy as np
import pandas as pd

def zscore(data):
    stats = pd.DataFrame(index=['mean', 'std'], columns=data.columns)
    stats.loc['mean'] = data.apply(np.mean)
    stats.loc['std'] = data.apply(np.std, ddof=0)

    def z(col):
        mean, std = stats[col.name]
        return (col - mean) / std
    def z_inv(col):
        mean, std = stats[col.name]
        return col * std + mean
    def _z(x):
        if isinstance(x, pd.DataFrame):
            return x.apply(z)
        elif isinstance(x, pd.Series):
            return z(x)
        else:
            raise ValueError('Pass either a DataFrame or a Series.')
    def _z_inv(x):
        if isinstance(x, pd.DataFrame):
            return x.apply(z_inv)
        elif isinstance(x, pd.Series):
            return z_inv(x)
        else:
            raise ValueError('Pass either a DataFrame of a Series.')
    return stats, _z, _z_inv
