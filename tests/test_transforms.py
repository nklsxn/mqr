import mqr

import numpy as np
import pandas as pd
import pytest
import scipy

@pytest.fixture
def data():
    np.random.seed(0)
    return pd.DataFrame({
        'x': scipy.stats.norm(1, 2).rvs(100),
        'y': scipy.stats.norm(3, 4).rvs(100),
        'z': scipy.stats.norm(5, 6).rvs(100),
    })

def test_zscore(data):
    stats, z, z_inv = mqr.transforms.zscore(data)
    assert list(stats.loc['mean']) == pytest.approx([1, 3, 5], abs=0.5)
    assert list(stats.loc['std']) == pytest.approx([2, 4, 6], abs=0.5)
    assert np.all(np.isclose(z_inv(z(data['x'])), data['x']))
    assert np.all(np.isclose(z(z_inv(data['x'])), data['x']))
    assert np.all(np.isclose(z_inv(z(data)), data))
    assert np.all(np.isclose(z(z_inv(data)), data))

    stats, z, z_inv = mqr.transforms.zscore(data, 3, 4)
    assert np.all(np.isclose(z_inv(z(data)), data))
    assert np.all(np.isclose(z(z_inv(data)), data))
