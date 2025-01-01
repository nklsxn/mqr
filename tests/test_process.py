import mqr

import numpy as np
import pandas as pd
import pytest
import scipy

def test_Summary():
    np.random.seed(0)
    data = pd.DataFrame({
        'x': scipy.stats.norm(1, 2).rvs(100)
    })
    summary = mqr.process.Summary(data)

def test_Capability():
    np.random.seed(0)
    data = pd.DataFrame({
        'x': scipy.stats.norm(1, 2).rvs(100)
    })
    spec = mqr.process.Specification(-1, -1-5*2, -1+5*2) # Potential around 1.67, capability around 1.33
    summary = mqr.process.Summary(data, {'x': spec})

    c = summary.capabilities['x']
    assert c.cp == pytest.approx(1.67, abs=0.1)
    assert c.cpk == pytest.approx(1.33, abs=0.1)
