import numpy as np

def isotonic_nondec(y):
    """
    Pool-Adjacent-Violators Algorithm (PAVA) for non-decreasing projection.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    vals = y.copy()
    wgts = np.ones(n, dtype=float)

    i = 0
    m = n
    while i < m - 1:
        if vals[i] <= vals[i+1]:
            i += 1
            continue
        new_w = wgts[i] + wgts[i+1]
        new_v = (wgts[i]*vals[i] + wgts[i+1]*vals[i+1]) / new_w
        vals[i] = new_v
        wgts[i] = new_w
        vals = np.delete(vals, i+1)
        wgts = np.delete(wgts, i+1)
        m -= 1
        if i > 0:
            i -= 1
    return np.repeat(vals, wgts.astype(int))
