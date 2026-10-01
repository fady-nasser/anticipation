import numpy as np
from numpy.linalg import lstsq

MIN_ACT_STD = 0.2  # minimum velocity std (m/s) to ensure active movement

def zscore(arr):
    """Z-score normalize an array with zero-division safeguard."""
    m = np.nanmean(arr)
    s = np.nanstd(arr)
    return np.zeros_like(arr) if s < 1e-9 else (arr - m) / s

def ols_rss(Y, X):
    """Compute Ordinary Least Squares Residual Sum of Squares (RSS)."""
    mask = ~(np.isnan(Y) | np.any(np.isnan(X), axis=1))
    if mask.sum() < 15:
        return np.nan, 0
    Ym, Xm = Y[mask], X[mask]
    b, _, _, _ = lstsq(np.c_[np.ones(len(Ym)), Xm], Ym, rcond=None)
    r = Ym - np.c_[np.ones(len(Ym)), Xm] @ b
    return np.sum(r**2), mask.sum()

def test_window_granger(vxA, vyA, vxB, vyB, bx, lag=2, f_critical=3.14):
    """
    Test Granger causality in a tactical window: Player A -> Player B
    conditioning on continuous ball movement (bx).
    
    Returns True if F > f_critical (p < 0.05).
    """
    n = len(vxA)
    if n <= lag + 5:
        return False
        
    # Check activity: players must be actively moving
    if np.nanstd(vxA) < MIN_ACT_STD and np.nanstd(vyA) < MIN_ACT_STD:
        return False
    if np.nanstd(vxB) < MIN_ACT_STD and np.nanstd(vyB) < MIN_ACT_STD:
        return False
        
    def make_lags(a, l_list):
        cols = []
        for l in l_list:
            c = np.full(n, np.nan)
            c[l:] = a[:n-l]
            cols.append(c)
        return np.column_stack(cols)
        
    Y = zscore(vyB)
    ol = list(range(1, lag + 1))
    
    # Restricted model: follower's own history + ball motion
    X_r = np.c_[
        make_lags(zscore(vxB), ol),
        make_lags(zscore(vyB), ol),
        make_lags(zscore(bx), ol)
    ]
    
    # Unrestricted model: restricted + leader's movement at specified lag
    X_u = np.c_[
        X_r,
        make_lags(zscore(vxA), [lag]),
        make_lags(zscore(vyA), [lag])
    ]
    
    rss_r, nr = ols_rss(Y, X_r)
    rss_u, nu = ols_rss(Y, X_u)
    
    if np.isnan(rss_r) or np.isnan(rss_u) or nu < 20 or rss_u < 1e-12:
        return False
        
    dfn = X_u.shape[1] - X_r.shape[1]
    dfd = nu - X_u.shape[1] - 1
    if dfd < 5 or dfn < 1:
        return False
        
    F = ((rss_r - rss_u) / dfn) / (rss_u / dfd)
    return F > f_critical

def compute_granger_f(vxA, vyA, vxB, vyB, bx, lag=2):
    """Compute and return exact F-statistic for a dyad in a window."""
    n = len(vxA)
    if n <= lag + 5:
        return np.nan
    if np.nanstd(vxA) < MIN_ACT_STD and np.nanstd(vyA) < MIN_ACT_STD:
        return np.nan
    if np.nanstd(vxB) < MIN_ACT_STD and np.nanstd(vyB) < MIN_ACT_STD:
        return np.nan
        
    def make_lags(a, l_list):
        cols = []
        for l in l_list:
            c = np.full(n, np.nan)
            c[l:] = a[:n-l]
            cols.append(c)
        return np.column_stack(cols)
        
    Y = zscore(vyB)
    ol = list(range(1, lag + 1))
    X_r = np.c_[make_lags(zscore(vxB), ol), make_lags(zscore(vyB), ol), make_lags(zscore(bx), ol)]
    X_u = np.c_[X_r, make_lags(zscore(vxA), [lag]), make_lags(zscore(vyA), [lag])]
    
    rss_r, nr = ols_rss(Y, X_r)
    rss_u, nu = ols_rss(Y, X_u)
    if np.isnan(rss_r) or np.isnan(rss_u) or nu < 20 or rss_u < 1e-12:
        return np.nan
    dfn = X_u.shape[1] - X_r.shape[1]
    dfd = nu - X_u.shape[1] - 1
    if dfd < 5 or dfn < 1:
        return np.nan
    return float(((rss_r - rss_u) / dfn) / (rss_u / dfd))
