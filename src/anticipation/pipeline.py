import numpy as np
import pandas as pd
from numpy.linalg import lstsq

def zscore(arr):
    m = np.nanmean(arr)
    s = np.nanstd(arr)
    return np.zeros_like(arr) if s < 1e-9 else (arr - m) / s

def ols_rss(Y, X):
    mask = ~(np.isnan(Y) | np.any(np.isnan(X), axis=1))
    if mask.sum() < 15:
        return np.nan, 0
    Ym, Xm = Y[mask], X[mask]
    b, _, _, _ = lstsq(np.c_[np.ones(len(Ym)), Xm], Ym, rcond=None)
    r = Ym - np.c_[np.ones(len(Ym)), Xm] @ b
    return np.sum(r**2), mask.sum()

def make_lags(a, l_list, n):
    cols = []
    for l in l_list:
        c = np.full(n, np.nan)
        c[l:] = a[:n-l]
        cols.append(c)
    return np.column_stack(cols)

def run_window_anticipation_counts(npz_path, leaders_dict, followers_dict, 
                                   outliers=None, window_s=15.0, step_s=7.5, hz=5, lag=2, f_critical=3.14):
    """
    Computes discrete anticipation counts directly from an extracted velocity features (.npz) file.
    
    leaders_dict:   dict of {feature_prefix_jersey: display_name}
    followers_dict: dict of {feature_prefix_jersey: display_name}
    outliers:       list of display_names to exclude from the final matrix
    """
    data = np.load(npz_path)
    period = data['period']
    ball_x = data['ball_x']
    total_pts = len(period)
    
    window_pts = int(window_s * hz)
    step_pts = int(step_s * hz)
    n = window_pts
    ol = list(range(1, lag + 1))
    
    # Pre-extract player velocity arrays
    all_keys = set(list(leaders_dict.keys()) + list(followers_dict.keys()))
    vel_p = {}
    for k in all_keys:
        vx_k = f"{k}_vx"
        vy_k = f"{k}_vy"
        if vx_k in data and vy_k in data:
            vel_p[k] = (data[vx_k], data[vy_k])
            
    # Segment match into rolling 15s tactical windows
    windows = []
    for start in range(0, total_pts - window_pts, step_pts):
        end = start + window_pts
        if period[start] == period[end - 1]:
            bx = ball_x[start:end]
            p_win = {k: (vx[start:end], vy[start:end]) for k, (vx, vy) in vel_p.items()}
            windows.append((bx, p_win))
            
    l_names = list(leaders_dict.values())
    f_names = list(followers_dict.values())
    count_df = pd.DataFrame(0, index=l_names, columns=f_names)
    
    # Compute Granger anticipation count per window
    for bx, p_win in windows:
        # Filter players actively moving in this window (speed std >= 0.2 m/s)
        active = {}
        for k, (vx, vy) in p_win.items():
            if np.nanstd(vx) >= 0.2 or np.nanstd(vy) >= 0.2:
                active[k] = (vx, vy)
                
        if not active:
            continue
            
        # Precompute restricted model (Follower own history + Ball) once per follower
        restr_cache = {}
        for jB, nB in followers_dict.items():
            if jB not in active:
                continue
            vxB, vyB = active[jB]
            YB = zscore(vyB)
            XrB = np.c_[make_lags(zscore(vxB), ol, n), make_lags(zscore(vyB), ol, n), make_lags(zscore(bx), ol, n)]
            rss_r, nr = ols_rss(YB, XrB)
            if not np.isnan(rss_r) and nr >= 20:
                restr_cache[jB] = (YB, XrB, rss_r, nr)
                
        # Test each leader's incremental predictive power on each follower
        for jB, (YB, XrB, rss_r, nr) in restr_cache.items():
            nB = followers_dict[jB]
            for jA, nA in leaders_dict.items():
                if jA == jB or jA not in active:
                    continue
                vxA, vyA = active[jA]
                Xu = np.c_[XrB, make_lags(zscore(vxA), [lag], n), make_lags(zscore(vyA), [lag], n)]
                rss_u, nu = ols_rss(YB, Xu)
                if np.isnan(rss_u) or nu < 20 or rss_u < 1e-12:
                    continue
                dfn = Xu.shape[1] - XrB.shape[1]
                dfd = nu - Xu.shape[1] - 1
                if dfd < 5 or dfn < 1:
                    continue
                F = ((rss_r - rss_u) / dfn) / (rss_u / dfd)
                if F > f_critical:
                    count_df.loc[nA, nB] += 1
                    
    # Drop outliers if specified
    if outliers:
        count_df = count_df.drop(index=outliers, columns=outliers, errors='ignore')
        
    return count_df

