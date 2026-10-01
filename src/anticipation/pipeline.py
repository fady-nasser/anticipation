import numpy as np
import pandas as pd
from .granger import test_window_granger

def run_window_anticipation_counts(npz_path, leaders, followers, window_s=15.0, step_s=7.5, hz=5):
    """
    Run rolling window Granger anticipation counting on an extracted features .npz file.
    
    leaders: dict of {jersey: display_name}
    followers: dict of {jersey: display_name}
    """
    data = np.load(npz_path)
    period = data['period']
    ball_x = data['ball_x']
    
    window_pts = int(window_s * hz)
    step_pts = int(step_s * hz)
    total_pts = len(period)
    
    # Pre-extract player velocity arrays
    all_jerseys = set(list(leaders.keys()) + list(followers.keys()))
    vel_arrays = {}
    for j in all_jerseys:
        vx_key = f"{j}_vx"
        vy_key = f"{j}_vy"
        if vx_key in data and vy_key in data:
            vel_arrays[j] = (data[vx_key], data[vy_key])
            
    # Extract rolling windows within same period
    windows = []
    for start in range(0, total_pts - window_pts, step_pts):
        end = start + window_pts
        if period[start] == period[end - 1]:
            bx = ball_x[start:end]
            p_win = {j: (vx[start:end], vy[start:end]) for j, (vx, vy) in vel_arrays.items()}
            windows.append((bx, p_win))
            
    # Initialize count dataframe
    l_names = list(leaders.values())
    f_names = list(followers.values())
    count_df = pd.DataFrame(0, index=l_names, columns=f_names)
    
    # Process windows
    for bx, p_win in windows:
        for jA, nA in leaders.items():
            if jA not in p_win: continue
            vxA, vyA = p_win[jA]
            for jB, nB in followers.items():
                if jA == jB or jB not in p_win: continue
                vxB, vyB = p_win[jB]
                if test_window_granger(vxA, vyA, vxB, vyB, bx, lag=2):
                    count_df.loc[nA, nB] += 1
                    
    return count_df
