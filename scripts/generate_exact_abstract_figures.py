import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
    data = np.load(npz_path)
    period = data['period']
    ball_x = data['ball_x']
    total_pts = len(period)
    
    window_pts = int(window_s * hz)
    step_pts = int(step_s * hz)
    n = window_pts
    ol = list(range(1, lag + 1))
    
    all_keys = set(list(leaders_dict.keys()) + list(followers_dict.keys()))
    vel_p = {}
    for k in all_keys:
        vx_k = f'{k}_vx'
        vy_k = f'{k}_vy'
        if vx_k in data and vy_k in data:
            vel_p[k] = (data[vx_k], data[vy_k])
            
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
    
    for bx, p_win in windows:
        active = {}
        for k, (vx, vy) in p_win.items():
            if np.nanstd(vx) >= 0.2 or np.nanstd(vy) >= 0.2:
                active[k] = (vx, vy)
                
        if not active:
            continue
            
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
                    
    if outliers:
        count_df = count_df.drop(index=outliers, columns=outliers, errors='ignore')
        
    return count_df

print('=== STARTING EXACT REPRODUCTION OF ABSTRACT FIGURES ===\n')

# ----------------- FIGURE 1 ROSTERS -----------------
# 1. France: string-sorted keys
fra_fig1_roster = {
    'FRA_10':'Mbappe', 'FRA_11':'Dembele', 'FRA_12':'KoloMuani', 'FRA_14':'Rabiot',
    'FRA_18':'Upamecano', 'FRA_20':'Coman', 'FRA_22':'T.Hernandez', 'FRA_25':'Camavinga',
    'FRA_26':'M.Thuram', 'FRA_4':'Varane', 'FRA_7':'Griezmann', 'FRA_8':'Tchouameni',
    'FRA_9':'Giroud'
}
t0 = time.time()
fra_counts = run_window_anticipation_counts(
    'data/extracted_features/match_10517_final_features.npz',
    fra_fig1_roster, fra_fig1_roster
)
np.fill_diagonal(fra_counts.values, 0)
print(f'France calculated in {time.time()-t0:.1f}s')

# 2. Argentina: string-sorted keys
arg_fig1_roster = {
    'ARG_10':'Messi', 'ARG_13':'Romero', 'ARG_19':'Otamendi', 'ARG_20':'Mac Allister',
    'ARG_21':'Dybala', 'ARG_25':'L.Martinez', 'ARG_26':'Molina', 'ARG_3':'Tagliafico',
    'ARG_7':'De Paul', 'ARG_9':'Alvarez'
}
t0 = time.time()
arg_counts = run_window_anticipation_counts(
    'data/extracted_features/match_10514_argentina_features.npz',
    arg_fig1_roster, arg_fig1_roster
)
np.fill_diagonal(arg_counts.values, 0)
print(f'Argentina calculated in {time.time()-t0:.1f}s')

# 3. Morocco: string-sorted keys
mar_fig1_roster = {
    'MAR_14':'Aboukhlal', 'MAR_15':'Amallah', 'MAR_17':'Boufal', 'MAR_18':'El Yamiq',
    'MAR_19':'En-Nesyri', 'MAR_2':'Hakimi', 'MAR_20':'Dari', 'MAR_25':'Attiyat Allah',
    'MAR_3':'Mazraoui', 'MAR_7':'Ziyech', 'MAR_8':'Ounahi', 'MAR_9':'Hamdallah'
}
t0 = time.time()
mar_counts = run_window_anticipation_counts(
    'data/extracted_features/match_10515_morocco_features.npz',
    mar_fig1_roster, mar_fig1_roster
)
np.fill_diagonal(mar_counts.values, 0)
print(f'Morocco calculated in {time.time()-t0:.1f}s')

# 4. Serbia: string-sorted keys
srb_fig1_roster = {
    'SRB_10':'Tadic', 'SRB_13':'S.Mitrovic', 'SRB_14':'Zivkovic', 'SRB_16':'Lukic',
    'SRB_17':'Kostic', 'SRB_2':'Pavlovic', 'SRB_20':'S.Milinkovic', 'SRB_26':'Grujic',
    'SRB_4':'Milenkovic', 'SRB_5':'Veljkovic', 'SRB_6':'Maksimovic', 'SRB_7':'Radonjic',
    'SRB_9':'A.Mitrovic'
}
t0 = time.time()
srb_counts = run_window_anticipation_counts(
    'data/extracted_features/match_3840_serbia_features.npz',
    srb_fig1_roster, srb_fig1_roster
)
np.fill_diagonal(srb_counts.values, 0)
print(f'Serbia calculated in {time.time()-t0:.1f}s')

# ----------------- RENDER FIGURE 1 -----------------
fig, axes = plt.subplots(2, 2, figsize=(16, 15))

teams_fig1 = [
    ('(A) France (2022 World Cup Final vs Argentina)', fra_counts, 'Rabiot -> Tchouameni (293 times)'),
    ('(B) Argentina (2022 World Cup Semi-Final 3-0 vs Croatia)', arg_counts, 'Messi -> Mac Allister (218 times)'),
    ('(C) Morocco (2022 World Cup Semi-Final vs France)', mar_counts, 'Hakimi -> Ounahi (225 times)'),
    ('(D) Serbia (2022 World Cup Group Stage 3-3 vs Cameroon)', srb_counts, 'Tadic -> A.Mitrovic (215 times)')
]
positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

for idx, (title, cdf, top_pair_str) in enumerate(teams_fig1):
    r, c = positions[idx]
    ax = axes[r, c]
    sns.heatmap(cdf, ax=ax, cmap='Blues', annot=True, fmt='d', linewidths=0.4,
                cbar_kws={'label': 'Anticipation Event Count'}, vmin=0)
    ax.set_title(f"{title}\nTop Pair: {top_pair_str}", fontsize=11, fontweight='bold', pad=10)
    ax.set_xlabel('Follower (Reactor)', fontsize=10)
    ax.set_ylabel('Leader (Trigger)', fontsize=10)
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='y', rotation=0)

plt.tight_layout()
os.makedirs('results/figures', exist_ok=True)
fig1_path = 'results/figures/figure1_exact_abstract.png'
plt.savefig(fig1_path, dpi=200, bbox_inches='tight')
plt.close()
print(f'Saved {fig1_path}')

# ----------------- FIGURE 2 ROSTERS -----------------
# Numerical jersey-sorted:
fra_fig2_roster = {
    'FRA_4':'Varane', 'FRA_7':'Griezmann', 'FRA_8':'Tchouameni', 'FRA_9':'Giroud',
    'FRA_10':'Mbappe', 'FRA_11':'Dembele', 'FRA_12':'KoloMuani', 'FRA_14':'Rabiot',
    'FRA_18':'Upamecano', 'FRA_20':'Coman', 'FRA_22':'T.Hernandez', 'FRA_25':'Camavinga',
    'FRA_26':'M.Thuram'
}
arg_fig2_roster = {
    'ARG_3':'Tagliafico', 'ARG_7':'De Paul', 'ARG_9':'Alvarez', 'ARG_10':'Messi',
    'ARG_13':'Romero', 'ARG_19':'Otamendi', 'ARG_20':'Mac Allister', 'ARG_21':'Dybala',
    'ARG_25':'L.Martinez', 'ARG_26':'Molina'
}

# (A) France In-Team
fra_in_counts = run_window_anticipation_counts(
    'data/extracted_features/match_10517_final_features.npz',
    fra_fig2_roster, fra_fig2_roster
)
np.fill_diagonal(fra_in_counts.values, 0)

# (B) Argentina Reads France
arg_reads_fra = run_window_anticipation_counts(
    'data/extracted_features/match_10517_final_features.npz',
    arg_fig2_roster, fra_fig2_roster
)

# (C) France Reads Argentina
fra_reads_arg = run_window_anticipation_counts(
    'data/extracted_features/match_10517_final_features.npz',
    fra_fig2_roster, arg_fig2_roster
)

# ----------------- RENDER FIGURE 2 -----------------
fig, axes = plt.subplots(1, 3, figsize=(22, 6.5))

# (A)
sns.heatmap(fra_in_counts, ax=axes[0], cmap='YlGnBu', annot=True, fmt='d', linewidths=0.3,
            cbar_kws={'label': 'Event Count (15s Windows)'}, vmin=0)
axes[0].set_title('(A) France In-Team Chemistry\n(Leader: France -> Follower: France)', fontsize=10.5, fontweight='bold', pad=8)
axes[0].set_xlabel('Follower (France)', fontsize=9.5)
axes[0].set_ylabel('Leader (France)', fontsize=9.5)
axes[0].tick_params(axis='x', rotation=45)
axes[0].tick_params(axis='y', rotation=0)

# (B)
sns.heatmap(arg_reads_fra, ax=axes[1], cmap='YlGnBu', annot=True, fmt='d', linewidths=0.3,
            cbar_kws={'label': 'Event Count (15s Windows)'}, vmin=0)
axes[1].set_title('(B) Cross-Team: Argentina Reads France\n(Leader: Argentina Defender/Midfielder -> Follower: France Attacker)', fontsize=10.5, fontweight='bold', pad=8)
axes[1].set_xlabel('Follower (France)', fontsize=9.5)
axes[1].set_ylabel('Leader (Argentina)', fontsize=9.5)
axes[1].tick_params(axis='x', rotation=45)
axes[1].tick_params(axis='y', rotation=0)

# (C)
sns.heatmap(fra_reads_arg, ax=axes[2], cmap='YlGnBu', annot=True, fmt='d', linewidths=0.3,
            cbar_kws={'label': 'Event Count (15s Windows)'}, vmin=0)
axes[2].set_title('(C) Cross-Team: France Reads Argentina\n(Leader: France Defender/Midfielder -> Follower: Argentina Attacker)', fontsize=10.5, fontweight='bold', pad=8)
axes[2].set_xlabel('Follower (Argentina)', fontsize=9.5)
axes[2].set_ylabel('Leader (France)', fontsize=9.5)
axes[2].tick_params(axis='x', rotation=45)
axes[2].tick_params(axis='y', rotation=0)

plt.tight_layout()
fig2_path = 'results/figures/figure2_exact_abstract.png'
plt.savefig(fig2_path, dpi=200, bbox_inches='tight')
plt.close()
print(f'Saved {fig2_path}')
print('\n=== ALL EXACT FIGURES GENERATED SUCCESSFULLY! ===')
