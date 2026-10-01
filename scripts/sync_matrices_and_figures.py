import os, sys, time
import numpy as np
import pandas as pd
from numpy.linalg import lstsq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
FEAT_DIR = os.path.join(BASE_DIR, "data", "extracted_features")
MAT_DIR = os.path.join(BASE_DIR, "data", "count_matrices")
FIG_DIR = os.path.join(BASE_DIR, "results", "figures")

from src.anticipation.pipeline import run_window_anticipation_counts


print("=== Computing Ground Truth Matrices from .npz Feature Files ===")

# 1. France Final (13 players)
t0 = time.time()
fra_roster = {
    'FRA_4':'Varane', 'FRA_5':'Kounde', 'FRA_7':'Griezmann', 'FRA_8':'Tchouameni',
    'FRA_9':'Giroud', 'FRA_10':'Mbappe', 'FRA_11':'Dembele', 'FRA_12':'KoloMuani',
    'FRA_14':'Rabiot', 'FRA_18':'Upamecano', 'FRA_20':'Coman', 'FRA_22':'THernandez',
    'FRA_25':'Camavinga', 'FRA_26':'M.Thuram'
}
fra_counts = run_window_anticipation_counts(
    os.path.join(FEAT_DIR, "match_10517_final_features.npz"),
    fra_roster, fra_roster, outliers=['Kounde']
)
np.fill_diagonal(fra_counts.values, 0)
fra_counts.to_csv(os.path.join(MAT_DIR, "france_anticipation_counts.csv"))
print(f"France saved: shape {fra_counts.shape}, Rabiot->Tchouameni = {fra_counts.loc['Rabiot', 'Tchouameni']} ({time.time()-t0:.1f}s)")

# 2. Argentina Semi-Final (10 players)
t_team = time.time()
arg_roster = {
    'ARG_3':'Tagliafico', 'ARG_5':'Paredes', 'ARG_7':'De Paul', 'ARG_9':'Alvarez',
    'ARG_10':'Messi', 'ARG_13':'Romero', 'ARG_14':'Palacios', 'ARG_19':'Otamendi',
    'ARG_20':'Mac Allister', 'ARG_21':'Dybala', 'ARG_24':'Enzo', 'ARG_25':'L.Martinez', 'ARG_26':'Molina'
}
arg_counts = run_window_anticipation_counts(
    os.path.join(FEAT_DIR, "match_10514_argentina_features.npz"),
    arg_roster, arg_roster, outliers=['Paredes', 'Enzo']
)
np.fill_diagonal(arg_counts.values, 0)
arg_counts.to_csv(os.path.join(MAT_DIR, "argentina_anticipation_counts.csv"))
print(f"Argentina saved: shape {arg_counts.shape}, Messi->Mac Allister = {arg_counts.loc['Messi', 'Mac Allister']} ({time.time()-t_team:.1f}s)")

# 3. Morocco Semi-Final (12 players)
t_team = time.time()
mar_roster = {
    'MAR_2':'Hakimi', 'MAR_3':'Mazraoui', 'MAR_4':'Amrabat', 'MAR_7':'Ziyech',
    'MAR_8':'Ounahi', 'MAR_9':'Hamdallah', 'MAR_14':'Aboukhlal', 'MAR_15':'Amallah',
    'MAR_17':'Boufal', 'MAR_18':'El Yamiq', 'MAR_19':'En-Nesyri', 'MAR_20':'Dari', 'MAR_25':'Attiyat Allah'
}
mar_counts = run_window_anticipation_counts(
    os.path.join(FEAT_DIR, "match_10515_morocco_features.npz"),
    mar_roster, mar_roster, outliers=['Amrabat']
)
np.fill_diagonal(mar_counts.values, 0)
mar_counts.to_csv(os.path.join(MAT_DIR, "morocco_anticipation_counts.csv"))
print(f"Morocco saved: shape {mar_counts.shape}, Hakimi->Ounahi = {mar_counts.loc['Hakimi', 'Ounahi']} ({time.time()-t_team:.1f}s)")

# 4. Serbia Group Stage (14 players)
t_team = time.time()
srb_roster = {
    'SRB_2':'Pavlovic', 'SRB_4':'Milenkovic', 'SRB_5':'Veljkovic', 'SRB_6':'Maksimovic',
    'SRB_7':'Radonjic', 'SRB_9':'A.Mitrovic', 'SRB_10':'Tadic', 'SRB_13':'S.Mitrovic',
    'SRB_14':'Zivkovic', 'SRB_16':'Lukic', 'SRB_17':'Kostic', 'SRB_20':'S.Milinkovic',
    'SRB_21':'Djuricic', 'SRB_26':'Grujic'
}
srb_counts = run_window_anticipation_counts(
    os.path.join(FEAT_DIR, "match_3840_serbia_features.npz"),
    srb_roster, srb_roster, outliers=[]
)
np.fill_diagonal(srb_counts.values, 0)
srb_counts.to_csv(os.path.join(MAT_DIR, "serbia_anticipation_counts.csv"))
print(f"Serbia saved: shape {srb_counts.shape}, Tadic->A.Mitrovic = {srb_counts.loc['Tadic', 'A.Mitrovic']} ({time.time()-t_team:.1f}s)")

# 5. Cross-Team Final (9x13 and 13x9)
t_team = time.time()
arg_cross_roster = {
    'ARG_3':'Tagliafico', 'ARG_7':'De Paul', 'ARG_9':'Alvarez', 'ARG_10':'Messi',
    'ARG_13':'Romero', 'ARG_19':'Otamendi', 'ARG_20':'Mac Allister', 'ARG_21':'Dybala',
    'ARG_26':'Molina'
}
fra_cross_roster = {
    'FRA_4':'Varane', 'FRA_7':'Griezmann', 'FRA_8':'Tchouameni', 'FRA_9':'Giroud',
    'FRA_10':'Mbappe', 'FRA_11':'Dembele', 'FRA_12':'KoloMuani', 'FRA_14':'Rabiot',
    'FRA_18':'Upamecano', 'FRA_20':'Coman', 'FRA_22':'T.Hernandez', 'FRA_25':'Camavinga', 'FRA_26':'M.Thuram'
}
arg_reads_fra = run_window_anticipation_counts(
    os.path.join(FEAT_DIR, "match_10517_final_features.npz"),
    arg_cross_roster, fra_cross_roster
)
fra_reads_arg = run_window_anticipation_counts(
    os.path.join(FEAT_DIR, "match_10517_final_features.npz"),
    fra_cross_roster, arg_cross_roster
)
arg_reads_fra.to_csv(os.path.join(MAT_DIR, "cross_arg_leads_fra_counts.csv"))
fra_reads_arg.to_csv(os.path.join(MAT_DIR, "cross_fra_leads_arg_counts.csv"))
print(f"Cross-team saved: Mac Allister->Tchouameni = {arg_reads_fra.loc['Mac Allister', 'Tchouameni']} ({time.time()-t_team:.1f}s)")

# Re-render Figure 1 cleanly
fig, axes = plt.subplots(2, 2, figsize=(22, 20))
fig.suptitle('FIFA World Cup 2022 : Cumulative Anticipation Event Counts (15s Windows)\n'
             'Total Discrete Tactical Phases Where Leader Triggered Follower (Controlling for Ball Motion)',
             fontsize=16, fontweight='bold', y=0.98)

teams = [
    ('(A) France (2022 WC Final vs Argentina)', fra_counts),
    ('(B) Argentina (2022 WC Semi-Final 3-0 vs Croatia)', arg_counts),
    ('(C) Morocco (2022 WC Semi-Final vs France)', mar_counts),
    ('(D) Serbia (2022 WC Group Stage 3-3 vs Cameroon)', srb_counts),
]
positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

for idx, (title, cdf) in enumerate(teams):
    r, c = positions[idx]
    ax = axes[r, c]
    sns.heatmap(cdf, ax=ax, cmap='Blues', annot=True, fmt='d', linewidths=0.4,
                mask=None, cbar_kws={'label': 'Anticipation Event Count'}, vmin=0)
    flat = cdf.stack()
    top_pair = flat.idxmax()
    top_val = flat.max()
    ax.set_title(f"{title}\nTop Pair: {top_pair[0]} -> {top_pair[1]} ({top_val} times)", fontsize=12, fontweight='bold')
    ax.set_xlabel('Follower (Reactor)', fontsize=10)
    ax.set_ylabel('Leader (Trigger)', fontsize=10)
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='y', rotation=0)

plt.tight_layout(rect=[0, 0, 1, 0.96])
fig1_path = os.path.join(FIG_DIR, "FIGURE_1_INTRA_TEAM_ANTICIPATION.png")
plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure 1 generated: {fig1_path}")

# Re-render Figure 2 cleanly
fig, axes = plt.subplots(1, 3, figsize=(28, 9.5))
fig.suptitle('FIFA World Cup Final 2022 : In-Team vs. Cross-Team Anticipation Event Counts (15s Windows)\n'
             'Total Discrete Moments Leader Triggered Follower (Controlling for Ball Motion, No Masks)',
             fontsize=16, fontweight='bold', y=0.98)

# Panel A: France In-Team
sns.heatmap(fra_counts, ax=axes[0], cmap='YlGnBu', annot=True, fmt='d', linewidths=0.3,
            mask=None, cbar_kws={'label': 'Event Count (15s Windows)'}, vmin=0)
axes[0].set_title('(A) France In-Team Chemistry\n(Leader: France -> Follower: France)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Follower (France)', fontsize=10); axes[0].set_ylabel('Leader (France)', fontsize=10)
axes[0].tick_params(axis='x', rotation=45); axes[0].tick_params(axis='y', rotation=0)

# Panel B: Argentina Reads France
sns.heatmap(arg_reads_fra, ax=axes[1], cmap='YlGnBu', annot=True, fmt='d', linewidths=0.3,
            mask=None, cbar_kws={'label': 'Event Count (15s Windows)'}, vmin=0)
axes[1].set_title('(B) Cross-Team: Argentina Reads France\n(Leader: Argentina Defender/Midfielder -> Follower: France Attacker)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Follower (France)', fontsize=10); axes[1].set_ylabel('Leader (Argentina)', fontsize=10)
axes[1].tick_params(axis='x', rotation=45); axes[1].tick_params(axis='y', rotation=0)

# Panel C: France Reads Argentina
sns.heatmap(fra_reads_arg, ax=axes[2], cmap='YlGnBu', annot=True, fmt='d', linewidths=0.3,
            mask=None, cbar_kws={'label': 'Event Count (15s Windows)'}, vmin=0)
axes[2].set_title('(C) Cross-Team: France Reads Argentina\n(Leader: France Defender/Midfielder -> Follower: Argentina Attacker)', fontsize=12, fontweight='bold')
axes[2].set_xlabel('Follower (Argentina)', fontsize=10); axes[2].set_ylabel('Leader (France)', fontsize=10)
axes[2].tick_params(axis='x', rotation=45); axes[2].tick_params(axis='y', rotation=0)

plt.tight_layout(rect=[0, 0, 1, 0.94])
fig2_path = os.path.join(FIG_DIR, "FIGURE_2_INTRA_VS_CROSS_TEAM.png")
plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure 2 generated: {fig2_path}")

print(f"\nALL DATA AND FIGURES RECOMPUTED AND SYNCHRONIZED IN {time.time()-t0:.1f}s!")
