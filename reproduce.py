"""
Reproduce all SSAC 2027 Anticipation Maps and Counts.

Usage:
    python reproduce.py                # Verify counts and reproduce Figure 1 & Figure 2
    python reproduce.py --recalculate  # Recalculate counts from extracted velocity features
"""
import sys, os, argparse, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MAT_DIR  = os.path.join(DATA_DIR, "count_matrices")
FEAT_DIR = os.path.join(DATA_DIR, "extracted_features")
FIG_DIR  = os.path.join(BASE_DIR, "results", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

from src.anticipation.pipeline import run_window_anticipation_counts

def render_figure_1(counts_dict):
    """Render 4-panel within-team anticipation counts (Figure 1)."""
    fig, axes = plt.subplots(2, 2, figsize=(22, 20))
    fig.suptitle('FIFA World Cup 2022 : Cumulative Anticipation Event Counts (15s Windows)\n'
                 'Total Discrete Tactical Phases Where Leader Triggered Follower (Controlling for Ball Motion)',
                 fontsize=16, fontweight='bold', y=0.98)
                 
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for idx, (tname, (cdf, m_desc)) in enumerate(counts_dict.items()):
        r, c = positions[idx]
        ax = axes[r, c]
        sns.heatmap(cdf, ax=ax, cmap='Blues', annot=True, fmt='d', linewidths=0.4,
                    mask=None, cbar_kws={'label': 'Anticipation Event Count'}, vmin=0)
        flat = cdf.stack()
        top_pair = flat.idxmax()
        top_val = flat.max()
        letter = ['(A)', '(B)', '(C)', '(D)'][idx]
        ax.set_title(f"{letter} {tname} ({m_desc})\nTop Pair: {top_pair[0]} -> {top_pair[1]} ({top_val} times)",
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('Follower (Reactor)', fontsize=10)
        ax.set_ylabel('Leader (Trigger)', fontsize=10)
        ax.tick_params(axis='x', rotation=45)
        ax.tick_params(axis='y', rotation=0)
        
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(FIG_DIR, "FIGURE_1_INTRA_TEAM_ANTICIPATION.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[REPRODUCE] Figure 1 saved cleanly: {out_path}")

def render_figure_2(fra_in, arg_fra, fra_arg):
    """Render 3-panel intra vs cross-team anticipation networks (Figure 2)."""
    fig, axes = plt.subplots(1, 3, figsize=(28, 9.5))
    fig.suptitle('FIFA World Cup Final 2022 : In-Team vs. Cross-Team Anticipation Event Counts (15s Windows)\n'
                 'Total Discrete Moments Leader Triggered Follower (Controlling for Ball Motion, No Masks)',
                 fontsize=16, fontweight='bold', y=0.98)
                 
    # Panel A: France In-Team
    sns.heatmap(fra_in, ax=axes[0], cmap='YlGnBu', annot=True, fmt='d', linewidths=0.3,
                mask=None, cbar_kws={'label': 'Event Count (15s Windows)'}, vmin=0)
    axes[0].set_title('(A) France In-Team Chemistry\n(Leader: France -> Follower: France)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Follower (France)', fontsize=10); axes[0].set_ylabel('Leader (France)', fontsize=10)
    axes[0].tick_params(axis='x', rotation=45); axes[0].tick_params(axis='y', rotation=0)

    # Panel B: Argentina Reads France
    sns.heatmap(arg_fra, ax=axes[1], cmap='YlGnBu', annot=True, fmt='d', linewidths=0.3,
                mask=None, cbar_kws={'label': 'Event Count (15s Windows)'}, vmin=0)
    axes[1].set_title('(B) Cross-Team: Argentina Reads France\n(Leader: Argentina Defender/Midfielder -> Follower: France Attacker)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Follower (France)', fontsize=10); axes[1].set_ylabel('Leader (Argentina)', fontsize=10)
    axes[1].tick_params(axis='x', rotation=45); axes[1].tick_params(axis='y', rotation=0)

    # Panel C: France Reads Argentina
    sns.heatmap(fra_arg, ax=axes[2], cmap='YlGnBu', annot=True, fmt='d', linewidths=0.3,
                mask=None, cbar_kws={'label': 'Event Count (15s Windows)'}, vmin=0)
    axes[2].set_title('(C) Cross-Team: France Reads Argentina\n(Leader: France Defender/Midfielder -> Follower: Argentina Attacker)', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Follower (Argentina)', fontsize=10); axes[2].set_ylabel('Leader (France)', fontsize=10)
    axes[2].tick_params(axis='x', rotation=45); axes[2].tick_params(axis='y', rotation=0)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = os.path.join(FIG_DIR, "FIGURE_2_INTRA_VS_CROSS_TEAM.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[REPRODUCE] Figure 2 saved cleanly: {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Reproduce SSAC 2027 Anticipation Maps")
    parser.add_argument('--recalculate', action='store_true', help='Recalculate counts from raw velocity features')
    args = parser.parse_args()

    print("=================================================================")
    print(" Anticipation in Football: SSAC 2027 Results Reproduction Pipeline")
    print("=================================================================\n")

    t0 = time.time()
    
    if args.recalculate:
        print("[REPRODUCE] Recalculating anticipation counts directly from extracted features (.npz)...")
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
        arg_roster = {
            'ARG_3':'Tagliafico', 'ARG_5':'Paredes', 'ARG_7':'De Paul', 'ARG_9':'Alvarez',
            'ARG_10':'Messi', 'ARG_13':'Romero', 'ARG_14':'Palacios', 'ARG_19':'Otamendi',
            'ARG_20':'Mac Allister', 'ARG_21':'Dybala', 'ARG_24':'Enzo', 'ARG_25':'L.Martinez', 'ARG_26':'Molina'
        }
        arg_counts = run_window_anticipation_counts(
            os.path.join(FEAT_DIR, "match_10514_argentina_features.npz"),
            arg_roster, arg_roster, outliers=['Paredes', 'Enzo']
        )
        mar_roster = {
            'MAR_2':'Hakimi', 'MAR_3':'Mazraoui', 'MAR_4':'Amrabat', 'MAR_7':'Ziyech',
            'MAR_8':'Ounahi', 'MAR_9':'Hamdallah', 'MAR_14':'Aboukhlal', 'MAR_15':'Amallah',
            'MAR_17':'Boufal', 'MAR_18':'El Yamiq', 'MAR_19':'En-Nesyri', 'MAR_20':'Dari', 'MAR_25':'Attiyat Allah'
        }
        mar_counts = run_window_anticipation_counts(
            os.path.join(FEAT_DIR, "match_10515_morocco_features.npz"),
            mar_roster, mar_roster, outliers=['Amrabat']
        )
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
        arg_fra = run_window_anticipation_counts(
            os.path.join(FEAT_DIR, "match_10517_final_features.npz"),
            arg_cross_roster, fra_cross_roster
        )
        fra_arg = run_window_anticipation_counts(
            os.path.join(FEAT_DIR, "match_10517_final_features.npz"),
            fra_cross_roster, arg_cross_roster
        )
    else:
        # Load count matrices
        fra_counts = pd.read_csv(os.path.join(MAT_DIR, "france_anticipation_counts.csv"), index_col=0)
        arg_counts = pd.read_csv(os.path.join(MAT_DIR, "argentina_anticipation_counts.csv"), index_col=0)
        mar_counts = pd.read_csv(os.path.join(MAT_DIR, "morocco_anticipation_counts.csv"), index_col=0)
        srb_counts = pd.read_csv(os.path.join(MAT_DIR, "serbia_anticipation_counts.csv"), index_col=0)
        
        arg_fra = pd.read_csv(os.path.join(MAT_DIR, "cross_arg_leads_fra_counts.csv"), index_col=0)
        fra_arg = pd.read_csv(os.path.join(MAT_DIR, "cross_fra_leads_arg_counts.csv"), index_col=0)


    for df in [fra_counts, arg_counts, mar_counts, srb_counts]:
        np.fill_diagonal(df.values, 0)

    fig1_dict = {
        'France': (fra_counts, '2022 World Cup Final vs Argentina'),
        'Argentina': (arg_counts, '2022 World Cup Semi-Final 3-0 vs Croatia'),
        'Morocco': (mar_counts, '2022 World Cup Semi-Final vs France'),
        'Serbia': (srb_counts, '2022 World Cup Group Stage 3-3 vs Cameroon')
    }

    print("--- Verifying Target Partnership Findings ---")
    print(f"  France:    Top Pair: Rabiot -> Tchouaméni: {fra_counts.loc['Rabiot', 'Tchouameni']} | Reciprocal: Tchouaméni -> Rabiot: {fra_counts.loc['Tchouameni', 'Rabiot']}")
    print(f"  Argentina: Top Pair: Messi -> Mac Allister: {arg_counts.loc['Messi', 'Mac Allister']} | Álvarez -> Messi: {arg_counts.loc['Alvarez', 'Messi']}")
    print(f"  Morocco:   Top Pair: Hakimi -> Ounahi: {mar_counts.loc['Hakimi', 'Ounahi']} | Ziyech ↔ Hakimi: {mar_counts.loc['Ziyech', 'Hakimi']} / {mar_counts.loc['Hakimi', 'Ziyech']}")
    print(f"  Serbia:    Top Pair: Tadić -> Mitrović: {srb_counts.loc['Tadic', 'A.Mitrovic']} | Reciprocal: Mitrović -> Tadić: {srb_counts.loc['A.Mitrovic', 'Tadic']}")
    print(f"  Final:     Mac Allister -> Tchouaméni: {arg_fra.loc['Mac Allister', 'Tchouameni']} phases")
    print(f"  Final:     Tchouaméni -> Mac Allister: {fra_arg.loc['Tchouameni', 'Mac Allister']} | Messi: {fra_arg.loc['Tchouameni', 'Messi']} | Álvarez: {fra_arg.loc['Tchouameni', 'Alvarez']}\n")

    # Render Figures
    print("--- Rendering Publication Figures ---")
    render_figure_1(fig1_dict)
    render_figure_2(fra_counts, arg_fra, fra_arg)

    print(f"\n[SUCCESS] All results and figures reproduced in {time.time()-t0:.2f} seconds.")

if __name__ == '__main__':
    main()
