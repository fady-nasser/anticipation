"""
Extract 5Hz velocity profiles from raw tracking data for the 4 paper matches:
- 10517: France vs Argentina (WC Final 2022)
- 10514: Argentina vs Croatia (WC Semi-Final 2022)
- 10515: France vs Morocco (WC Semi-Final 2022)
- 3840:  Serbia vs Cameroon (WC Group Stage 2022)

Extracted features:
- period, periodElapsedTime (t)
- ball_x, ball_y, ball_vx, ball_vy
- For each player: {prefix}_{jersey}_vx, {prefix}_{jersey}_vy
No raw player tracking coordinates (x, y) are exported, ensuring compliance with data redistribution policies.
"""
import sys, os, bz2, json, time
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

TRACK_DIR = r"data raw\FIFA World Cup 2022\Tracking Data"
OUT_DIR   = os.path.join("data", "extracted_features")
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_HZ = 5
SKIP = 5

MATCH_CONFIGS = {
    'match_10517_final_features': {
        'gid': '10517',
        'home_prefix': 'ARG',
        'away_prefix': 'FRA'
    },
    'match_10514_argentina_features': {
        'gid': '10514',
        'home_prefix': 'ARG',
        'away_prefix': 'CRO'
    },
    'match_10515_morocco_features': {
        'gid': '10515',
        'home_prefix': 'FRA',
        'away_prefix': 'MAR'
    },
    'match_3840_serbia_features': {
        'gid': '3840',
        'home_prefix': 'CMR',
        'away_prefix': 'SRB'
    }
}

for out_name, cfg in MATCH_CONFIGS.items():
    t0 = time.time()
    raw_path = os.path.join(TRACK_DIR, f"{cfg['gid']}.jsonl.bz2")
    out_path = os.path.join(OUT_DIR, f"{out_name}.npz")
    
    if os.path.exists(out_path):
        print(f"[SKIP] {out_path} already exists ({os.path.getsize(out_path)/(1024*1024):.2f} MB).")
        continue
        
    print(f"\nProcessing Match {cfg['gid']} -> {out_name}...")
    frames = []
    with bz2.open(raw_path, 'rt', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i % SKIP != 0: continue
            obj = json.loads(line)
            row = {'period': obj['period'], 't': obj['periodElapsedTime']}
            if obj.get('balls'):
                b = obj['balls'][0]
                row['ball_x'] = b.get('x', np.nan)
                row['ball_y'] = b.get('y', np.nan)
            else:
                row['ball_x'] = np.nan; row['ball_y'] = np.nan
                
            for p in obj.get('homePlayers', []):
                j = p['jerseyNum']
                vis = p.get('visibility', 'ESTIMATED')
                row[f"{cfg['home_prefix']}_{j}_x"] = p['x'] if vis == 'VISIBLE' else np.nan
                row[f"{cfg['home_prefix']}_{j}_y"] = p['y'] if vis == 'VISIBLE' else np.nan
                
            for p in obj.get('awayPlayers', []):
                j = p['jerseyNum']
                vis = p.get('visibility', 'ESTIMATED')
                row[f"{cfg['away_prefix']}_{j}_x"] = p['x'] if vis == 'VISIBLE' else np.nan
                row[f"{cfg['away_prefix']}_{j}_y"] = p['y'] if vis == 'VISIBLE' else np.nan
            frames.append(row)
            
    df = pd.DataFrame(frames)
    print(f"Loaded {len(df)} frames at 5Hz.")
    
    # Compute velocity features
    vel_dict = {
        'period': df['period'].values.astype(np.int32),
        't': df['t'].values.astype(np.float32),
        'ball_x': df['ball_x'].values.astype(np.float32),
        'ball_y': df['ball_y'].values.astype(np.float32),
        'ball_vx': (df['ball_x'].diff().fillna(0) * TARGET_HZ).values.astype(np.float32),
        'ball_vy': (df['ball_y'].diff().fillna(0) * TARGET_HZ).values.astype(np.float32),
    }
    
    pc = (df['period'].diff().fillna(0) != 0).values
    vel_dict['ball_vx'][pc] = np.nan
    vel_dict['ball_vy'][pc] = np.nan
    
    for c in df.columns:
        if c.endswith('_x') and not c.startswith('ball_'):
            base = c[:-2]
            yc = f"{base}_y"
            vx = (df[c].diff().fillna(0) * TARGET_HZ).values.astype(np.float32)
            vy = (df[yc].diff().fillna(0) * TARGET_HZ).values.astype(np.float32)
            vx[pc] = np.nan
            vy[pc] = np.nan
            vel_dict[f"{base}_vx"] = vx
            vel_dict[f"{base}_vy"] = vy
            
    np.savez_compressed(out_path, **vel_dict)
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"Successfully saved {out_path} ({size_mb:.2f} MB in {time.time()-t0:.1f}s)")

print("\nAll extracted features ready in data/extracted_features/!")
