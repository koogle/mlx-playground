"""Bound paired budget interaction when the censored neural game scores 0 or 1."""
import json
from pathlib import Path
import numpy as np
root = Path(__file__).with_name('results')
rows = {a: {(r['pair_id'], r['color']): r for r in json.loads((root / (a+'-games.json')).read_text())} for a in ['neural', 'zero']}
results = {}
for censored_score in [0, 1]:
    families = [[] for _ in range(16)]
    for key, zero in rows['zero'].items():
        neural = rows['neural'][key]
        assert zero['opening'] == neural['opening'] and zero['score'] is not None
        score = neural['score'] if neural['score'] is not None else censored_score
        families[key[0] % 16].append(zero['score'] - score)
    means = np.array([np.mean(f) for f in families])
    boot = means[np.random.default_rng(222016).integers(0, 16, (20000, 16))].mean(1)
    results[str(censored_score)] = dict(gain=float(means.mean()), interval95=np.quantile(boot, [.025, .975]).tolist())
(root / 'censor-sensitivity.json').write_text(json.dumps(results, indent=2)+'\n')
print(json.dumps(results))
