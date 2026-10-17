"""Paired discovery contrast; bank reused, not independent confirmation."""
import json
from pathlib import Path
import numpy as np
root = Path(__file__).parent
old = json.loads((root.parent / 'pr29-data-refresh/results/fresh_data-games.json').read_text())
new = json.loads((root / 'results/fresh_value_zero-games.json').read_text())
old = {(r['pair_id'], r['color']):r for r in old}
units = [[] for _ in range(16)]
for r in new:
    other = old[(r['pair_id'],r['color'])]
    assert other['opening'] == r['opening']
    assert other['score'] is not None and r['score'] is not None
    units[r['pair_id'] % 16].append(r['score'] - other['score'])
means = np.array([np.mean(v) for v in units])
boot = means[np.random.default_rng(162016).integers(0,16,(20000,16))].mean(1)
x = dict(gain=float(means.mean()), interval95=np.quantile(boot,[.025,.975]).tolist(), discovery_only=True)
(root/'results/paired-comparison.json').write_text(json.dumps(x,indent=2)+'\n')
print(x)
