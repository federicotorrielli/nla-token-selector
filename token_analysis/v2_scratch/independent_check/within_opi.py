"""OPI: AUROC within target-task subsets (removes target-task confound) for key parts. Usage: within_opi.py model"""
import sys, numpy as np
sys.path.insert(0, '/mnt/DATA/Nimbus/Projects/Research/nla-token-selector/token_analysis/v2_scratch/independent_check')
import lib
from joblib import Parallel, delayed
model = sys.argv[1]
PARTS = ['open_first5', 'tail5', 'tail1', 'tailtok1', 'tailtok3', 'pfirst5', 'plast5', 'prand5', 'punct5', 'plast20', 'prand20', 'pfirst20', 'inter5']
rows = []
for kind in ('opi', 'opistart', 'opimiddle', 'opihistory'):
    r = lib.build(kind, model); y = r['y']; yb = (y == np.unique(y)[-1]).astype(int)
    tgt = np.array([c.split('__')[0] for c in r['case_ids']])
    fold = lib.folds_for('orig', 40, lib.strict_groups(kind, r['case_ids'], r['g']), yb)
    def job(p):
        if p not in r['docs'] or any(d is None for d in r['docs'][p]): return None
        sc, oof = lib.cv_oof(r['docs'][p], y, fold, 'lr_word')
        w = {t: __import__('sklearn.metrics', fromlist=['x']).roc_auc_score(yb[tgt == t], oof[tgt == t]) for t in np.unique(tgt)}
        return p, sc, w
    for out in Parallel(n_jobs=13)(delayed(job)(p) for p in PARTS):
        if out is None: continue
        p, sc, w = out
        rows.append((kind, model, p, sc, w.get('sst2', np.nan), w.get('mrpc', np.nan)))
with open(f'{lib.OUT}/within_opi_{model}.tsv', 'w') as f:
    f.write('kind\tmodel\tpart\tpooled\twithin_target_sst2\twithin_target_mrpc\n')
    for x in rows: f.write('\t'.join(f'{v:.4f}' if isinstance(v, float) else str(v) for v in x) + '\n')
