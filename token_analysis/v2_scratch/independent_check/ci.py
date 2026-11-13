"""Paired bootstrap (resampling cases) of pooled out-of-fold AUROC: tail5 minus other parts."""
import sys, numpy as np
sys.path.insert(0, '/mnt/DATA/Nimbus/Projects/Research/nla-token-selector/token_analysis/v2_scratch/independent_check')
import lib
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed
OTH = ['punct5', 'plast5', 'plast20', 'prand5', 'pfirst5', 'open_first5', 'rlast5', 'rrand5']
out = []
for kind, model in [('liars', 'g27'), ('liars', 'l70'), ('tt', 'g12'), ('tt', 'g27'), ('tt', 'l70'), ('tt', 'q7')]:
    r = lib.build(kind, model); y = r['y']; yb = (y == np.unique(y)[-1]).astype(int)
    fold = lib.folds_for('orig', 50, r['g'], yb)
    P = ['tail5'] + OTH
    oofs = dict(zip(P, Parallel(n_jobs=7)(delayed(lambda p: lib.cv_oof(r['docs'][p], y, fold)[1])(p) for p in P)))
    rng = np.random.default_rng(0); n = len(yb); B = []
    for _ in range(300):
        i = rng.integers(0, n, n)
        if yb[i].min() == yb[i].max(): continue
        B.append(i)
    for p in OTH:
        ok = ~np.isnan(oofs[p]) & ~np.isnan(oofs['tail5'])
        def D(i):
            i = i[ok[i]]
            return roc_auc_score(yb[i], oofs['tail5'][i]) - roc_auc_score(yb[i], oofs[p][i])
        d = [D(i) for i in B]
        pt = D(np.arange(n))
        out.append((kind, model, p, pt, np.percentile(d, 2.5), np.percentile(d, 97.5)))
        print(out[-1], flush=True)
with open(f'{lib.OUT}/ci_tail5_minus.tsv', 'w') as f:
    f.write('kind\tmodel\tother\tdiff_pooled_auroc\tci_lo\tci_hi\n')
    for x in out: f.write('\t'.join(f'{v:.4f}' if isinstance(v, float) else str(v) for v in x) + '\n')
