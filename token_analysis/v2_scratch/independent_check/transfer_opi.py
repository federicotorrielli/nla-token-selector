"""Cross-dataset transfer: train TF-IDF reader on one OPI placement, test on another (batch artifacts do not transfer)."""
import sys, numpy as np
sys.path.insert(0, '/mnt/DATA/Nimbus/Projects/Research/nla-token-selector/token_analysis/v2_scratch/independent_check')
import lib
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed
KINDS = ['opi', 'opistart', 'opimiddle', 'opihistory']
PARTS = ['open_first5', 'tail5', 'tail1', 'pfirst5', 'plast5', 'prand5', 'punct5', 'plast20', 'prand20']
def run(model):
    R = {k: lib.build(k, model) for k in KINDS}
    jobs = [(a, b, p) for a in KINDS for b in KINDS if a != b for p in PARTS if p in R[a]['docs'] and p in R[b]['docs'] and not any(d is None for d in R[a]['docs'][p] + R[b]['docs'][p])]
    def j(a, b, p):
        m = lib.make_reader('lr_word'); ya = (R[a]['y'] == 'sst2').astype(int); yb = (R[b]['y'] == 'sst2').astype(int)
        m.fit(np.array(R[a]['docs'][p], dtype=object), ya)
        return model, a, b, p, roc_auc_score(yb, m.predict_proba(np.array(R[b]['docs'][p], dtype=object))[:, 1])
    return Parallel(n_jobs=12)(delayed(j)(*x) for x in jobs)
rows = []
for m in ('g12', 'g27', 'l70', 'q7'): rows += run(m)
with open(f'{lib.OUT}/transfer_opi.tsv', 'w') as f:
    f.write('model\ttrain\ttest\tpart\tauroc\n')
    for x in rows: f.write('\t'.join(f'{v:.4f}' if isinstance(v, float) else str(v) for v in x) + '\n')
import polars as pl
pl.Config.set_tbl_rows(100); pl.Config.set_tbl_width_chars(200)
d = pl.read_csv(f'{lib.OUT}/transfer_opi.tsv', separator='\t')
print(d.group_by('test', 'part').agg(pl.col('auroc').mean().round(3)).pivot(on='part', index='test', values='auroc').select(['test'] + PARTS))
print(d.filter(pl.col('train') == 'opistart').group_by('test', 'part').agg(pl.col('auroc').mean().round(3)).pivot(on='part', index='test', values='auroc').select(['test'] + PARTS))
