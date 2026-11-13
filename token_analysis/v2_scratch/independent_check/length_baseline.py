"""Metadata-only baselines: token counts per region (+ row rank) -> label, same CV. Shows how much label info lengths/order carry."""
import sys, numpy as np, polars as pl
sys.path.insert(0, '/mnt/DATA/Nimbus/Projects/Research/nla-token-selector/token_analysis/v2_scratch/independent_check')
import lib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
rows = []
for kind, models in [('liars', ['g27', 'l70']), ('tt', ['g12', 'q7']), ('opistart', ['g12']), ('opimiddle', ['g12']), ('opihistory', ['g12']), ('opi', ['g12']), ('taboo20', ['g12', 'l70']), ('taboo', ['g12'])]:
    for m in models:
        meta, f = lib.read_meta(kind, m)
        c = meta.group_by('case_id', maintain_order=True).agg([pl.len().alias('n')] + [(pl.col('region') == rg).sum().alias(rg) for rg in ('template', 'system', 'user', 'assistant_prior', 'assistant')])
        ids = c['case_id'].to_numpy()
        r = lib.build(kind, m); y = r['y']; assert (r['case_ids'] == ids).all()
        classes = np.unique(y); binary = len(classes) == 2; yb = (y == classes[-1]).astype(int) if binary else y
        X = c.select(['n', 'template', 'system', 'user', 'assistant_prior', 'assistant']).to_numpy().astype(float)
        for name, XX in [('lengths', X), ('lengths+rank', np.c_[X, np.arange(len(y))]), ('rank_only', np.arange(len(y))[:, None].astype(float))]:
            for sch in ('orig', 'block'):
                if sch == 'block' and not (kind in ('liars', 'tt')): continue
                fold = lib.folds_for(sch, 50 if kind == 'liars' else 40, r['g'], yb)
                sc = []
                for tr, te in fold:
                    g = HistGradientBoostingClassifier(max_iter=100).fit(XX[tr], yb[tr])
                    if binary:
                        if len(np.unique(yb[te])) == 2: sc.append(roc_auc_score(yb[te], g.predict_proba(XX[te])[:, 1]))
                    else: sc.append(accuracy_score(yb[te], g.predict(XX[te])))
                rows.append((kind, m, sch, name, np.mean(sc)))
                print(rows[-1], flush=True)
with open(f'{lib.OUT}/length_baseline.tsv', 'w') as f:
    f.write('kind\tmodel\tscheme\tfeatures\tscore\n')
    for x in rows: f.write('\t'.join(f'{v:.4f}' if isinstance(v, float) else str(v) for v in x) + '\n')
