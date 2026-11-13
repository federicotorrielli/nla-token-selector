"""Liars leak probe with balanced label-free prefixes (minority share>=0.3) and within-subset AUROC."""
import sys, numpy as np, collections
sys.path.insert(0, '/mnt/DATA/Nimbus/Projects/Research/nla-token-selector/token_analysis/v2_scratch/independent_check')
import lib, polars as pl
from joblib import Parallel, delayed
kind, model = sys.argv[1], sys.argv[2]
MINFRAC = float(sys.argv[3]) if len(sys.argv) > 3 else 0.3
r = lib.build(kind, model); y = r['y']; yb = (y == np.unique(y)[-1]).astype(int)
sub = np.array([c.rsplit('_', 1)[0] for c in r['case_ids']])
meta, f = lib.read_meta(kind, model)
cids = meta['case_id'].to_numpy(); toks = meta['token'].to_list()
starts = np.r_[0, np.flatnonzero(cids[1:] != cids[:-1]) + 1]; ends = np.r_[starts[1:], len(cids)]
hs = []; cnt = collections.defaultdict(lambda: [0, 0])
for ci, (s, e) in enumerate(zip(starts, ends)):
    h = 0; hl = []
    for t in toks[s:e]:
        h = hash((h, t)); hl.append(h)
    hs.append(hl)
    for h in set(hl): cnt[h][yb[ci]] += 1
# also require the prefix to be label-balanced WITHIN subset: use global balance among cases sharing it
free = {h for h, (a, b) in cnt.items() if min(a, b) >= 8 and min(a, b) / (a + b) >= MINFRAC}
loc = [np.array([i for i, h in enumerate(hs[ci]) if h in free], int) + s for ci, s in enumerate(starts)]
nfree = np.array([len(x) for x in loc]); print(kind, model, 'free positions/case mean', nfree.mean(), 'frac cases>=20', (nfree >= 20).mean(), flush=True)
ex = pl.read_parquet(f, columns=['explanation'])['explanation'].to_numpy()
rng = np.random.default_rng(1); parts = {}
for k in (5, 20):
    for nm in ('rand', 'last'):
        docs = []
        for x in loc:
            if len(x) < k: docs.append(None); continue
            sel = np.sort(rng.choice(x, k, replace=False)) if nm == 'rand' else x[-k:]
            docs.append(lib.SEP.join(ex[i] or '' for i in sel))
        parts[f'leakbal_{nm}{k}'] = docs
fold = lib.folds_for('orig', 50, r['g'], yb)
def job(p):
    ok = np.array([d is not None for d in parts[p]])
    docs = [d if d is not None else '' for d in parts[p]]
    sc, oof = lib.cv_oof(docs, y, fold)
    oof[~ok] = np.nan
    return p, sc, lib.within_subset_auc(oof, yb, sub), ok.mean()
for p, sc, w, cov in Parallel(n_jobs=4)(delayed(job)(p) for p in parts):
    print(kind, model, p, 'pooled', round(sc, 3), 'within', round(w, 3), 'coverage', round(cov, 2), flush=True)
