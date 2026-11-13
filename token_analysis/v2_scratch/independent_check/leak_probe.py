"""Leak probe: positions whose whole prefix text is shared by cases of BOTH classes (activation identical up to numerics).
Any reader signal on their explanations is batch-numerics leak. Usage: leak_probe.py kind model"""
import sys, numpy as np, collections
sys.path.insert(0, '/mnt/DATA/Nimbus/Projects/Research/nla-token-selector/token_analysis/v2_scratch/independent_check')
import lib, polars as pl
from joblib import Parallel, delayed
kind, model = sys.argv[1], sys.argv[2]
r = lib.build(kind, model)
y = r['y']; classes = np.unique(y); yb = (y == classes[-1]).astype(int)
meta, f = lib.read_meta(kind, model)
cids = meta['case_id'].to_numpy(); toks = meta['token'].to_list(); reg = meta['region'].to_numpy()
starts = np.r_[0, np.flatnonzero(cids[1:] != cids[:-1]) + 1]; ends = np.r_[starts[1:], len(cids)]
hs = []  # per case list of prefix hashes
cnt = collections.defaultdict(lambda: [0, 0])
for ci, (s, e) in enumerate(zip(starts, ends)):
    h = 0; hl = []
    for t in toks[s:e]:
        h = hash((h, t)); hl.append(h)
    hs.append(hl)
    for h in set(hl): cnt[h][yb[ci]] += 1
MIN = 5
free = {h for h, (a, b) in cnt.items() if a >= MIN and b >= MIN}
rng = np.random.default_rng(1)
loc = []
for ci, (s, e) in enumerate(zip(starts, ends)):
    idx = np.array([i for i, h in enumerate(hs[ci]) if h in free], int)
    loc.append(idx + s)
nfree = np.array([len(x) for x in loc])
print(kind, model, 'label-free positions per case: mean', nfree.mean(), 'min', nfree.min(), 'cases with >=5:', (nfree >= 5).mean())
ex = pl.read_parquet(f, columns=['explanation'])['explanation'].to_numpy()
parts = {}
for k in (1, 5, 20):
    for nm in ('rand', 'last', 'first'):
        docs = []
        for x in loc:
            if len(x) < k: docs.append(None); continue
            sel = np.sort(rng.choice(x, k, replace=False)) if nm == 'rand' else (x[-k:] if nm == 'last' else x[:k])
            docs.append(lib.SEP.join(ex[i] or '' for i in sel))
        parts[f'leak_{nm}{k}'] = docs
B = 50 if kind == 'liars' else 40
fold = {'orig': lib.folds_for('orig', B, r['g'], yb)}
if kind == 'liars' or kind == 'tt': fold['block'] = lib.folds_for('block', B, r['g'], yb)
if kind.startswith('opi'): fold['strict'] = lib.folds_for('orig', B, lib.strict_groups(kind, r['case_ids'], r['g']), yb)
tasks = [(p, s) for p in parts for s in fold]
res = Parallel(n_jobs=8)(delayed(lambda p, s: (p, s, lib.cv_score(parts[p], y, fold[s], 'lr_word')))(p, s) for p, s in tasks)
with open(f'{lib.OUT}/leak_{kind}_{model}.tsv', 'w') as fo:
    fo.write('kind\tmodel\tscheme\tpart\tscore\tmean_nfree\n')
    for p, s, sc in res: fo.write(f'{kind}\t{model}\t{s}\t{p}\t{sc:.4f}\t{nfree.mean():.1f}\n')
