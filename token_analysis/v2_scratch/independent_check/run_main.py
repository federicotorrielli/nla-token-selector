"""Main comparison: every part x scheme x reader for one (kind, model). Usage: run_main.py kind model [schemes] [readers]"""
import sys, numpy as np, time
sys.path.insert(0, '/mnt/DATA/Nimbus/Projects/Research/nla-token-selector/token_analysis/v2_scratch/independent_check')
import lib
from joblib import Parallel, delayed

kind, model = sys.argv[1], sys.argv[2]
schemes = (sys.argv[3] if len(sys.argv) > 3 else 'orig,strict').split(',')
readers = (sys.argv[4] if len(sys.argv) > 4 else 'lr_word').split(',')
r = lib.build(kind, model)
y = r['y']; classes = np.unique(y); binary = len(classes) == 2
yb = (y == classes[-1]).astype(int) if binary else y
sub = np.array([c.rsplit('_', 1)[0] for c in r['case_ids']]) if kind == 'liars' else None
B = {'liars': 50, 'tt': 40}.get(kind, 40)
fold = {}
for s in schemes:
    if s == 'orig': fold[s] = lib.folds_for('orig', B, r['g'], yb)
    elif s == 'strict': fold[s] = lib.folds_for('orig', B, lib.strict_groups(kind, r['case_ids'], r['g']), yb)
    elif s == 'block': fold[s] = lib.folds_for('block', B, r['g'], yb)

def job(part, scheme, reader, src):
    docs = r[src][part]
    if reader == 'raw_char': rd = 'lr_char'
    else: rd = reader
    if binary and sub is not None:
        try:
            sc, oof = lib.cv_oof([d for d in docs], y, fold[scheme], rd)
        except Exception as e:
            return (part, scheme, reader, src, float('nan'), float('nan'))
        return (part, scheme, reader, src, sc, lib.within_subset_auc(oof, yb, sub))
    return (part, scheme, reader, src, lib.cv_score(docs, y, fold[scheme], rd), float('nan'))

tasks = []
for s in schemes:
    for p in r['docs']:
        for rd in readers: tasks.append((p, s, rd, 'docs'))
    for p in ('plast5', 'plast20', 'pfirst5', 'prand5', 'tail5', 'ALLPROMPT'):
        if p in r['raw']: tasks.append((p, s, 'raw_char', 'raw'))
t = time.time()
res = Parallel(n_jobs=14)(delayed(job)(*a) for a in tasks)
with open(f'{lib.OUT}/main_{kind}_{model}{'' if readers==['lr_word'] else '_r'+'-'.join(readers)}.tsv', 'w') as f:
    f.write('kind\tmodel\tscheme\treader\tsrc\tpart\tscore\twithin\tn\n')
    for part, s, rd, src, sc, w in res:
        n = sum(d is not None and len(d) > 0 for d in r[src][part])
        f.write(f'{kind}\t{model}\t{s}\t{rd}\t{src}\t{part}\t{sc:.4f}\t{w:.4f}\t{n}\n')
print(kind, model, 'done', time.time() - t)
