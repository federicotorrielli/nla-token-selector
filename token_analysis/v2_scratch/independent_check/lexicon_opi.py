"""Training-free lexicon reader for OPI: (#sentiment words - #hate words) in the part's explanations -> AUROC for injected=sst2."""
import sys, re, numpy as np
sys.path.insert(0, '/mnt/DATA/Nimbus/Projects/Research/nla-token-selector/token_analysis/v2_scratch/independent_check')
import lib
from sklearn.metrics import roc_auc_score
S = re.compile(r'sentiment|review|movie|film|positive|negative|rating|opinion', re.I)
H = re.compile(r'hate|hateful|offensive|toxic|slur|racis|insult|abusive|derogator|vulgar|profan|harass|discriminat|insulting|tweet', re.I)
PARTS = ['open_first5', 'tail5', 'tail1', 'pfirst5', 'plast5', 'prand5', 'punct5', 'plast20', 'prand20']
rows = []
for kind in ('opi', 'opistart', 'opimiddle', 'opihistory'):
    for model in ('g12', 'g27', 'l70', 'q7'):
        r = lib.build(kind, model); y = r['y']; yb = (y == 'sst2').astype(int)
        tgt = np.array([c.split('__')[0] for c in r['case_ids']]); m = tgt == 'mrpc'
        for p in PARTS:
            docs = r['docs'].get(p)
            if docs is None or any(d is None for d in docs): continue
            s = np.array([len(S.findall(d)) - len(H.findall(d)) for d in docs], float)
            rows.append((kind, model, p, roc_auc_score(yb, s), roc_auc_score(yb[m], s[m])))
with open(f'{lib.OUT}/lexicon_opi.tsv', 'w') as f:
    f.write('kind\tmodel\tpart\tauroc_pooled\tauroc_mrpc_target\n')
    for x in rows: f.write('\t'.join(f'{v:.3f}' if isinstance(v, float) else str(v) for v in x) + '\n')
import polars as pl
pl.Config.set_tbl_rows(100); pl.Config.set_tbl_width_chars(200)
d = pl.read_csv(f'{lib.OUT}/lexicon_opi.tsv', separator='\t')
print(d.pivot(on='part', index=['kind', 'model'], values='auroc_mrpc_target').select(['kind', 'model'] + PARTS))
print(d.pivot(on='part', index=['kind', 'model'], values='auroc_pooled').select(['kind', 'model'] + PARTS))
