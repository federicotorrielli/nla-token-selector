"""Zero-training keyword rule: predict the word that is mentioned most often (word boundary, case-insens) in the part's explanations."""
import sys, re, numpy as np
sys.path.insert(0, '/mnt/DATA/Nimbus/Projects/Research/nla-token-selector/token_analysis/v2_scratch/independent_check')
import lib
PARTS = ['open_first5', 'tail5', 'tail1', 'plast5', 'pfirst5', 'prand5', 'plast20', 'prand20', 'rfirst5', 'rlast5', 'rrand5', 'rrand20', 'rlast20', 'rand5_any']
rows = []
for kind in ('taboo', 'taboo20'):
    for model in ('g12', 'g27', 'l70', 'q7'):
        r = lib.build(kind, model); y = r['y']; words = sorted(set(y))
        pats = [re.compile(r'\b%s\b' % w, re.I) for w in words]
        rng = np.random.default_rng(0)
        for p in PARTS:
            docs = r['docs'].get(p)
            if docs is None: continue
            acc = []; hit = []; fp = []
            for d, t in zip(docs, y):
                if not d: continue
                c = np.array([len(pt.findall(d)) for pt in pats])
                ti = words.index(t)
                hit.append(c[ti] > 0)  # word present at all
                fp.append((c > 0).sum() - (c[ti] > 0))  # other candidate words mentioned
                if c.max() == 0: acc.append(1 / len(words)); continue
                best = np.flatnonzero(c == c.max()); acc.append(float(ti in best) / len(best))
            rows.append((kind, model, p, np.mean(acc), np.mean(hit), np.mean(fp), len(acc)))
with open(f'{lib.OUT}/keyword_taboo.tsv', 'w') as f:
    f.write('kind\tmodel\tpart\tkeyword_acc\tword_present\tother_words_mean\tn\n')
    for x in rows: f.write('\t'.join(f'{v:.3f}' if isinstance(v, float) else str(v) for v in x) + '\n')
