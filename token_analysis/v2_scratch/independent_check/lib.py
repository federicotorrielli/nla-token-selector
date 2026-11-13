"""Independent re-implementation: loading, part definitions, readers, CV schemes."""
import os, re, hashlib, pickle, numpy as np, polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, accuracy_score

ROOT = '/mnt/DATA/Nimbus/Projects/Research/nla-token-selector/results/bridge'
OUT = '/mnt/DATA/Nimbus/Projects/Research/nla-token-selector/token_analysis/v2_scratch/independent_check'
CONTENT = ('system', 'user', 'assistant_prior')
PUNCT = {'.', '?', '!', ':', ';', ','}
SEP = ' \n '


def case_meta(kind, cid):
    """label and original-style group for a case."""
    if kind.startswith('opi'):
        t, i, rest = cid.split('__')
        return i, f'{t}_{rest.rsplit("_", 1)[1]}'
    if kind == 'liars':
        return None, cid
    if kind == 'tt':
        return int(cid.endswith('_attack')), cid.split('_')[0]
    if kind == 'misalign':
        a, p, s = cid.split('__')
        return int(a != 'base'), p
    w, p, s = cid.split('__')  # taboo, taboo20
    return w, p


def read_meta(kind, model):
    f = f'{ROOT}/all_{kind}_{model}.parquet'
    return pl.read_parquet(f, columns=['case_id', 'tok_idx', 'token', 'region', 'label']), f


def part_indices(reg, toks, rng, ks):
    """Return dict part -> array of local indices (sorted) for one case."""
    n = len(reg); idx = np.arange(n)
    isc = np.isin(reg, CONTENT); isr = reg == 'assistant'
    ci = idx[isc]; ri = idx[isr]
    out = {}
    if len(ci) == 0:
        return out
    fc, lc = ci[0], ci[-1]
    fo = ri[0] if len(ri) else n
    op = idx[:fc]; tail = idx[lc + 1:fo]
    mid_tmpl = idx[(reg == 'template') & (idx > fc) & (idx < lc)]
    for k in ks:
        out[f'open_first{k}'] = op[:k]
        out[f'open_last{k}'] = op[-k:]
        out[f'tail{k}'] = tail[-k:]
        out[f'pfirst{k}'] = ci[:k]
        out[f'plast{k}'] = ci[-k:]
        m = ci[k:-k]
        out[f'prand{k}'] = np.sort(rng.choice(m, k, replace=False)) if len(m) >= k else ci[:0]
        out[f'inter{k}'] = mid_tmpl[-k:]
        if len(ri):
            out[f'rfirst{k}'] = ri[:k]; out[f'rlast{k}'] = ri[-k:]
            m = ri[k:-k]
            out[f'rrand{k}'] = np.sort(rng.choice(m, k, replace=False)) if len(m) >= k else ri[:0]
        # sentence-final punctuation tokens in prompt content
        pm = np.array([t.replace('▁', '').replace('Ġ', '').strip() in PUNCT for t in (toks[i] for i in ci)], bool)
        pi = ci[pm]
        out[f'punct{k}'] = pi[-k:]
    for j in range(1, 6):  # single template token, offset from the end of the tail / from start of open
        out[f'tailtok{j}'] = tail[-j:][:1] if len(tail) >= j else tail[:0]
        out[f'opentok{j}'] = op[j - 1:j]
    end = (ri[-1] + 1) if len(ri) else n
    out['rand5_any'] = np.sort(rng.choice(end, min(5, end), replace=False))
    return out


def build(kind, model, ks=(1, 2, 5, 10, 20), seed=0, cache=True):
    """Return dict with per-case labels, groups, row rank, n tokens, part docs (explanations) and part raw text."""
    cp = f'{OUT}/cache_{kind}_{model}.pkl'
    if cache and os.path.exists(cp):
        return pickle.load(open(cp, 'rb'))
    meta, f = read_meta(kind, model)
    meta = meta.with_row_index('row')
    reg_all = meta['region'].to_numpy(); tok_all = meta['token'].to_list(); lab_all = meta['label'].to_numpy()
    cids = meta['case_id'].to_numpy()
    starts = np.r_[0, np.flatnonzero(cids[1:] != cids[:-1]) + 1]; ends = np.r_[starts[1:], len(cids)]
    rng = np.random.default_rng(seed)
    case_ids, ys, gs, ns, glob = [], [], [], [], {}
    for s, e in zip(starts, ends):
        cid = cids[s]; lab, grp = case_meta(kind, cid)
        if kind == 'liars':
            lab = int(lab_all[s:e].max())
        ps = part_indices(reg_all[s:e], tok_all[s:e], rng, ks)
        for p, ix in ps.items():
            glob.setdefault(p, {})[len(case_ids)] = ix + s
        # constant-prefix / all-prompt handled separately
        case_ids.append(cid); ys.append(lab); gs.append(grp); ns.append(e - s)
    need = np.unique(np.concatenate([v for d in glob.values() for v in d.values() if len(v)]))
    ex = pl.read_parquet(f, columns=['explanation'])['explanation'].to_numpy()
    exd = {int(i): (ex[i] or '') for i in need}
    nc = len(case_ids)
    docs, raw = {}, {}
    for p, d in glob.items():
        docs[p] = [SEP.join(exd[int(i)] for i in d[c]) if c in d and len(d[c]) else None for c in range(nc)]
        raw[p] = [' '.join(tok_all[int(i)] for i in d[c]) if c in d and len(d[c]) else None for c in range(nc)]
    # all-prompt raw text
    allp = []
    for s, e in zip(starts, ends):
        m = np.isin(reg_all[s:e], CONTENT); allp.append(' '.join(t for t, mm in zip(tok_all[s:e], m) if mm))
    raw['ALLPROMPT'] = allp
    res = dict(case_ids=np.array(case_ids), y=np.array(ys), g=np.array(gs), n=np.array(ns), docs=docs, raw=raw,
               glob={p: d for p, d in glob.items() if p in ('tail5', 'open_first5', 'plast5', 'tail1')})
    if cache:
        pickle.dump(res, open(cp, 'wb'))
    return res


def folds_for(scheme, B, g, yb, seed=0):
    """Return a splitter-like list of (train,test). scheme: 'orig' | 'block'."""
    n = len(yb)
    if scheme == 'orig':
        groups = g
    elif scheme == 'block':
        groups = np.arange(n) // B
    else:
        raise ValueError(scheme)
    ng = len(np.unique(groups))
    cv = StratifiedGroupKFold(n_splits=min(5, ng), shuffle=True, random_state=seed)
    return list(cv.split(np.zeros(n), yb, groups))


def make_reader(name):
    if name == 'lr_word':
        return make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=50000), LogisticRegression(C=10, max_iter=2000))
    if name == 'lr_uni':
        return make_pipeline(TfidfVectorizer(ngram_range=(1, 1), min_df=2, sublinear_tf=True, max_features=20000), LogisticRegression(C=1, max_iter=2000))
    if name == 'nb':
        return make_pipeline(CountVectorizer(binary=True, min_df=2, max_features=30000), MultinomialNB(alpha=0.5))
    if name == 'lr_char':
        return make_pipeline(TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), min_df=2, sublinear_tf=True, max_features=50000), LogisticRegression(C=10, max_iter=2000))
    raise ValueError(name)


def cv_score(docs, y, folds, reader='lr_word'):
    """docs: list[str|None], y labels (any), folds. Returns AUROC (binary) or accuracy (multiclass)."""
    ok = np.array([d is not None and len(d) > 0 for d in docs])
    if ok.sum() < 50:
        return float('nan')
    X = np.array([d or '' for d in docs], dtype=object)
    classes = np.unique(y); binary = len(classes) == 2
    yy = (y == classes[-1]).astype(int) if binary else y
    sc = []
    for tr, te in folds:
        tr = tr[ok[tr]]; te = te[ok[te]]
        if len(te) == 0: continue
        m = make_reader(reader)
        try:
            m.fit(X[tr], yy[tr])
        except ValueError:
            sc.append(0.5 if binary else 1 / len(classes)); continue
        if binary:
            if len(np.unique(yy[te])) < 2: continue
            sc.append(roc_auc_score(yy[te], m.predict_proba(X[te])[:, 1]))
        else:
            sc.append(accuracy_score(yy[te], m.predict(X[te])))
    return float(np.mean(sc)) if sc else float('nan')


def strict_groups(kind, ids, g):
    if kind.startswith('opi'):
        return np.array([c.rsplit('_', 1)[1] for c in ids])
    return g


def cv_oof(docs, y, folds, reader='lr_word'):
    """Like cv_score but also returns out-of-fold probabilities (binary only)."""
    ok = np.array([d is not None and len(d) > 0 for d in docs])
    X = np.array([d or '' for d in docs], dtype=object)
    classes = np.unique(y); yy = (y == classes[-1]).astype(int)
    oof = np.full(len(y), np.nan); sc = []
    for tr, te in folds:
        tr = tr[ok[tr]]; te = te[ok[te]]
        m = make_reader(reader); m.fit(X[tr], yy[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
        if len(np.unique(yy[te])) == 2: sc.append(roc_auc_score(yy[te], oof[te]))
    return float(np.mean(sc)), oof


def within_subset_auc(oof, yb, sub, minper=15):
    w = []
    for s in np.unique(sub):
        m = (sub == s) & ~np.isnan(oof)
        if (yb[m] == 1).sum() >= minper and (yb[m] == 0).sum() >= minper:
            w.append(roc_auc_score(yb[m], oof[m]))
    return float(np.mean(w)) if w else float('nan')
