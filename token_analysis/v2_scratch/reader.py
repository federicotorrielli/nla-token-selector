"""Per-part reader: TF-IDF + logistic regression on k explanations from one part, scored against transcript labels."""
import sys, glob, os, re, numpy as np, polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, accuracy_score
K=5; rng=np.random.default_rng(0)
ROOT='/mnt/DATA/Nimbus/Projects/Research/nla-token-selector/results/bridge'

def labels(kind, cid):
    if kind.startswith('opi'):
        t,i,rest=cid.split('__'); return i, f'{t}_{rest.rsplit("_",1)[1]}'
    if kind=='liars': return None, cid
    if kind=='tt': return int(cid.endswith('_attack')), cid.split('_')[0]
    if kind=='misalign': a,p,s=cid.split('__'); return int(a!='base'), p
    if kind.startswith('taboo'): w,p,s=cid.split('__'); return w, p

def parts(region, n):
    idx=np.arange(n); tm=region=='template'; content=np.isin(region,['system','user','assistant_prior']); rep=region=='assistant'
    out={}
    ci=idx[content]; ri=idx[rep]
    if len(ci)==0: return out
    fc,lc=ci[0],ci[-1]
    fo=ri[0] if len(ri) else n
    pc=ci  # prompt content tokens
    op=idx[:fc]; out['tmpl_open']=op[-K:]
    out['prompt_first']=pc[:K]; out['prompt_last']=pc[-K:]
    mid=pc[K:-K]; out['prompt_rand']=np.sort(rng.choice(mid,K,replace=False)) if len(mid)>=K else pc[:0]
    out['boundary']=idx[lc+1:fo][-K:]
    if len(ri):
        out['reply_first']=ri[:K]; out['reply_last']=ri[-K:]
        m=ri[K:-K]; out['reply_rand']=np.sort(rng.choice(m,K,replace=False)) if len(m)>=K else ri[:0]
    end=(ri[-1]+1) if len(ri) else n
    out['even']=np.unique(np.linspace(0,end-1,K).round().astype(int))
    out['random']=np.sort(rng.choice(end,min(K,end),replace=False))
    return out

def run(f):
    name=os.path.basename(f)[4:-8]; kind,model=name.rsplit('_',1)
    df=pl.read_parquet(f,columns=['case_id','tok_idx','region','label','on_task','explanation','token']).sort('case_id','tok_idx')
    docs={}; judge={}; text={}; y=[]; g=[]; sub=[]
    for (cid,),d in df.group_by('case_id',maintain_order=True):
        lab,grp=labels(kind,cid)
        if kind=='liars': lab=int(d['label'].max())
        reg=d['region'].to_numpy(); ex=d['explanation'].to_list(); ot=d['on_task'].to_numpy(); tk=d['token'].to_list(); sub.append(cid.rsplit('_',1)[0] if kind=='liars' else '')
        ps=parts(reg,len(reg)); y.append(lab); g.append(grp)
        for p,ix in ps.items():
            docs.setdefault(p,[]).append(' \n '.join((ex[i] or '') for i in ix) if len(ix) else None)
            judge.setdefault(p,[]).append(float(np.mean(ot[ix])) if len(ix) else np.nan)
            text.setdefault(p,[]).append(' '.join(tk[i] for i in ix) if len(ix) else None)
        cont=np.isin(reg,['system','user','assistant_prior']); rp=reg=='assistant'
        for p,msk in (('ALLTEXT_prompt',cont),('ALLTEXT_prompt+reply',cont|rp)):
            docs.setdefault(p,[]).append(None); judge.setdefault(p,[]).append(np.nan)
            text.setdefault(p,[]).append(' '.join(t for t,m in zip(tk,msk) if m))
        for p in docs:
            if len(docs[p])<len(y): docs[p].append(None); judge[p].append(np.nan); text[p].append(None)
    y=np.array(y); g=np.array(g); classes=np.unique(y); binary=len(classes)==2
    yb=(y==classes[-1]).astype(int) if binary else y
    ng=len(np.unique(g)); sub=np.array(sub)
    def cvscore(D,ok,char=False):
        X=np.array([d or '' for d in D])[ok]; yy=yb[ok]; gg=g[ok]; ss=sub[ok]
        chance=0.5 if binary else 1/len(classes)
        if len(np.unique(X))<2: return chance, float('nan')
        cv=StratifiedGroupKFold(n_splits=min(5,len(np.unique(gg))),shuffle=True,random_state=0)
        sc=[]; pred=np.full(len(yy),np.nan)
        for tr,te in cv.split(X,yy,gg):
            vec=TfidfVectorizer(analyzer='char_wb',ngram_range=(2,5),min_df=2,sublinear_tf=True,max_features=50000) if char else TfidfVectorizer(ngram_range=(1,2),min_df=2,sublinear_tf=True,max_features=50000)
            m=make_pipeline(vec,LogisticRegression(C=10,max_iter=2000))
            try: m.fit(X[tr],yy[tr])
            except ValueError: sc.append(chance); continue
            if binary:
                p=m.predict_proba(X[te])[:,1]; pred[te]=p
                if len(np.unique(yy[te]))==2: sc.append(roc_auc_score(yy[te],p))
            else: sc.append(accuracy_score(yy[te],m.predict(X[te])))
        within=float('nan')
        if binary and kind=='liars':
            w=[roc_auc_score(yy[ss==u],pred[ss==u]) for u in np.unique(ss) if len(np.unique(yy[ss==u]))==2 and (ss==u).sum()>=50 and not np.isnan(pred[ss==u]).any()]
            within=float(np.mean(w)) if w else float('nan')
        return float(np.mean(sc)), within
    for p in docs:
        okd=np.array([d is not None and len(d)>0 for d in docs[p]]); okt=np.array([d is not None and len(d)>0 for d in text[p]])
        r,rw=cvscore(docs[p],okd) if okd.sum()>=50 else (float('nan'),float('nan'))
        t,tw=cvscore(text[p],okt,char=True) if okt.sum()>=50 else (float('nan'),float('nan'))
        j=np.array(judge[p]); okj=~np.isnan(j)
        ja=roc_auc_score(yb[okj],j[okj]) if binary and okj.sum()>=50 and len(np.unique(yb[okj]))==2 else float('nan')
        print(f'{kind}\t{model}\t{p}\t{r:.3f}\t{t:.3f}\t{ja:.3f}\t{rw:.3f}\t{tw:.3f}\t{max(okd.sum(),okt.sum())}',flush=True)

if __name__=='__main__':
    for f in sys.argv[1:]: run(f)
