import polars as pl, glob, os, sys
rows=[]
for f in sorted(glob.glob('results/bridge/all_*_*.parquet')):
    name=os.path.basename(f)[4:-8]; kind,model=name.rsplit('_',1)
    if kind=='hand': continue
    cols=['case_id','tok_idx','region','on_task']
    df=pl.read_parquet(f,columns=cols).sort('case_id','tok_idx')
    # final assistant start: first tok of last 'assistant' run; boundary: template run right before it (or trailing template for opi)
    df=df.with_columns(n=pl.len().over('case_id'))
    last_content_in=df.filter(pl.col('region').is_in(['system','user','assistant_prior'])).group_by('case_id').agg(lc=pl.col('tok_idx').max(), fc=pl.col('tok_idx').min())
    first_out=df.filter(pl.col('region')=='assistant').group_by('case_id').agg(fo=pl.col('tok_idx').min(), lo=pl.col('tok_idx').max())
    df=df.join(last_content_in,on='case_id',how='left').join(first_out,on='case_id',how='left')
    t=pl.col('tok_idx')
    fo=pl.col('fo').fill_null(pl.col('n'))
    part=(pl.when((pl.col('region')=='template')&(t<pl.col('fc'))).then(pl.lit('1 tmpl_open'))
      .when((t<=pl.col('lc'))&(pl.col('region')!='template')&((t-pl.col('fc'))<(pl.col('lc')-pl.col('fc')+1)/2)).then(pl.lit('2 prompt_first_half'))
      .when((t<=pl.col('lc'))&(pl.col('region')!='template')).then(pl.lit('3 prompt_second_half'))
      .when((t<=pl.col('lc'))).then(pl.lit('2b tmpl_inside_prompt'))
      .when((t>pl.col('lc'))&(t<fo)).then(pl.lit('4 tmpl_boundary'))
      .when((pl.col('region')=='assistant')&((t-pl.col('fo'))<(pl.col('lo')-pl.col('fo')+1)/2)).then(pl.lit('5 resp_first_half'))
      .when(pl.col('region')=='assistant').then(pl.lit('6 resp_second_half'))
      .otherwise(pl.lit('7 trailer')))
    df=df.with_columns(part=part)
    # case-macro on-task rate per part, and share of cases with >=1 on-task in the part
    g=df.group_by('case_id','part').agg(r=pl.col('on_task').mean(), a=pl.col('on_task').max(), k=pl.len())
    s=g.group_by('part').agg(rate=pl.col('r').mean(), any=pl.col('a').mean(), ntok=pl.col('k').mean()).with_columns(kind=pl.lit(kind),model=pl.lit(model))
    rows.append(s)
out=pl.concat(rows)
pl.Config.set_tbl_rows(400); pl.Config.set_tbl_cols(30); pl.Config.set_tbl_width_chars(250)
for metric in ['rate','any','ntok']:
    print('==',metric)
    print(out.with_columns(km=pl.col('kind')+'_'+pl.col('model')).pivot(on='part',index='km',values=metric,sort_columns=True).sort('km').with_columns(pl.selectors.float().round(3)))
