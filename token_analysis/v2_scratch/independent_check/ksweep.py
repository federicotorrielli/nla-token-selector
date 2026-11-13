import polars as pl, glob, sys
pl.Config.set_tbl_rows(200); pl.Config.set_tbl_cols(40); pl.Config.set_tbl_width_chars(250)
d=pl.concat([pl.read_csv(f,separator='\t',null_values=['nan'],schema_overrides={'within':pl.Float64,'score':pl.Float64}) for f in glob.glob('main_*.tsv')])
scheme=sys.argv[1]; fam=sys.argv[2].split(',')
x=d.filter((pl.col('scheme')==scheme)&(pl.col('reader')=='lr_word')&(pl.col('src')=='docs'))
x=x.with_columns(pl.col('part').str.extract(r'^([a-z_]+?)(\d+)$',1).alias('fam'),pl.col('part').str.extract(r'(\d+)$',1).cast(pl.Int32).alias('k')).filter(pl.col('fam').is_in(fam))
print(x.pivot(on='k',index=['kind','model','fam'],values='score').sort('kind','model','fam').with_columns(pl.col(pl.Float64).round(3)))
