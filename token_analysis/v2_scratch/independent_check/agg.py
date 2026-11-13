import polars as pl, glob, sys
pl.Config.set_tbl_rows(200); pl.Config.set_tbl_cols(30); pl.Config.set_tbl_width_chars(250)
d=pl.concat([pl.read_csv(f,separator='\t',null_values=['nan'],schema_overrides={'within':pl.Float64,'score':pl.Float64}) for f in glob.glob('main_*.tsv')])
scheme=sys.argv[1]; reader=sys.argv[2]; src=sys.argv[3]; parts=sys.argv[4].split(',')
x=d.filter((pl.col('scheme')==scheme)&(pl.col('reader')==reader)&(pl.col('src')==src)&pl.col('part').is_in(parts))
w=x.pivot(on='part',index=['kind','model'],values='score').sort('kind','model')
print(w.select(['kind','model']+[p for p in parts if p in w.columns]).with_columns(pl.col(pl.Float64).round(3)))
