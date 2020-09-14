import pandas as pd
df = pd.read_pickle('df_beauty.pk')
df[['mall_goods_name']].to_csv(r'spm_text.txt', header = None, index = None, sep = '\t')
