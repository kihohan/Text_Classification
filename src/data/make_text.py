import numpy as np
import pandas as pd

df = pd.read_pickle('brand_dw.pk')

df[['mall_goods_name']].to_csv(r'spm_text.txt', header = None, index = None, sep = '\t')
np.mean(df['mall_goods_name'].apply(lambda x:len(x)))
