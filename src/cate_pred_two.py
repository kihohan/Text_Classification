import os 
import re
import pickle
import numpy as np
import pandas as pd

from function import *

import sentencepiece as spm
from tensorflow import keras
from tensorflow.keras.preprocessing import sequence

path = os.getcwd()

# spm_load
sp = spm.SentencePieceProcessor()
sp.Load(path + '/out/cate_spm.model')
# tkn_load
with open(path + '/out/cate_tkn.pickle', 'rb') as handle:
    tkn = pickle.load(handle)
# label load
with open(path + '/out/labels.pickle', 'rb') as handle:
    mapping_dct = pickle.load(handle)
# model_load
classification_model = keras.models.load_model(path + '/out/cate_model.h5')

def cate_pred_two(lst, max_len):
    pre = [' '.join(clean_spm(sp.encode_as_pieces(text))) for text in lst]
    t = sequence.pad_sequences(tkn.texts_to_sequences(pre), maxlen = max_len)
    P = classification_model.predict_on_batch(t)
    pred = np.argsort(P)[0][::-1][:2]
    prob = np.partition(P.flatten(), -2)[::-1][:2]
    X = pd.Series(pred).map(mapping_dct).to_list()
    return dict(zip(X, prob))

if __name__ == '__main__':
    lst = ['삼성전자TV 32인치TV 43인치TV 49인치TV Full HD 삼성TV 소형TV 티비'] * 1
    cate_pred(lst, 50)
