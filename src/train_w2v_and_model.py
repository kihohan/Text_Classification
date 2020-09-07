import os
import re
import pickle
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

from function import *

import sentencepiece as spm

from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from gensim.models import Word2Vec

from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Activation, Dense, Embedding, Flatten, Dropout, Conv2D, Reshape, GlobalMaxPooling2D
# main fucntion
def modeling(df_pickle, embedding_dim, max_len):
    # load spm model 
    path = os.getcwd()
    spm_path = path + '/out/cate_spm.model'
    sp = spm.SentencePieceProcessor()
    sp.Load(spm_path)
    '''
    data  sample
    ===================================================================================
    mall_goods_name	master_tag
    신원 자외선 칫솔살균기 SW-15A 노랑	구강가전
    NS홈쇼핑 삼성전자 MC32K7056CT 세라믹조리실 오븐 32L 쇼핑도 건강하게 ...	주방가전
    교세라 정품 TK-5154KY P6035cdn 10K 노랑	사무가전(프린터/복합기)
    캐슬 Avon2 에이본2 리본트위터 북쉘프스피커	스피커
    ===================================================================================
    '''
    df = pd.read_pickle(df_pickle)
    classes = df['master_tag'].nunique()
    # pre processing -> if you do experiment, should be use mp
    df['mall_goods_name'] = df['mall_goods_name'].apply(lambda x:' '.join(clean_spm(sp.encode_as_pieces(x))))
    # generate word2vec embedding layer
    sentences = df['mall_goods_name'].drop_duplicates().apply(lambda x:x.split(' ')).to_list()

    # embedding_dim = 600
    model = Word2Vec(sentences, size = embedding_dim, window = 5, min_count = 2, workers = 8)

    word_vectors = model.wv
    vocabs = word_vectors.vocab.keys()
    word_vectors_list = [word_vectors[v] for v in vocabs]
    print ('Vocab Size:',len(model.wv.vocab))

    filename = path + '/out/cate_w2v.txt'
    model.wv.save_word2vec_format(filename, binary = False)
    # load embedding layer
    embedding_index = {}
    f = open(os.path.join('',filename), encoding = 'utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:])
        embedding_index[word] = coefs
    f.close()
    # train model
    X = df[['mall_goods_name']]
    y = df['master_tag']

    label_encoder = preprocessing.LabelEncoder()
    y = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 2020)
    X_train, X_test = X_train['mall_goods_name'], X_test['mall_goods_name']

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    # max_len = 90 -> example
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train)

    sequences = tokenizer.texts_to_sequences(X_train)
    X_train = sequence.pad_sequences(sequences,maxlen = max_len) #  padding='post'
    sequences = tokenizer.texts_to_sequences(X_test)
    X_test = sequence.pad_sequences(sequences, maxlen = max_len)

    # embedding_dim = 600 -> example
    word_index = tokenizer.word_index

    num_words = len(word_index) + 1
    embedding_matrix = np.zeros((num_words, embedding_dim))
    for word, i in word_index.items():
        if i > num_words:
            continue
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    print ('num_words:',num_words)

    model = Sequential()
    embedding_layer = Embedding(num_words,
                                embedding_dim,
                                weights = [embedding_matrix],
                                input_length = max_len)
    model.add(embedding_layer)
    model.add(Reshape((max_len, embedding_dim, 1), input_shape = (max_len, embedding_dim)))
    model.add(Conv2D(filters = 32, kernel_size = (4, embedding_dim), strides = (2,2), padding = 'valid'))
    model.add(GlobalMaxPooling2D())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['acc'])
    history = model.fit(x = X_train, y = y_train, batch_size = 128, epochs = 1, verbose = 1, validation_split = 0.1)
    # evaluate
    acc = model.evaluate(X_test,y_test)
    print('Loss: {:0.3f} | Accuracy: {:0.3f}'.format(acc[0],acc[1]))
    print ('=' * 50)
    pred = model.predict(X_test)
    pred_bool = np.argmax(pred,1)
    y_test_bool = np.argmax(y_test,1)
    print(classification_report(y_test_bool, pred_bool))
    # classification_report = pd.DataFrame(classification_report(y_test_bool, pred_bool)).transpose()
    # save labels
    _class = label_encoder.classes_
    _num = [x for x in range(len(_class))]
    mapping_dct = dict(zip(_num,_class))
    
    # save tkn_model
    with open(path + '/out/cate_tkn.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    # save classification_model
    model.save(path + '/out/cate_model.h5')
    
    # save labels_dictionary
    _class = label_encoder.classes_
    _num = [x for x in range(len(_class))]
    mapping_dct = dict(zip(_num,_class))
    with open(path + '/out/labels.pickle', 'wb') as handle:
        pickle.dump(mapping_dct, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    path = os.getcwd()
    df_pickle = path + '/data/ele.pk'
    embedding_dim = 100
    max_len = 50
    modeling(df_pickle, embedding_dim, max_len)
