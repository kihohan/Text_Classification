def cate_pred_two(lst, max_len):
    pre = [' '.join(clean_spm(sp.encode_as_pieces(text))) for text in lst]
    t = sequence.pad_sequences(tkn.texts_to_sequences(pre), maxlen = max_len)
    P = classification_model.predict_on_batch(t)
    pred = np.argsort(P)[0][::-1][:2]
    prob = np.partition(P.flatten(), -2)[::-1][:2]
    X = pd.Series(pred).map(mapping_dct).to_list()
    return dict(zip(X, prob))
