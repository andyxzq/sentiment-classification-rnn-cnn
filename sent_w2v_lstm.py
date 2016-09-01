#!/usr/bin/python
#-*- coding: utf-8 -*-
'''
date:2016/8/23
author:zhiqiangxu
description:构建LSTM网络对说说进行情感分类
'''

import sys, os
reload(sys)
sys.setdefaultencoding('utf-8')
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout, Activation
from keras.models import model_from_json
from keras.callbacks import EarlyStopping

from preprocess_talk import build_input_data, load_talk_data
np.random.seed(1)

#设置参数
vocab_dim = 50     #词向量维度
maxlen = 64        #序列最大长度
window_size = 7    #词向量上下文窗口
batch_size = 32    #批大小
n_epoch = 10       #迭代轮数
input_length = 64  #最大输入序列长度

#根据当前需要训练和测试中的所有说说的词来训练词向量
def word2vec_train(talklist):
    print('Trainging Word2Vec ...')
    model = Word2Vec(size=vocab_dim, min_count=3, window=window_size, workers=4, iter=5)
    model.build_vocab(talklist)
    model.train(talklist)
    model.save('word2vec_model.pkl')
    gensim_dict = Dictionary()
    gensim_dict.doc2bow(model.vocab.keys(), allow_update=True)
    #记录词的索引
    word2index = {v:k+1 for k, v in gensim_dict.items()}
    # 生成词向量列表
    word2vec = {word:model[word] for word in word2index.keys()}
    #转换词的列表成为词索引列表
    data = []
    for sentence in talklist:
        word_ids = []
        for word in sentence:
            try:
                word_ids.append(word2index[word])
            except:
                word_ids.append(0)       #频数小于min_count的词的索引为0
        data.append(word_ids)
    talklist = sequence.pad_sequences(data, maxlen=maxlen)
    return word2index, word2vec, talklist

#构建LSTM神经网络，并进行在训练集和测试集上训练和预测
def train_lstm(n_tokens, embedding_weights, X_train, y_train, X_test, y_test):

    if os.path.isfile('sent_lstm.json') == False:
        model = Sequential()
        print(embedding_weights.shape)
        model.add(Embedding(input_dim=n_tokens+1, output_dim=vocab_dim, mask_zero=True,
                            weights=[embedding_weights], input_length=input_length))
        model.add(LSTM(output_dim=50, activation='sigmoid', inner_activation='hard_sigmoid'))
        model.add(Dropout(0.5))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        print('Compiling the Model ...')
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        print('Training ...')
        model.fit(X_train, y_train, batch_size=batch_size,
                                    nb_epoch=n_epoch,
                                    validation_split=0.1,
                                    callbacks=[EarlyStopping()],
                                    verbose=2)
        json_string = model.to_json()
        with open('sent_lstm.json', 'w') as outFile:
            outFile.write(json_string)
        model.save_weights('lstm.h5', overwrite=True)
    else:
        model = model_from_json(open('sent_lstm.json').read())
        model.load_weights('lstm.h5')

    print('Predicting ...')
    y_pred_test = model.predict_classes(X_test, batch_size=32)
    print('Test Accuracy: %.5f' % (np.sum(y_pred_test[:, 0] == y_test) / float(y_pred_test.shape[0])))
    print('Confusion Matrix: ')
    print(confusion_matrix(y_test, y_pred_test[:, 0], labels=[0, 1]))
    print(classification_report(y_test, y_pred_test[:, 0], digits=5))

#读取输入数据文件,构建LSTM需要的输入,调用神经网络进行训练
def load_data_and_train(tokenizednegfile, tokenizedposfile, negsamples, possamples):
    talklist, labels = load_talk_data(tokenizednegfile, tokenizedposfile, negsamples, possamples)
    x, y, wordToIndex, embedding_weights = build_input_data(talklist, labels, 'vocabulary.csv')
    idx_all = np.random.permutation(len(y))
    idx_train = idx_all[:int(0.8 * len(y))]
    idx_test = idx_all[int(0.8 * len(y)):]
    X_train, y_train = x[idx_train], y[idx_train]
    X_test, y_test = x[idx_test], y[idx_test]
    train_lstm(len(wordToIndex.keys()), embedding_weights, X_train, y_train, X_test, y_test)

if __name__ == '__main__':
    load_data_and_train('tokenized_neg_labeled_talk.csv', 'tokenized_pos_labeled_talk.csv', 40000, 2000)