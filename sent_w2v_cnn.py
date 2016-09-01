#!/usr/bin/python
#-*- coding: utf-8 -*-
'''
date:2016/8/23
author:zhiqiangxu
description:构建CNN对说说进行情感分类
'''

import sys
reload(sys)
sys.setdefaultencoding('utf8')
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import Sequential, Model, model_from_json
from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Input, Merge, Convolution1D, MaxPooling1D
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping
from sent_tfidf_hash_classify import check_label
from preprocess_talk import load_talk_data,build_input_data

np.random.seed(2)
#模型超参数
sequence_length = 64    #序列最大长度
embedding_dim = 50     #词向量维度
filter_lengths = (3, 4) #卷积核的长度
num_filters = 150       #卷积核的数目
dropout_prob = (0.25, 0.5)  #Dropout层的比例
hidden_dims = 150       #池化层之后的全连接层大小

#训练参数
batch_size = 32
num_epochs = 10
val_split = 0.1        #验证集在训练集中的比例

#Word2Vec参数
min_word_count = 1     #词的最小个数
context = 10           #词上下文窗口大小

#构建用于情感分类的CNN模型的层级结构
def build_cnn_model(word2index, embedding_weights):

    print('Building Model ...')
    model = Sequential()
    model.add(Embedding(len(word2index)+1, embedding_dim, input_length=sequence_length, weights=[embedding_weights], trainable=True))
    model.add(Dropout(dropout_prob[0], input_shape=(sequence_length, embedding_dim)))
    graphIn = Input(shape=(sequence_length, embedding_dim))
    conv_layers = []
    for filter_len in filter_lengths:
        conv = Convolution1D(nb_filter=num_filters, filter_length=filter_len,
                             border_mode='valid', activation='relu',subsample_length=1)(graphIn)
        pool = MaxPooling1D(pool_length=2)(conv)
        flatten = Flatten()(pool)
        conv_layers.append(flatten)
    out = Merge(mode='concat')(conv_layers)
    graph = Model(input=graphIn, output=out)
    model.add(graph)
    model.add(Dense(hidden_dims))
    model.add(Dropout(dropout_prob[1]))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model

#在已标注正负情感训练集和测试集中训练、测试、并保存CNN模型
def train_cnn(neglabeledfile, poslabeledfile, possamples, negsamples):

    sentences, labels = load_talk_data(neglabeledfile, poslabeledfile, possamples, negsamples)
    x, y, wordToIndex, embedding_weights = build_input_data(sentences, labels, 'vocabulary.csv')
    idx_all = np.random.permutation(len(y))
    idx_train = idx_all[:int(0.8*len(y))]
    idx_test = idx_all[int(0.8*len(y)):]
    X_train, y_train = x[idx_train], y[idx_train]
    X_test, y_test = x[idx_test], y[idx_test]

    if os.path.isfile('sent_cnn.json') == False:
        cnn_model = build_cnn_model(wordToIndex, embedding_weights)
        print('Compiling ...')
        cnn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        print('Training ...')
        cnn_model.fit(X_train, y_train,
                      batch_size=batch_size,
                      nb_epoch=num_epochs,
                      validation_split=0.1,
                      callbacks=[EarlyStopping()],
                      verbose=2)
        json_string = cnn_model.to_json()
        with open('sent_cnn.json', 'w') as outFile:
            outFile.write(json_string)
        cnn_model.save_weights('cnn.h5')
    else:
        cnn_model = model_from_json(open('sent_cnn.json').read())
        cnn_model.load_weights('cnn.h5')

    print('Predicting ...')
    y_pred_test = cnn_model.predict_classes(X_test, batch_size=32)
    print('Test Accuracy: %.5f' % (np.sum(y_pred_test[:, 0] == y_test) / float(y_pred_test.shape[0])))
    print('Confusion Matrix: ')
    print(confusion_matrix(y_test, y_pred_test[:, 0], labels=[0, 1]))
    print(classification_report(y_test, y_pred_test[:, 0], digits=5))

#针对新的未标注说说数据用CNN模型做标注
def label_neg_talk(negunlabeledfile, outlabelfile, negsamples, vocabfile):

    vocabdict = {}
    for line in open(vocabfile, 'r'):
        word = line.split('\t')[0]
        index = int(line.split('\t')[1])
        vocabdict[word] = index

    talklist = []
    rawNegList = open(negunlabeledfile, 'r').readlines()
    wordToIndex = {}
    for i in range(0, negsamples):
        talkWords = rawNegList[i].split(',')[1].split(' ')
        wordlist = []
        for word in talkWords:
            if word.find('\n') != -1:
                word = word.replace('\n', '')
            wordlist.append(word)
        talklist.append(wordlist)

    for sentence in talklist:
        for word in sentence:
            if vocabdict.has_key(word) == False:
                wordToIndex[word] = 0
            else:
                wordToIndex[word] = vocabdict[word]

    x = np.array([[wordToIndex[word] for word in sentence] for sentence in talklist])
    x = sequence.pad_sequences(x, maxlen=sequence_length)
    lstm_model = model_from_json(open('sent_cnn.json').read())
    lstm_model.load_weights('cnn.h5')
    label = lstm_model.predict_classes(x, batch_size=32)
    outLabelFile = open(outlabelfile, 'w')
    for i in range(0, negsamples):
        outLabelFile.write(label[i]+','+rawNegList[i])
    outLabelFile.close()

if __name__ == '__main__':
    train_cnn('tokenized_neg_labeled_talk.csv', 'tokenized_pos_labeled_talk.csv', 2000, 40000)
    label_neg_talk('tokenized_neg_unlabeled_talk.csv', 'cnn_labeled_2000.txt', 2000, 'vocabulary.csv')