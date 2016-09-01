#!/usr/bin/python
#-*- coding: utf-8 -*-
'''
date:2016/8/23
author:zhiqiangxu
description:用svm对词向量累加特征的说说进行情感分类
'''

import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC, LinearSVC
from gensim.models.word2vec import Word2Vec

vocab_dim = 50 #词向量维度

#把说说中每个词的词向量累加构成的向量作为句子特征
def we_feature_extract(tokenizednegfile, tokenizedposfile, negsamples, possamples, vocabfile):

    tokenizedNegFile = open(tokenizednegfile, 'r')
    negtalklist = tokenizedNegFile.readlines()
    talklist = []
    for i in range(1, possamples+1):
        txtinfo = negtalklist[i].split(',')[2]
        wordlist = txtinfo.split(' ')
        talklist.append(wordlist)
    tokenizedPosFile = open(tokenizedposfile, 'r')
    postalklist = tokenizedPosFile.readlines()
    for i in range(1, negsamples + 1):
        txtinfo = postalklist[i].split(',')[2]
        wordlist = txtinfo.split(' ')
        talklist.append(wordlist)

    embedding_model = Word2Vec.load('corpus_word2vec_model.pkl')
    vocabdict = {}
    for line in open(vocabfile, 'r'):
        word = line.split('\t')[0]
        index = int(line.split('\t')[1])
        vocabdict[word] = index

    X = np.zeros((negsamples+possamples, vocab_dim), dtype=np.float32)
    for i in range(possamples+negsamples):
        for item in talklist[i]:
            if vocabdict.has_key(item):
                X[i] = X[i] + embedding_model[item]
    y = np.array([1]*possamples + [0]*negsamples)
    return (X, y)

#根据句子的特征训练分类器并做预测
def classify(tokenizednegfile, tokenizedposfile, negsamples, possamples):

    X, y = we_feature_extract(tokenizednegfile, tokenizedposfile, negsamples, possamples, 'vocabulary.csv')
    idx_all = np.random.permutation(len(y))
    idx_train = idx_all[:int(0.8 * len(y))]
    idx_test = idx_all[int(0.8 * len(y)):]
    X_train, y_train = X[idx_train], y[idx_train]
    X_test, y_test = X[idx_test], y[idx_test]
    X_train = StandardScaler().fit_transform(X_train)
    X_test = StandardScaler().fit_transform(X_test)

    clf = SVC(class_weight='balanced', random_state=1, C=100)
    clf.fit(X_train, y_train)
    y_pred_train = clf.predict(X_train)
    print('Train Accuracy: %.5f' % (np.sum(y_train == y_pred_train) / len(y_train)))
    print(classification_report(y_train, y_pred_train, digits=5))
    y_pred_test = clf.predict(X_test)
    print('Test Accuracy: %.5f' % (np.sum(y_pred_test == y_test) / len(y_test)))
    print('Confusion Matrix: ')
    print(confusion_matrix(y_test, y_pred_test, labels=[0, 1]))
    print(classification_report(y_test, y_pred_test, digits=5))

if __name__ == '__main__':
    classify('tokenized_neg_labeled_talk.csv', 'tokenized_pos_labeled_talk.csv', 40000, 2000)