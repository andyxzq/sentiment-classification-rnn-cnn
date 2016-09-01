#!/usr/bin/python
#-*- coding: utf-8 -*-
'''
date:2016/8/23
author:zhiqiangxu
description:用原始的tf-idf特征+hash trick进行情感分类
'''

import sys
reload(sys)
sys.setdefaultencoding('utf8')
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction import FeatureHasher
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC, LinearSVC
from sklearn.externals import joblib

np.random.seed(1)

#对原始词特征做hash trick变换
def hash_feature_extract(tokenizednegfile, tokenizedposfile, negsamples, possamples, n_features):
    tokenizedNegFile = open(tokenizednegfile, 'r')
    talkList = tokenizedNegFile.readlines()
    featureList = []
    for i in range(1, possamples+1):
        txtinfo = talkList[i].split(',')[2]
        wordlist = txtinfo.split(' ')
        featureList.append(wordlist)
    tokenizedPosFile = open(tokenizedposfile, 'r')
    talkList = tokenizedPosFile.readlines()
    talkList = talkList[1:]
    np.random.shuffle(talkList)      #随机打乱负样本
    for i in range(0, negsamples):
        txtinfo = talkList[i].split(',')[2]
        wordlist = txtinfo.split(' ')
        featureList.append(wordlist)
    hasher = FeatureHasher(n_features=n_features, dtype=np.float32, input_type='string')
    hashFeature = hasher.transform(featureList)
    return hashFeature

#计算分词后的说说中词的索引及词频
def word_index_and_frequency(tokenizednegfile, tokenizedposfile, negsamples):
    wordIndex = {}
    wordFrequency = {}
    tokenizedNegFile = open(tokenizednegfile, 'r')
    talkList = tokenizedNegFile.readlines()
    index = 1
    for i in range(1, 1001):
        txtinfo = talkList[i].split(',')[2]
        wordlist = txtinfo.split(' ')
        for word in wordlist:
            if word not in wordIndex.keys():
                wordIndex[word] = index
                wordFrequency[word] = 1
                index = index + 1
            else:
                wordFrequency[word] = wordFrequency[word] + 1
    tokenizedPosFile = open(tokenizedposfile, 'r')
    talkList = tokenizedPosFile.readlines()
    for i in range(1, negsamples+1):
        txtinfo = talkList[i].split(',')[2]
        wordlist = txtinfo.split(' ')
        for word in wordlist:
            if word not in wordIndex.keys():
                wordIndex[word] = index
                wordFrequency[word] = 1
                index = index + 1
            else:
                wordFrequency[word] = wordFrequency[word] + 1
    tokenizedNegFile.close()
    tokenizedPosFile.close()
    return (wordIndex, wordFrequency)

#计算分词后的说说的词频矩阵，并转换为tf-idf特征矩阵
def tfidf_feature_extract(tokenizednegfile, tokenizedposfile):
    wordIndex, wordFrequency = word_index_and_frequency(tokenizednegfile, tokenizedposfile)
    countMatrix = np.zeros((11000, cols), dtype=np.int32)
    tokenizedNegFile = open(tokenizednegfile, 'r')
    talkList = tokenizedNegFile.readlines()
    for i in range(1, 1001):
        txtinfo = talkList[i].split(',')[2]
        wordlist = txtinfo.split(' ')
        wordlist.remove('\n')
        for word in wordlist:
            colIndex = wordIndex[word] - 1
            countMatrix[i-1, colIndex] = wordFrequency[word]
    tokenizedPosFile = open(tokenizedposfile, 'r')
    talkList = tokenizedPosFile.readlines()
    for i in range(1, 10001):
        txtinfo = talkList[i].split(',')[2]
        wordlist = txtinfo.split(' ')
        wordlist.remove('\n')
        for word in wordlist:
            colIndex = wordIndex[word] - 1
            countMatrix[1000+i - 1, colIndex] = wordFrequency[word]
    csrCountMatrix = csr_matrix(countMatrix)
    del(countMatrix)
    tfidf_feature = TfidfTransformer().fit_transform(csrCountMatrix)
    return tfidf_feature

#对已标记的正负样本进行训练和测试
def talk_classify(tokenizednegfile, tokenizedposfile, negsamples, possamples):
    hash_feature = hash_feature_extract(tokenizednegfile, tokenizedposfile, negsamples, possamples, 5000)
    target = np.array([1]*possamples + [0]*negsamples)
    X_train, X_test, y_train, y_test = train_test_split(hash_feature, target, test_size=0.2, random_state=1)
    X_train = StandardScaler().fit_transform(X_train.toarray())
    X_test = StandardScaler().fit_transform(X_test.toarray())
    clf = LinearSVC(class_weight='balanced', random_state=1)
    clf.fit(X_train, y_train)
    joblib.dump(clf, 'linearsvm.pkl')
    y_pred_train = clf.predict(X_train)
    print('Train Accuracy: %.5f' % (np.sum(y_train == y_pred_train)/len(y_train)))
    print(classification_report(y_train, y_pred_train, digits=5))
    y_pred_test = clf.predict(X_test)
    print('Test Accuracy: %.5f' % (np.sum(y_pred_test == y_test)/len(y_test)))
    print('Confusion Matrix: ')
    print(confusion_matrix(y_test, y_pred_test, labels=[0, 1]))
    print(classification_report(y_test, y_pred_test, digits=5))

#对新的说说进行标注
def label_neg_txt(tokenizedrawnegfile, outlabelfile, negsamples, n_features):
    rawNegFile = open(tokenizedrawnegfile, 'r')
    rawNegList = rawNegFile.readlines()
    unlabeledNegList = []
    for i in range(1, negsamples+1):
        txtinfo = rawNegList[i].split(',')[1]
        unlabeledNegList.append(txtinfo.split(' '))
    hasher = FeatureHasher(n_features=n_features, dtype=np.float32, input_type='string')
    hashed_feature = hasher.transform(unlabeledNegList)
    hashed_feature = StandardScaler().fit_transform(hashed_feature.toarray())
    lr = joblib.load('linearsvm.pkl')
    y_pred = lr.predict(hashed_feature)
    outLabelFile = open(outlabelfile, 'w')
    for i in range(0, negsamples):
        outLabelFile.write(y_pred[i] + ',' + rawNegList[i])
    outLabelFile.close()

if __name__ == '__main__':
    talk_classify('tokenized_neg_labeled_txt.csv', 'tokenized_filtered_pos_labeled_txt_1.csv', 20000, 2000)
    label_neg_txt('tokenized_neg_unlabeled_talk.csv', 'out_pred_label.txt', 11000, 5000)