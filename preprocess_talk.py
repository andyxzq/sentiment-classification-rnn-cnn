#!/usr/bin/python
#-*- coding:utf-8 -*-
'''
date:2016/8/23
author:zhiqiangxu
description:对原始说说进行过滤、分词、过滤停用词、预训练词向量、模型输入数据构建的预处理
'''

import sys, time
reload(sys)
sys.setdefaultencoding('utf8')
import re,jieba
import numpy as np
from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary
from keras.preprocessing import sequence

sequence_length = 64  #最大序列长度
embedding_dim = 50    #词向量维度

#过滤掉说说数据中的所有非中文字符
def filter_talk(rawTalkFile, filteredTalkFile):
    filteredTalk = open(filteredTalkFile, 'w')
    i = 0
    for line in open(rawTalkFile, 'r'):
        commaIndex = line.find(',')
        txtinfo = line[commaIndex+1:]
        txtinfo = re.sub(r'[^\u4e00-\u9fa5]', '', txtinfo)
        if i < 10:
            print(txtinfo)
        i = i + 1
        if txtinfo != '':
            filteredTalk.write(txtinfo)
    filteredTalk.close()

#对过滤后的说说进行分词和词性标注
def tokenize_talk(filteredTalkFile, tokenizedTalkFile):
    tokenizedTalk = open(tokenizedTalkFile, 'w')
    for line in open(filteredTalkFile, 'r'):
        uin = line.split(',')[0]
        txtinfo = line.split(',')[1]
        wordlist = []
        for word in jieba.cut(txtinfo, cut_all=False):
            if word != '\r\n':
                wordlist.append(word)
        words = ' '.join(wordlist)
        tokenizedTalk.write(uin + ',' + words+'\n')
    tokenizedTalk.close()

#根据说说预料训练词向量并保存词库
def word2vec_train(tokenizedtalkfile, vocabularyfile):
    wordlist = []
    for line in open(tokenizedtalkfile, 'r'):
        talkwords = []
        for word in line.split(' '):
            if word.find('\n') != -1:
                word = word.replace('\n', '')
            talkwords.append(word)
        wordlist.append(talkwords)
    print('Start Training ...')
    start = time.time()
    model = Word2Vec(size=50, min_count=1, window=7, workers=4, sg=1, iter=5)
    model.build_vocab(wordlist)
    model.train(wordlist)
    model.save('corpus_word2vec_model.pkl')
    end = time.time()
    print('Training Time: %.5f' % (end-start))
    model = Word2Vec.load('corpus_word2vec_model.pkl')
    gensim_dict = Dictionary()
    gensim_dict.doc2bow(model.vocab.keys(), allow_update=True)
    word2index = {v : k for k, v in gensim_dict.items()}
    with open(vocabularyfile, 'w') as vocabFile:
        for item in word2index.keys():
            vocabFile.write(item+'\t'+str(word2index[item])+'\n')

#对分词后的说说进行停用词过滤
def filter_stopwords(talkfile, filteredtalkfile, stopwordsfile):
    stopWordsFile = open(stopwordsfile, 'r')
    stopwords = set()
    for line in stopWordsFile:
        line = line.replace('\r\n', '')
        stopwords.add(line.encode('utf-8'))
    stopWordsFile.close()
    filteredTalkFile = open(filteredtalkfile, 'w')
    filteredTalkFile.write('label,uin,txtinfo\n')
    with open(talkfile, 'r') as talkFile:
        talklist = talkFile.readlines()
        for i in range(1, len(talklist)):
            label = talklist[i].split(',')[0]
            uin = talklist[i].split(',')[1]
            txtinfo = talklist[i].split(',')[2]
            print(txtinfo)
            wordlist = []
            words = txtinfo.split(' ')[:-1]
            for word in words:
                if word not in stopwords:
                    wordlist.append(word)
            if len(wordlist) != 0:
                txt = ' '.join(wordlist)
                filteredTalkFile.write(str(label)+','+str(uin)+','+txt+'\n')
    filteredTalkFile.close()

#读取分词后的文本文件至列表中
def load_talk_data(tokenizednegfile, tokenizedposfile, negsamples, possamples):
    tokenizedNegFile = open(tokenizednegfile, 'r')
    negtalklist = tokenizedNegFile.readlines()
    talklist = []
    for i in range(1, possamples + 1):
        txtinfo = negtalklist[i].split(',')[2]
        wordlist = txtinfo.split(' ')
        for j in range(len(wordlist)):
            if wordlist[j].find('\n') != -1:
                wordlist[j] = wordlist[j].replace('\n', '')
            wordlist[j] = wordlist[j].decode('utf-8')
        talklist.append(wordlist)
    tokenizedPosFile = open(tokenizedposfile, 'r')
    postalklist = tokenizedPosFile.readlines()
    for i in range(1, negsamples + 1):
        txtinfo = postalklist[i].split(',')[2]
        wordlist = txtinfo.split(' ')
        for j in range(len(wordlist)):
            if wordlist[j].find('\n') != -1:
                wordlist[j] = wordlist[j].replace('\n', '')
            wordlist[j] = wordlist[j].decode('utf-8')
        talklist.append(wordlist)
    labels = [1] * possamples + [0] * negsamples
    return (talklist, labels)

#根据正负样本说说构建词典、词索引矩阵、词向量矩阵
def build_input_data(sentences, labels, vocabfile):
    vocabdict = {}
    for line in open(vocabfile, 'r'):
        word = line.split('\t')[0]
        index = int(line.split('\t')[1])
        vocabdict[word] = index
    wordToIndex = {}
    print('Initializing input ...')
    for sentence in sentences:
        for word in sentence:
            if vocabdict.has_key(word) == False:
                wordToIndex[word] = 0
            else:
                wordToIndex[word] = vocabdict[word]
    print('Vocabulary Size: %d'%len(vocabdict.keys()))
    x = np.array([[wordToIndex[word] for word in sentence] for sentence in sentences])
    x = sequence.pad_sequences(x, maxlen=sequence_length)
    y = np.array(labels)
    print('Initializing word embedding matrix ...')
    embedding_model = Word2Vec.load('corpus_word2vec_model.pkl')
    embedding_weights = np.zeros((len(vocabdict.keys()) + 1, embedding_dim))
    for word, i in vocabdict.items():
        embedding_weights[i] = embedding_model[word]
    print('Embedding Matrix Rows: %d' % embedding_weights.shape[0])
    return (x, y, vocabdict, embedding_weights)

if __name__ == '__main__':
    filter_talk('word2vec_corpus.csv', 'filtered_word2vec_corpus.csv')
    tokenize_talk('filtered_neg_unlabeled_talk.csv', 'tokenized_neg_unlabeled_talk.csv')
    word2vec_train('tokenized_word2vec_corpus.csv', 'vocabulary.csv')
    #filter_stopwords('tokenized_filtered_neg_labeled_txt.csv', 'tokenized_neg_labeled_txt.csv', 'stopwords.txt')

