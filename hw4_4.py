"""
Author： YuYue Yang
Date：2021/11/06~
Assignment 4 Kaggle score
Achieve PLSA and query likelihood
Doc Part
    Remove stop words
pLSA + Unigram + Background Language Model

DataSet：10000 Documents, 100 Queries(.txt)
Running time:  hours
"""
# -*- coding: utf-8 -*-

import os
import sys
import gzip
import utils
import random
import marshal
import numpy as np
from datetime import datetime


def cos_sim(p, q):
    sum0 = sum(map(lambda x: x * x, p))
    sum1 = sum(map(lambda x: x * x, q))
    sum2 = sum(map(lambda x: x[0] * x[1], zip(p, q)))
    return sum2 / (sum0 ** 0.5) / (sum1 ** 0.5)


def _rand_mat(sizex, sizey):
    ret = []
    for i in range(sizex):
        ret.append([])
        for _ in range(sizey):
            ret[-1].append(random.random())
        norm = sum(ret[-1])
        for j in range(sizey):
            ret[-1][j] /= norm
    return ret


class Plsa:
    def __init__(self, topics=2):
        self.Doc_str = 'D:/Python/NLP_DataSet/q_100_d_10000/docs/'
        self.Que_str = 'D:/Python/NLP_DataSet/q_100_d_10000/queries/'
        self.DocFileList = os.listdir(self.Doc_str)
        self.QueFileList = os.listdir(self.Que_str)
        self.topics = topics
        self.vocab = None
        self.docs = len(self.DocFileList)
        self.n_word = None
        self.likelihood = 0
        self.topic_word_prob = None
        self.document_topic_prob = None
        self.term_doc_matrix = None
        self.topic_prob = None
        self.BG_prob = None
        self.unigram_freqList = None
        self.unigram_freqDict = None

    def load(self, fname, iszip=True):
        if sys.version_info[0] == 3:
            fname = fname + '.3'
        if not iszip:
            d = marshal.load(open(fname, 'rb'))
        else:
            try:
                f = gzip.open(fname, 'rb')
                d = marshal.loads(f.read())
            except IOError:
                f = open(fname, 'rb')
                d = marshal.loads(f.read())
            f.close()
        for k, v in d.items():
            if hasattr(self.__dict__[k], '__dict__'):
                self.__dict__[k].__dict__ = v
            else:
                self.__dict__[k] = v

    def loadCorpus(self):
        self.vocab = []
        for docfile in self.QueFileList:
            f = open(self.Que_str + docfile)
            temp_split = f.read().split(' ')
            for word in temp_split:
                if word not in self.vocab:
                    self.vocab.append(word)
            f.close()
        self.n_word = len(self.vocab)
        print("n_word size : " + str(len(self.vocab)))

        "EM iteration begins..."
        # build term-doc matrix
        d_index = 0
        self.term_doc_matrix = np.zeros([self.docs, self.n_word])
        for docfile in self.DocFileList:
            document = open(self.Doc_str + docfile)
            segDoc = document.read().split(' ')
            term_count = np.zeros(self.n_word)
            for word in set(segDoc):
                if word in self.vocab:
                    w_index = self.vocab.index(word)
                    term_count[w_index] = term_count[w_index] + segDoc.count(word)
            self.term_doc_matrix[d_index] = term_count
            d_index += 1

        # Create the counter arrays.
        self.document_topic_prob = np.zeros([self.docs, self.topics])  # P(z | d)
        self.topic_word_prob = np.zeros([self.topics, len(self.vocab)])  # P(w | z)
        self.topic_prob = np.zeros([self.docs, len(self.vocab), self.topics])  # P(z | d, w)

        # Initialize
        print("Initializing...")
        # randomly assign values
        self.document_topic_prob = np.random.random(size=(self.docs, self.topics))
        for d_index in range(self.docs):
            utils.normalize(self.document_topic_prob[d_index])  # normalize for each document
        self.topic_word_prob = np.random.random(size=(self.topics, len(self.vocab)))
        for z in range(self.topics):
            utils.normalize(self.topic_word_prob[z])  # normalize for each topic

    # Run the EM algorithm
    def train(self, max_iter=50):
        for iteration in range(max_iter):
            print(datetime.now().strftime('%H:%M:%S') + " Iter #" + str(iteration + 1))
            print("E step:")
            for d_index, docname in enumerate(self.DocFileList):
                for w_index in range(self.n_word):
                    prob = self.document_topic_prob[d_index, :] * self.topic_word_prob[:, w_index]
                    if sum(prob) != 0.0:
                        utils.normalize(prob)
                    self.topic_prob[d_index][w_index] = prob
            print("M step:")
            # update P(w | z)
            for z in range(self.topics):
                for w_index in range(self.n_word):
                    s = 0
                    for d_index in range(self.docs):
                        count = self.term_doc_matrix[d_index][w_index]
                        s = s + count * self.topic_prob[d_index, w_index, z]
                    self.topic_word_prob[z][w_index] = s
                utils.normalize(self.topic_word_prob[z])

            # update P(z | d)
            for d_index in range(self.docs):
                for z in range(self.topics):
                    s = 0
                    s2 = 0
                    for w_index in range(self.n_word):
                        count = self.term_doc_matrix[d_index][w_index]
                        s = s + count * self.topic_prob[d_index, w_index, z]
                        s2 = s2 + count
                    if s2 == 0:
                        self.document_topic_prob[d_index][z] = s
                    else:
                        utils.normalize(self.document_topic_prob[d_index])

    def BG_model(self):
        BG_frequecy = dict()
        Doc_len = []
        for word in self.vocab:
            BG_frequecy[word] = sum(self.term_doc_matrix[:, self.vocab.index(word)])
        for docfile in self.DocFileList:
            dFile = open(self.Doc_str + docfile)
            segDc = dFile.read().split(' ')
            Doc_len.append(len(segDc))
        self.BG_prob = dict()
        for word in BG_frequecy:
            self.BG_prob[word] = (BG_frequecy[word] / sum(Doc_len))
        print("BG_model length：" + str(len(self.BG_prob)))

    def Unigram_model2(self):
        self.unigram_freqList = []
        Doc_len = []
        for docfile in self.DocFileList:
            dFile = open(self.Doc_str + docfile)
            segDc = dFile.read().split(' ')
            self.unigram_freqDict = dict()
            Doc_len.append(len(segDc))
            for word in set(segDc):
                if word in self.vocab:
                    self.unigram_freqDict[word] = segDc.count(word) / len(segDc)
            self.unigram_freqList.append(self.unigram_freqDict)
        print("Unigram_Model length：" + str(len(self.unigram_freqList)))

    def score(self):
        use2sort = dict.fromkeys(self.DocFileList, 0)
        for quefile in self.QueFileList:
            qFile = open(self.Que_str + quefile)
            qword = qFile.read().split(' ')
            for i in range(0, 9999):
                total_score = 1
                for word in qword:
                    pLsa_score = 0
                    word_score = 0
                    if word in self.unigram_freqList[i].keys():
                        word_score += (self.unigram_freqList[i][word] * 0.77)
                    if word in self.BG_prob.keys():
                        word_score += (self.BG_prob[word] * 0.03)
                    for k in range(self.topics):
                        j = self.vocab.index(word)
                        pLsa_score += self.topic_word_prob[k][j] * self.document_topic_prob[i][k]
                    total_score *= (pLsa_score * 0.2 + word_score)
                use2sort[self.DocFileList[i]] = total_score
            sort_res = sorted(use2sort.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
            print(str(quefile) + " End")
            # Zero setting
            use2sort = dict.fromkeys(self.DocFileList, 0)
            # write result in file
            self.write2res(str(quefile), sort_res)

    def pre_write2res(self):
        write_path = 'D:/Python/NLP_DataSet/q_100_d_10000/result_4_42.txt'
        f = open(write_path, 'a')
        f.write("Query,RetrievedDocuments" + '\n')
        f.close()

    def write2res(self, quefile, sort_res):
        write_path = 'D:/Python/NLP_DataSet/q_100_d_10000/result_4_42.txt'
        f = open(write_path, 'a')
        f.write(quefile.replace('.txt', '') + ',')
        for i in range(0, 2999):
            f.write(sort_res[i][0].replace('.txt', ''))
            if i <= 2998:
                f.write(" ")
        f.write('\n')
        f.close()


if __name__ == '__main__':
    _plsa = Plsa(50)
    print("Build one class End")
    _plsa.loadCorpus()
    _plsa.train(50)
    print("Train End")
    _plsa.BG_model()
    _plsa.Unigram_model2()
    _plsa.pre_write2res()
    _plsa.score()
