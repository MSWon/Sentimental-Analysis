# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 01:44:28 2017

@author: Big Data Guru
"""

import os
from konlpy.tag import Twitter
import gensim 
import tensorflow as tf
import numpy as np
import codecs

os.chdir("C:\\Users\\jbk48\\Desktop\\Sentimental-Analysis-master\\Sentimental-Analysis-master\\Word2Vec\\Movie_rating_data")

def read_data(filename):    
    with open(filename, 'r',encoding='utf-8') as f:
        data = [line.split('\t') for line in f.read().splitlines()]        
        data = data[1:]   # header 제외 #    
    return data 
    
train_data = read_data('ratings_train.txt') 
test_data = read_data('ratings_test.txt') 

pos_tagger = Twitter() 

def tokenize(doc):

    return ['/'.join(t) for t in pos_tagger.pos(doc, norm=True, stem=True)]


## training Word2Vec model using skip-gram   
tokens = [tokenize(row[1]) for row in train_data]
model = gensim.models.Word2Vec(size=300,sg = 1, alpha=0.025,min_alpha=0.025, seed=1234)
model.build_vocab(tokens)
    
for epoch in range(30):
           
    model.train(tokens,model.corpus_count,epochs = model.iter)
    model.alpha -= 0.002
    model.min_alpha = model.alpha
    
os.chdir("C:\\Users\\jbk48\\Desktop\\Sentimental-Analysis-master\\Sentimental-Analysis-master\\Word2Vec")    
model.save('Word2vec.model')
model.most_similar('팝콘/Noun',topn = 20)  ## topn = len(model.wv.vocab)
