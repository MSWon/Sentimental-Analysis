# -*- coding: utf-8 -*-
"""
Created on Mon May 21 15:02:41 2018

@author: jbk48
"""

from konlpy.tag import Twitter
import numpy as np
import gensim


class Word2Vec():
    
    def __init__(self):
        None

    def tokenize(self, doc):
        pos_tagger = Twitter()
        return ['/'.join(t) for t in pos_tagger.pos(doc, norm=True, stem=True)]
    
    def read_data(self, filename):
        with open(filename, 'r',encoding='utf-8') as f:
            data = [line.split('\t') for line in f.read().splitlines()]
            data = data[1:]
        return data  
    
    def Word2vec_model(self, model_name):
        
        model = gensim.models.word2vec.Word2Vec.load(model_name)
        return model
    
    def Convert2Vec(self, model_name, doc):  ## Convert corpus into vectors
        word_vec = []
        model = gensim.models.word2vec.Word2Vec.load(model_name)
        for sent in doc:
            sub = []
            for word in sent:
                if(word in model.wv.vocab):
                    sub.append(model.wv[word])
                else:
                    sub.append(np.random.uniform(-0.25,0.25,300)) ## used for OOV words
            word_vec.append(sub)
        
        return word_vec
    
    def Zero_padding(self, train_batch_X, Batch_size, Maxseq_length, Vector_size):
        
        zero_pad = np.zeros((Batch_size, Maxseq_length, Vector_size))
        for i in range(Batch_size):
            zero_pad[i,:np.shape(train_batch_X[i])[0],:np.shape(train_batch_X[i])[1]] = train_batch_X[i]
            
        return zero_pad
    
    def One_hot(self, data):
       
        index_dict = {value:index for index,value in enumerate(set(data))}
        result = []
        
        for value in data:
            
            one_hot = np.zeros(len(index_dict))
            index = index_dict[value]
            one_hot[index] = 1
            result.append(one_hot)
        
        return result
    
    
    