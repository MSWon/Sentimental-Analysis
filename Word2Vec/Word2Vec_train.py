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

os.chdir("C:\\Users\\jbk48\\OneDrive\\바탕 화면\\nsmc-master")

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
    
model.save('Word2vec.model')
model.most_similar('팝콘/Noun',topn = 20)  ## topn = len(model.wv.vocab)


## model 불러오기
model = gensim.models.word2vec.Word2Vec.load('Word2vec.model')
model.most_similar('공포/Noun',topn = 20)  ## topn = len(model.wv.vocab)


###########################
##Plotting to Tensorboard##
###########################


os.chdir("C:\\Users\\jbk48\\OneDrive") 


max_size = len(model.wv.vocab)-1
w2v = np.zeros((max_size,model.layer1_size))


with codecs.open("metadata.tsv",'w+',encoding='utf8') as file_metadata:
    for i,word in enumerate(model.wv.index2word[:max_size]):
        w2v[i] = model.wv[word]
        file_metadata.write(word + "\n")

from tensorflow.contrib.tensorboard.plugins import projector

sess = tf.InteractiveSession()
##  Create 2D tensor called embedding that holds our embeddings ##  
with tf.device("/cpu:0"):
    embedding = tf.Variable(w2v, trainable = False,  name = 'embedding')   

tf.global_variables_initializer().run() 

path = 'word2vec'

saver = tf.train.Saver()
writer = tf.summary.FileWriter(path, sess.graph)

## adding into project
config = projector.ProjectorConfig()
embed = config.embeddings.add()
embed.tensor_name = 'embedding'
embed.metadata_path =  'C:\\Users\\jbk48\\OneDrive\\metadata.tsv'
# Specify the width and height of a single thumbnail.
projector.visualize_embeddings(writer, config)
saver.save(sess, path + '/model.ckpt' , global_step=max_size)
## cmd 실행후 -> tensorboard --logdir=./word2vec 입력
