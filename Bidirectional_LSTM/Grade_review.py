# -*- coding: utf-8 -*-
"""
Created on Thu May 24 11:15:44 2018

@author: jbk48
"""

import os
import tensorflow as tf
import Bi_LSTM
import Word2Vec

W2V = Word2Vec.Word2Vec()

Batch_size = 1
Vector_size = 300
Maxseq_length = 95   ## Max length of training data
learning_rate = 0.001
lstm_units = 128
num_class = 2
keep_prob = 1.0

X = tf.placeholder(tf.float32, shape = [None, Maxseq_length, Vector_size], name = 'X')
Y = tf.placeholder(tf.float32, shape = [None, num_class], name = 'Y')
seq_len = tf.placeholder(tf.int32, shape = [None])

BiLSTM = Bi_LSTM.Bi_LSTM(lstm_units, Maxseq_length, num_class, keep_prob)

with tf.variable_scope("loss", reuse = tf.AUTO_REUSE):
    logits = BiLSTM.logits(X, BiLSTM.W, BiLSTM.b, seq_len, Maxseq_length)
    ## loss, optimizer, _ = BiLSTM.model_build(logits, Y, learning_rate)
    loss, optimizer = BiLSTM.model_build(logits, Y, learning_rate)

prediction = tf.nn.softmax(logits)

os.chdir("C:\\Users\\jbk48\\Desktop\\Sentimental-Analysis-master\\Sentimental-Analysis-master\\Word2Vec")

def Grade(sentence):

    tokens = W2V.tokenize(sentence)
    embedding = W2V.Convert2Vec('Word2vec.model', tokens)
    zero_pad = W2V.Zero_padding(embedding, Batch_size, Maxseq_length, Vector_size)
        
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    with tf.Session() as sess:    
        
        sess.run(init)    
        modelName = "C:\\Users\\jbk48\\Desktop\\Sentimental-Analysis-master\\Sentimental-Analysis-master\\Bidirectional_LSTM\\BiLSTM_model.ckpt"
        # load the variables from disk.
        saver.restore(sess, modelName)
        
        result = sess.run(tf.argmax(prediction,1), feed_dict = {X: zero_pad , seq_len: [len(tokens)] } ) 
        if(result == 1):
            print("긍정입니다")
        else:
            print("부정입니다")
            

while(1):
    
    s = input("문장을 입력하세요 : ")
    if(s == 1):
        break
    else:
        Grade(s)


