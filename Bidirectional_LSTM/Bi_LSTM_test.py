# -*- coding: utf-8 -*-
"""
Created on Wed May 23 21:16:52 2018

@author: jbk48
"""


import os
import tensorflow as tf
import numpy as np
import Bi_LSTM
import Word2Vec

W2V = Word2Vec.Word2Vec()
os.chdir("..")
test_data = W2V.read_data("./Word2Vec/Movie_rating_data/ratings_test.txt")

## tokenize the data we have
print("Tokenize Start!\nCould take minutes...")
tokens = [[W2V.tokenize(row[1]),int(row[2])] for row in test_data if W2V.tokenize(row[1]) != []]
tokens = np.array(tokens)
print("Tokenize Done!")


test_X = tokens[:,0]
test_Y = tokens[:,1]

Batch_size = 32
test_size = len(test_X)
test_batch = int(test_size / Batch_size)
seq_length = [len(x) for x in test_X]
Vector_size = 300
Maxseq_length = 95   ## Max length of training data
learning_rate = 0.001
lstm_units = 128
num_class = 2
keep_prob = 1.0


test_Y_ = W2V.One_hot(test_Y)  ## Convert to One-hot
test_X_ = W2V.Convert2Vec("./Word2Vec/Word2vec.model",test_X)  ## import word2vec model where you have trained before

X = tf.placeholder(tf.float32, shape = [None, Maxseq_length, Vector_size], name = 'X')
Y = tf.placeholder(tf.float32, shape = [None, num_class], name = 'Y')
seq_len = tf.placeholder(tf.int32, shape = [None])

BiLSTM = Bi_LSTM.Bi_LSTM(lstm_units, num_class, keep_prob)

with tf.variable_scope("loss", reuse = tf.AUTO_REUSE):
    logits = BiLSTM.logits(X, BiLSTM.W, BiLSTM.b, seq_len)
    loss, optimizer = BiLSTM.model_build(logits, Y, learning_rate)

prediction = tf.nn.softmax(logits)
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

modelName = "./Bidirectional_LSTM/BiLSTM_model.ckpt"
init = tf.global_variables_initializer()
saver = tf.train.Saver()


with tf.Session() as sess:
    
    sess.run(init)
    # load the variables from disk.
    saver.restore(sess, modelName)
    print("Model restored")

    total_acc = 0

    for step in range(test_batch):

        test_batch_X = test_X_[step*Batch_size : step*Batch_size+Batch_size]
        test_batch_Y = test_Y_[step*Batch_size : step*Batch_size+Batch_size]
        batch_seq_length = seq_length[step*Batch_size : step*Batch_size+Batch_size]
        test_batch_X = W2V.Zero_padding(test_batch_X, Batch_size, Maxseq_length, Vector_size)

        acc = sess.run(accuracy , feed_dict={X: test_batch_X, Y: test_batch_Y, seq_len: batch_seq_length})
        print("step :{} Accuracy : {}".format(step+1,acc))
        total_acc += acc/test_batch

    print("Total Accuracy : {}".format(total_acc))
