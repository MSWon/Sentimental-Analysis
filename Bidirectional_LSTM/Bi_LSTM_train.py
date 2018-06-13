# -*- coding: utf-8 -*-
"""
Created on Mon May 21 16:14:45 2018

@author: jbk48
"""

import time
import os
import tensorflow as tf
import numpy as np
import Bi_LSTM
import Word2Vec

W2V = Word2Vec.Word2Vec()

os.chdir("C:\\Users\\jbk48\\Desktop\\Sentimental-Analysis-master\\Sentimental-Analysis-master\\Word2Vec\\Movie_rating_data")
train_data = W2V.read_data("ratings_train.txt")

## tokenize the data we have
print("Tokenize Start!\nCould take minutes...")
tokens = [[W2V.tokenize(row[1]),int(row[2])] for row in train_data if W2V.tokenize(row[1]) != []]
tokens = np.array(tokens)
print("Tokenize Done!")

train_X = tokens[:,0]
train_Y = tokens[:,1]

os.chdir("C:\\Users\\jbk48\\Desktop\\Sentimental-Analysis-master\\Sentimental-Analysis-master\\Word2Vec")
train_Y_ = W2V.One_hot(train_Y)  ## Convert to One-hot
train_X_ = W2V.Convert2Vec("Word2vec.model",train_X)  ## import word2vec model where you have trained before

Batch_size = 32
Total_size = len(train_X)
Vector_size = 300
seq_length = [len(x) for x in train_X]
Maxseq_length = max(seq_length)
learning_rate = 0.001
lstm_units = 128
num_class = 2
training_epochs = 10
keep_prob = 0.75

X = tf.placeholder(tf.float32, shape = [None, Maxseq_length, Vector_size], name = 'X')
Y = tf.placeholder(tf.float32, shape = [None, num_class], name = 'Y')
seq_len = tf.placeholder(tf.int32, shape = [None])

BiLSTM = Bi_LSTM.Bi_LSTM(lstm_units, Maxseq_length, num_class, keep_prob)

with tf.variable_scope("loss", reuse = tf.AUTO_REUSE):
    logits = BiLSTM.logits(X, BiLSTM.W, BiLSTM.b, seq_len, Maxseq_length)
    loss, optimizer, merged = BiLSTM.model_build(logits, Y, learning_rate)

prediction = tf.nn.softmax(logits)
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

total_batch = int(Total_size / Batch_size)

print("Start training!")

modelName = "C:\\Users\\jbk48\\Desktop\\Sentimental-Analysis-master\\Sentimental-Analysis-master\\Bidirectional_LSTM\\BiLSTM_model.ckpt"
saver = tf.train.Saver()


with tf.Session() as sess:

    start_time = time.time()
    sess.run(init)
    ## train_writer = tf.summary.FileWriter('C:\\Users\\jbk48\\Desktop\\Sentimental-Analysis-master\\Sentimental-Analysis-master\\Bidirectional_LSTM', sess.graph)
    ## i = 0
    for epoch in range(training_epochs):

        avg_loss = 0.
        for step in range(total_batch):

            train_batch_X = train_X_[step*Batch_size : step*Batch_size+Batch_size]
            train_batch_Y = train_Y_[step*Batch_size : step*Batch_size+Batch_size]
            batch_seq_length = seq_length[step*Batch_size : step*Batch_size+Batch_size]
            
            train_batch_X = W2V.Zero_padding(train_batch_X, Batch_size, Maxseq_length, Vector_size)

            ## summary, _ = sess.run([merged,optimizer], feed_dict={X: train_batch_X, Y: train_batch_Y, seq_len: batch_seq_length})
            sess.run(optimizer, feed_dict={X: train_batch_X, Y: train_batch_Y, seq_len: batch_seq_length})
            # Compute average loss
            loss_ = sess.run(loss, feed_dict={X: train_batch_X, Y: train_batch_Y, seq_len: batch_seq_length})
            avg_loss += sess.run(loss, feed_dict={X: train_batch_X, Y: train_batch_Y, seq_len: batch_seq_length})/total_batch
            acc = sess.run(accuracy , feed_dict={X: train_batch_X, Y: train_batch_Y, seq_len: batch_seq_length})
            ## train_writer.add_summary(summary, i)
            ## i += 1
            print("step:", '%04d' %(step+1), "loss = {:.6f} accuracy= {:.6f}".format(loss_, acc))
    

    duration = time.time() - start_time
    minute = int(duration / 60)
    second = int(duration) % 60
    print("%dminutes %dseconds" % (minute,second))
    save_path = saver.save(sess, modelName)
    ## train_writer.close()
    print ('save_path',save_path)
    
    ## cmd 실행 -> cd C:\Users\jbk48\Desktop\Sentimental-Analysis-master\Sentimental-Analysis-master\Bidirectional_LSTM
    ## tensorboard --logdir=./ 입력
