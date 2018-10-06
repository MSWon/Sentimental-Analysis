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
import pandas as pd


config = tf.ConfigProto()
config.gpu_options.allow_growth = True

W2V = Word2Vec.Word2Vec()
os.chdir("..")
train_data = W2V.read_data("./Word2Vec/Movie_rating_data/ratings_train.txt")
test_data = W2V.read_data("./Word2Vec/Movie_rating_data/ratings_test.txt")

## tokenize the data we have
print("Tokenize Start!\nCould take minutes...")
train_tokens = [[W2V.tokenize(row[1]),int(row[2])] for row in train_data if W2V.tokenize(row[1]) != []]
train_tokens = np.array(train_tokens)
test_tokens = [[W2V.tokenize(row[1]),int(row[2])] for row in test_data if W2V.tokenize(row[1]) != []]
test_tokens = np.array(test_tokens)

print("Tokenize Done!")

train_X = train_tokens[:,0]
train_Y = train_tokens[:,1]
test_X = test_tokens[:,0]
test_Y = test_tokens[:,1]

train_Y_ = W2V.One_hot(train_Y)  ## Convert to One-hot
train_X_ = W2V.Convert2Vec("./Word2Vec/Word2vec.model",train_X)  ## import word2vec model where you have trained before
test_Y_ = W2V.One_hot(test_Y)  ## Convert to One-hot
test_X_ = W2V.Convert2Vec("./Word2Vec/Word2vec.model",test_X)  ## import word2vec model where you have trained before


Batch_size = 32
Total_size = len(train_X)
Vector_size = 300
train_seq_length = [len(x) for x in train_X]
test_seq_length = [len(x) for x in test_X]
Maxseq_length = max(train_seq_length) ## 95
learning_rate = 0.001
lstm_units = 128
num_class = 2
training_epochs = 4

X = tf.placeholder(tf.float32, shape = [None, Maxseq_length, Vector_size], name = 'X')
Y = tf.placeholder(tf.float32, shape = [None, num_class], name = 'Y')
seq_len = tf.placeholder(tf.int32, shape = [None])
keep_prob = tf.placeholder(tf.float32, shape = None)

BiLSTM = Bi_LSTM.Bi_LSTM(lstm_units, num_class, keep_prob)

with tf.variable_scope("loss", reuse = tf.AUTO_REUSE):
    logits = BiLSTM.logits(X, BiLSTM.W, BiLSTM.b, seq_len)
    loss, optimizer = BiLSTM.model_build(logits, Y, learning_rate)

prediction = tf.nn.softmax(logits)
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

total_batch = int(len(train_X) / Batch_size)
test_batch = int(len(test_X) / Batch_size)

print("Start training!")

modelName = "./Bidirectional_LSTM/BiLSTM_model.ckpt"
saver = tf.train.Saver()

train_acc = []
train_loss = []
test_acc = []
test_loss = []


with tf.Session(config = config) as sess:

    start_time = time.time()
    sess.run(init)
    train_writer = tf.summary.FileWriter('./Bidirectional_LSTM', sess.graph)
    merged = BiLSTM.graph_build()
    
    for epoch in range(training_epochs):

        avg_acc, avg_loss = 0. , 0.
        mask = np.random.permutation(len(train_X_))
        train_X_ = train_X_[mask]
        train_Y_ = train_Y_[mask]
        
        for step in range(total_batch):

            train_batch_X = train_X_[step*Batch_size : step*Batch_size+Batch_size]
            train_batch_Y = train_Y_[step*Batch_size : step*Batch_size+Batch_size]
            batch_seq_length = train_seq_length[step*Batch_size : step*Batch_size+Batch_size]
            
            train_batch_X = W2V.Zero_padding(train_batch_X, Batch_size, Maxseq_length, Vector_size)
            
            sess.run(optimizer, feed_dict={X: train_batch_X, Y: train_batch_Y, seq_len: batch_seq_length})
            # Compute average loss
            loss_ = sess.run(loss, feed_dict={X: train_batch_X, Y: train_batch_Y, seq_len: batch_seq_length,
                                              keep_prob : 0.75})
            avg_loss += loss_ / total_batch
            
            acc = sess.run(accuracy , feed_dict={X: train_batch_X, Y: train_batch_Y, seq_len: batch_seq_length,
                                                 keep_prob : 0.75})
            avg_acc += acc / total_batch
            print("epoch : {:02d} step : {:04d} loss = {:.6f} accuracy= {:.6f}".format(epoch+1, step+1, loss_, acc))
   
        summary = sess.run(merged, feed_dict = {BiLSTM.loss : avg_loss, BiLSTM.acc : avg_acc})       
        train_writer.add_summary(summary, epoch)
    
        t_avg_acc, t_avg_loss = 0. , 0.
        print("Test cases, could take few minutes")
        for step in range(test_batch):
            
            test_batch_X = test_X_[step*Batch_size : step*Batch_size+Batch_size]
            test_batch_Y = test_Y_[step*Batch_size : step*Batch_size+Batch_size]
            batch_seq_length = test_seq_length[step*Batch_size : step*Batch_size+Batch_size]
            
            test_batch_X = W2V.Zero_padding(test_batch_X, Batch_size, Maxseq_length, Vector_size)
            
            # Compute average loss
            loss2 = sess.run(loss, feed_dict={X: test_batch_X, Y: test_batch_Y, seq_len: batch_seq_length,
                                              keep_prob : 1.0})
            t_avg_loss += loss2 / test_batch
            
            t_acc = sess.run(accuracy , feed_dict={X: test_batch_X, Y: test_batch_Y, seq_len: batch_seq_length,
                                                   keep_prob : 1.0})
            t_avg_acc += t_acc / test_batch

        print("<Train> Loss = {:.6f} Accuracy = {:.6f}".format(avg_loss, avg_acc))
        print("<Test> Loss = {:.6f} Accuracy = {:.6f}".format(t_avg_loss, t_avg_acc))
        train_loss.append(avg_loss)
        train_acc.append(avg_acc)
        test_loss.append(t_avg_loss)
        test_acc.append(t_avg_acc)

    train_loss = pd.DataFrame({"train_loss":train_loss})
    train_acc = pd.DataFrame({"train_acc":train_acc})
    test_loss = pd.DataFrame({"test_loss":test_loss})
    test_acc = pd.DataFrame({"test_acc":test_acc})
    df = pd.concat([train_loss,train_acc,test_loss,test_acc], axis = 1)
    df.to_csv("./Bidirectional_LSTM/loss_accuracy.csv", sep =",", index=False)
    
    train_writer.close()
    duration = time.time() - start_time
    minute = int(duration / 60)
    second = int(duration) % 60
    print("%dminutes %dseconds" % (minute,second))
    save_path = saver.save(sess, modelName)
    
    print ('save_path',save_path)
    
