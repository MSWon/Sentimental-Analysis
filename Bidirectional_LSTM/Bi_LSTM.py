# -*- coding: utf-8 -*-
"""
Created on Mon May 21 14:30:52 2018

@author: jbk48
"""

import tensorflow as tf

class Bi_LSTM():
    
    def __init__(self, lstm_units, num_class, keep_prob):
        
        self.lstm_units = lstm_units
        
        with tf.variable_scope('forward', reuse = tf.AUTO_REUSE):
            
            self.lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(lstm_units, forget_bias=1.0, state_is_tuple=True)
            self.lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(self.lstm_fw_cell, output_keep_prob = keep_prob)
            
        with tf.variable_scope('backward', reuse = tf.AUTO_REUSE):
            
            self.lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(lstm_units, forget_bias=1.0, state_is_tuple=True)
            self.lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(self.lstm_fw_cell, output_keep_prob = keep_prob)
        
        with tf.variable_scope('Weights', reuse = tf.AUTO_REUSE):
           
            self.W = tf.get_variable(name="W", shape=[2 * lstm_units, num_class],
                                dtype=tf.float32, initializer = tf.contrib.layers.xavier_initializer())
            self.b = tf.get_variable(name="b", shape=[num_class], dtype=tf.float32,
                                initializer=tf.zeros_initializer())
            
            
    def logits(self, X, W, b, seq_len):
        
        (output_fw, output_bw), states = tf.nn.bidirectional_dynamic_rnn(self.lstm_fw_cell, self.lstm_bw_cell,dtype=tf.float32,
                                                                            inputs = X, sequence_length = seq_len)
        ## concat fw, bw final states
        outputs = tf.concat([states[0][1], states[1][1]], axis=1)
        pred = tf.matmul(outputs, W) + b        
        return pred
        
    def model_build(self, logits, labels, learning_rate = 0.001):
        
        with tf.variable_scope("loss"):
            
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits , labels = labels)) # Softmax loss
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss) # Adam Optimizer
            
        return loss, optimizer
    
    def graph_build(self, avg_loss, avg_acc):
        
        tf.summary.scalar('Loss', avg_loss)
        tf.summary.scalar('Accuracy', avg_acc)
        merged = tf.summary.merge_all()
        return merged
