# -*- coding: utf-8 -*-
"""
Project 2
Dallas Dituri & Jose Garcia
"""
import numpy as np
import tensorflow as tf

# Map input data to tensorflow
data = open('C:/Users/dcdit/Desktop/alice.txt', 'r').read() # should be simple plain text file
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

# hyperparameters
hidden_size = 100 # size of hidden layer of neurons
seq_length = 25 # number of steps to unroll the RNN for
learning_rate = 1e-1

# preprocessing
X = []
y = []
for i in range (0, len(data)-seq_length-1, 1):
    X.append([char_to_ix[ch] for ch in data[i:i+seq_length]])
    y.append([char_to_ix[ch] for ch in data[i+1:i+seq_length+1]])
X_modified = np.reshape(X, (len(X), seq_length))
y_modified = np.reshape(y, (len(y), seq_length))

X0_weight = np.random.randn(hidden_size, vocab_size)*0.01
X1_weight = np.random.randn(hidden_size, hidden_size)*0.01
X2_weight = np.random.randn(vocab_size, hidden_size)*0.01

tf.reset_default_graph()
# model parameters
X0 = tf.placeholder(tf.float32, [None, seq_length]) # input to hidden
# X1 = tf.placeholder(tf.float32, [None, seq_length]) # hidden to hidden
y = tf.placeholder(tf.float32, [None, seq_length])
rnn_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_size)
outputs, states = tf.nn.dynamic_rnn(rnn_cell, X0, dtype=tf.float32)

# Wx0 = tf.Variable(tf.random_normal(shape=[seq_length, hidden_size],dtype=tf.float32))
# Wx1 = tf.Variable(tf.random_normal(shape=[hidden_size, hidden_size],dtype=tf.float32))
# Wy = tf.Variable(tf.random_normal(shape=[hidden_size, seq_length],dtype=tf.float32))
# hidden_bias = tf.Variable(tf.zeros([1, hidden_size], dtype=tf.float32))
# output_bias = tf.Variable(tf.zeros([1, vocab_size], dtype=tf.float32))
init = tf.gloabal_varaibles_initializer()
with tf.Session() as sess:
    init.run()
    outputs_val = outputs.eval(feed_dict={X: X_modified})
    
