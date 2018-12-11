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

X0_weight = np.random.randn(vocab_size, hidden_size)*0.01
X1_weight = np.random.randn(hidden_size, hidden_size)*0.01
X2_weight = np.random.randn(vocab_size, hidden_size)*0.01

tf.reset_default_graph()

# model parameters
X0 = tf.placeholder(tf.float32, [None, seq_length]) # input to hidden
y = tf.placeholder(tf.float32, [None, vocab_size])

X_onehot = tf.one_hot(X_modified, vocab_size)
cell = tf.contrib.rnn.OutputProjectionWrapper(
        tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_size, activation=tf.nn.leaky_relu),
        output_size=vocab_size)
outputs, states = tf.nn.dynamic_rnn(cell, X_onehot, dtype=tf.float32)

prediction = tf.nn.softmax(outputs)
loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=outputs, labels=y_modified))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
train_op = optimizer.minimize(loss_op)

init = tf.global_varaibles_initializer()

with tf.Session() as sess:
    init.run()
    outputs_val = outputs.eval(feed_dict={X: X_modified})
    
"""