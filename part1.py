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

tf.reset_default_graph()

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

# model parameters
X0 = tf.placeholder(tf.float32, [None, seq_length]) # input to hidden
y = tf.placeholder(tf.float32, [None, vocab_size])

X_onehot = tf.one_hot(X_modified, vocab_size)
y_onehot = tf.one_hot(y_modified, vocab_size)
cell = tf.contrib.rnn.OutputProjectionWrapper(
        tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_size, activation=tf.nn.leaky_relu),
        output_size=vocab_size)
outputs, states = tf.nn.dynamic_rnn(cell, X_onehot, dtype=tf.float32)

prediction = tf.nn.softmax(outputs)
loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=outputs, labels=y_modified))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
train_op = optimizer.minimize(loss_op)

init = tf.global_variables_initializer()

def mini_batches(X, y, batchsize, shuffle=False):
   assert X.shape[0] == y.shape[0]
   if shuffle:
       indices = np.arange(X.shape[0])
       np.random.shuffle(indices)
   for start in range(0, X.shape[0] - batchsize + 1, batchsize):
       if shuffle:
           batch_indexes = indices[start:start + batchsize]
       else:
           batch_indexes = slice(start,start + batchsize)
       yield X[batch_indexes], y[batch_indexes]

def fetch_batch(ix,iteration):
   np.random.seed(ix+iteration)
   indices = np.random.randint(X_modified.shape[1], size=iteration)
   X_batch = list()
   y_batch = list()
   for i in indices:
       X_batch.append(X_modified[i])
       y_batch.append(y_modified[i])

   X_batch, y_batch = np.asanyarray(X_batch), np.asanyarray(y_batch)
   X_batch = X_batch.reshape(-1,seq_length, vocab_size)
   y_batch = y_batch.reshape(-1,seq_length, vocab_size)
   return X_batch, y_batch

n_epochs = 10
batch_size = 25

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        X_batch, y_batch = fetch_batch(epoch, batch_size)
        sess.run(train_op, feed_dict={X: X_batch[epoch], y: y_batch[epoch]})
        
    if epoch % 1 == 0:
        print(prediction[0])
      