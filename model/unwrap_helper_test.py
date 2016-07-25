
"""helper functions to unrap the network for testing.
"""
import tensorflow as tf
import numpy as np
import ring_net 
import architecture

FLAGS = tf.app.flags.FLAGS

def encoding(inputs, keep_prob):
  # calc y_0
  y_0 = ring_net.encoding(inputs[:, 0, :, :, :], keep_prob)  
  return y_0

def fully_connected_step(y_0, keep_prob):
  # calc x_0
  x_0 = ring_net.decoding(y_0)
 
  # calc next state
  y_1 = ring_net.compression(y_0, keep_prob)

  return x_0, y_1

def lstm_step(y_0, hidden_state, keep_prob):
  # calc x_0
  x_0 = ring_net.decoding(y_0)
  
  # calc next state
  y_1, hidden_state = ring_net.lstm_compression(y_0, hidden_state, keep_prob)

  return x_0, y_1, hidden_state

