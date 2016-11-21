
"""Builds the ring network.

Summary of available functions:

  # Compute pics of the simulation runnig.
  
  # Create a graph to train on.
"""


import tensorflow as tf
import numpy as np
import architecture
import input.ring_net_input as ring_net_input

FLAGS = tf.app.flags.FLAGS

# Constants describing the training process.
tf.app.flags.DEFINE_string('model', 'lstm_210x160x3',
                           """ model name to train """)
tf.app.flags.DEFINE_string('atari_game', 'space_invaders.bin',
                            """atari game to run""")
tf.app.flags.DEFINE_float('moving_average_decay', 0.9999,
                          """The decay to use for the moving average""")
tf.app.flags.DEFINE_float('momentum', 0.9,
                          """momentum of learning rate""")
tf.app.flags.DEFINE_float('alpha', 0.1,
                          """Leaky RElu param""")
tf.app.flags.DEFINE_float('weight_decay', 0.0005,
                          """ """)

# possible models and systems to train are
# lstm_84x84x1 atari
# lstm_210x160x3 atari with rgb

def inputs(batch_size, seq_length):
  """makes input vector
  Return:
    x: input vector, may be filled 
  """
  state, reward, action = ring_net_input.atari_inputs(batch_size, seq_length)
  return state, reward, action 

def encoding(state, keep_prob):
  """Builds encoding part of ring net.
  Args:
    inputs: input to encoder
    keep_prob: dropout layer
  """
  #--------- Making the net -----------
  # x_1 -> y_1 -> y_2 -> x_2
  # this peice x_1 -> y_1
  if FLAGS.model == "lstm_84x84x1": 
    y_1 = architecture.encoding_84x84x1(state, keep_prob)
  elif FLAGS.model == "lstm_210x160x3": 
    y_1 = architecture.encoding_210x160x3(state, keep_prob)

  return y_1 

def lstm_compression(inputs, action, hidden_state, keep_prob):
  """Builds compressed dynamical system part of the net.
  Args:
    inputs: input to system
    keep_prob: dropout layer
  """
  #--------- Making the net -----------
  # x_1 -> y_1 -> y_2 -> x_2
  # this peice y_1 -> y_2
  if FLAGS.model == "lstm_84x84x1": 
    y_2, reward, hidden_state = architecture.lstm_compression_84x84x1(inputs, action, hidden_state, keep_prob)
  elif FLAGS.model == "lstm_210x160x3": 
    y_2, reward, hidden_state = architecture.lstm_compression_210x160x3(inputs, action, hidden_state, keep_prob)
  return y_2, reward, hidden_state

def decoding(inputs):
  """Builds decoding part of ring net.
  Args:
    inputs: input to decoder
  """
  #--------- Making the net -----------
  # x_1 -> y_1 -> y_2 -> x_2
  # this peice y_2 -> x_2
  if FLAGS.model == "lstm_84x84x1": 
    x_2 = architecture.decoding_84x84x1(inputs)
  elif FLAGS.model == "lstm_210x160x3": 
    x_2 = architecture.decoding_210x160x3(inputs)

  return x_2 

def encode_compress_decode(state, action, hidden_state, keep_prob_encoding, keep_prob_lstm):
  
  y_1 = encoding(state, keep_prob_encoding)
  y_2, reward_2, hidden_state = lstm_compression(y_1, action, hidden_state, keep_prob_encoding)
  x_2 = decoding(y_2) 

  return x_2, reward_2, hidden_state

def train(total_loss, lr):
   train_op = tf.train.AdamOptimizer(lr).minimize(total_loss)
   return train_op

