
"""Builds the ring network.

Summary of available functions:

  # Compute pics of the simulation runnig.
  
  # Create a graph to train on.
"""


import tensorflow as tf
import numpy as np
import architecture
import unwrap_helper
import input.ring_net_input as ring_net_input

FLAGS = tf.app.flags.FLAGS

# Constants describing the training process.
tf.app.flags.DEFINE_string('model', 'lstm_210x160x3',
                           """ model name to train """)
tf.app.flags.DEFINE_string('atari_game', 'space_invaders.bin',
                            """atari game to run""")
tf.app.flags.DEFINE_string('system', 'cannon',
                           """ system to compress """)
tf.app.flags.DEFINE_float('moving_average_decay', 0.9999,
                          """The decay to use for the moving average""")
tf.app.flags.DEFINE_float('momentum', 0.9,
                          """momentum of learning rate""")
tf.app.flags.DEFINE_float('alpha', 0.1,
                          """Leaky RElu param""")
tf.app.flags.DEFINE_float('weight_decay', 0.0005,
                          """ """)
tf.app.flags.DEFINE_float('dropout_hidden', 0.5,
                          """ dropout on hidden """)
tf.app.flags.DEFINE_float('dropout_input', 0.8,
                          """ dropout on input """)
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

def unwrap(state, action, keep_prob_encoding, keep_prob_lstm, seq_length, train_peice, return_hidden=False):
  """Unrap the system for training.
  Args:
    inputs: input to system, should be [minibatch, seq_length, image_size]
    action: input action to the system, should be [minibatch, seq_length, action]
    keep_prob: dropout layers
    seq_length: how far to unravel 
 
  Return: 
    output_t: calculated y values from iterating t'
    output_g: calculated x values from g
    output_f: calculated y values from f 
  """
  if return_hidden:
    output_f, output_t, output_g, output_reward, output_autoencoder, hidden = unwrap_helper.lstm_unwrap(state, action, keep_prob_encoding, keep_prob_lstm, seq_length, train_peice, return_hidden)
    return output_f, output_t, output_g, output_reward, output_autoencoder, hidden
  else:
    output_f, output_t, output_g, output_reward, output_autoencoder = unwrap_helper.lstm_unwrap(state, action, keep_prob_encoding, keep_prob_lstm, seq_length, train_peice, return_hidden)
    return output_f, output_t, output_g, output_reward, output_autoencoder
  


def loss(state, reward, output_f, output_t, output_g, output_reward, output_autoencoding, train_piece):
  """Calc loss for unrap output.
  Args.
    inputs: true x values
    output_t: calculated y values from iterating t'
    output_g: calculated x values from g
    output_f: calculated y values from f 

  Return:
    error: loss value
  """
  # constants in loss
  autoencoder_loss_constant = 1.0
  compression_loss_constant = 1.0
  reward_loss_constant = 1.0

  # autoencoder loss peice
  print(state.get_shape())
  print(output_autoencoding.get_shape())
  loss_reconstruction_autoencoder = tf.nn.l2_loss(state - output_autoencoding)
  if train_piece == "all":
    loss_reconstruction_autoencoder = autoencoder_loss_constant * loss_reconstruction_autoencoder
  tf.scalar_summary('loss_reconstruction_autoencoder', loss_reconstruction_autoencoder)
   
  # compression loss piece
  seq_length = int(state.get_shape()[1])
  if seq_length > 1 and train_piece == "all":
    print(output_f.get_shape())
    print(output_t.get_shape())
    loss_t = compression_loss_constant * tf.nn.l2_loss(output_f[:,5:,:] - output_t[:,4:seq_length-1,:])
    # check this peice
    print(reward[:,5:,:].get_shape())
    print(output_reward[:,4:seq_length-1,:].get_shape())
    loss_reward = reward_loss_constant*tf.nn.l2_loss(reward[:,5:,:] - output_reward[:,4:seq_length-1,:])
    tf.scalar_summary('loss_t', loss_t)
    tf.scalar_summary('loss_reward', loss_reward)
  else:
    loss_t = 0.0
    loss_reward = 0.0

  total_loss = tf.reduce_sum(loss_reconstruction_autoencoder + loss_t + loss_reward)
  tf.scalar_summary('total_loss', total_loss)

  return total_loss 

def train(total_loss, lr):
   train_op = tf.train.AdamOptimizer(lr).minimize(total_loss)
   return train_op

