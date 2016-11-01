
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
  if FLAGS.model == "lstm_84x84x4": 
    y_2, reward, hidden_state = architecture.lstm_compression_84x84x4(inputs, action, hidden_state, keep_prob)
  elif FLAGS.model == "lstm_210x160x12": 
    y_2, reward, hidden_state = architecture.lstm_compression_210x160x12(inputs, action, hidden_state, keep_prob)
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
    x_2 = architecture.decoding_84x84x4(inputs)
  elif FLAGS.model == "lstm_210x160x3": 
    x_2 = architecture.decoding_210x160x12(inputs)

  return x_2 

def unwrap(state, action, keep_prob, seq_length):
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

  output_t, output_g, output_f, output_reward = unwrap_helper.lstm_unwrap(state, action, keep_prob, seq_length)

  return output_t, output_g, output_f, output_reward 

def loss(state, reward, output_t, output_g, output_f, output_reward):
  """Calc loss for unrap output.
  Args.
    inputs: true x values
    output_t: calculated y values from iterating t'
    output_g: calculated x values from g
    output_f: calculated y values from f 

  Return:
    error: loss value
  """
  # calc encodeing error
  error_xg = tf.nn.l2_loss(output_g - state)
  tf.scalar_summary('error_xg', error_xg)

  if output_f is not None:
    # calc tf error
    # Scale the t f error based on the ratio of image size to compressed size. This has somewhat undetermined effects
    if FLAGS.model in ("fully_connected_84x84x4", "lstm_84x84x4"):
      tf_scaling_factor = 30.0
    elif FLAGS.model in ("fully_connected_210x160x12", "lstm_210x160x12"):
      tf_scaling_factor = 200.0
    error_tf = tf.mul(tf_scaling_factor, tf.nn.l2_loss(output_f - output_t))
    tf.scalar_summary('error_tf', error_tf)
    
    # calc reward error 
    # Scale the reward error based on the ratio of image size to reward size. This has somewhat undetermined effects
    if FLAGS.model in ("fully_connected_84x84x4", "lstm_84x84x4"):
      reward_scaling_factor = 2800.0 * 255.0 * 40.0
    elif FLAGS.model in ("fully_connected_210x160x12", "lstm_210x160x12"):
      reward_scaling_factor = 400000.0 * 255.0 * 40.0
    error_reward = tf.nn.l2_loss(reward[:,1:,:] - output_reward)
    error_reward = tf.mul(reward_scaling_factor, error_reward)
    tf.scalar_summary('error_reward', error_reward)
   
    # either add up the two errors or train on the greator one. (play with this peice)
    error = tf.maximum(error_tf, error_xg)
    error = tf.add_n([error, error_reward])
  else:
    error = error_xg
  tf.scalar_summary('error', error)
  error.set_shape([])
  tf.add_to_collection('losses', error)
  return tf.add_n(tf.get_collection('losses'), name='total_loss')

def train(total_loss, lr):
   train_op = tf.train.AdamOptimizer(lr).minimize(total_loss)
   return train_op

