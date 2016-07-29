
"""helper functions to unrap the network for training.
"""
import tensorflow as tf
import numpy as np
import ring_net 
import architecture

FLAGS = tf.app.flags.FLAGS

def fully_connected_unwrap(state, action, keep_prob, seq_length):
  # make a list for the outputs
  output_t = []
  output_g = []
  output_reward = []
  if seq_length > 1:
    output_f = []
  else:
    output_f = None

  # first I will run once to create the graph and then set reuse to true so there is weight sharing when I roll out t
  # do f
  y_0 = ring_net.encoding(state[:, 0, :, :, :],keep_prob) 
  # do g
  x_0 = ring_net.decoding(y_0) 
  tf.image_summary('images_encoder', x_0)
  # do T' 
  y_1, reward = ring_net.compression(y_0, action[:, 0, :], keep_prob) 
  # set weight sharing   
  tf.get_variable_scope().reuse_variables()

  # append these to the lists (I dont need output f. there will be seq_length elements of output_g and seq_length-1 of output_t and output_f)
  output_g.append(x_0)
  output_t.append(y_1)
  output_reward.append(reward)

  # loop throught the seq
  for i in xrange(seq_length - 1):
    # calc f for all in seq 
    y_f_i = ring_net.encoding(state[:, i+1, :, :, :],keep_prob)
    output_f.append(y_f_i)
    # calc g for all in seq
    x_g_i = ring_net.decoding(y_1) 
    tf.image_summary('images_seq_' + str(i), x_g_i)
    output_g.append(x_g_i)
    # calc t for all in seq
    if i != (seq_length - 2):
      y_1, reward = ring_net.compression(y_1, action[:, i+1, :] ,keep_prob)
      output_t.append(y_1)
      output_reward.append(reward)
    
  # compact output_f and output_t 
  if seq_length > 1:
    output_f = tf.pack(output_f)
    output_t = tf.pack(output_t)
    output_reward = tf.pack(output_reward)
    output_reward = tf.transpose(output_reward, perm=[1,0,2])

  # compact output g
  output_g = tf.pack(output_g)
  output_g = tf.transpose(output_g, perm=[1,0,2,3,4]) # this will make it look like x (I should check to see if transpose is not flipping or doing anything funny)
  return output_t, output_g, output_f, output_reward

def lstm_unwrap(state, action, keep_prob, seq_length):
  # make a list for the outputs
  output_t = []
  output_g = [] 
  output_reward = []
  if seq_length > 1:
    output_f = []
  else:
    output_f = None

  # first I will run once to create the graph and then set reuse to true so there is weight sharing when I roll out t
  # do f
  y_0 = ring_net.encoding(state[:, 0, :, :, :],keep_prob) 
  # do g
  x_0 = ring_net.decoding(y_0) 
  tf.image_summary('images_encoder', tf.slice(x_0, [0, 0, 0, 0], [1, -1, -1, 3]))
  #tf.image_summary('images_encoder', x_0)
  # do T'
  if seq_length > 1:
    y_1, reward, hidden_state = ring_net.lstm_compression(y_0, action[:, 0, :], None, keep_prob) 
    output_t.append(y_1)
    output_reward.append(reward)
  # set weight sharing   
  tf.get_variable_scope().reuse_variables()

  # append these to the lists (I dont need output f. there will be seq_length elements of output_g and seq_length-1 of output_t and output_f)
  output_g.append(x_0)

  # loop throught the seq
  for i in xrange(seq_length - 1):
    # calc f for all in seq 
    y_f_i = ring_net.encoding(state[:, i+1, :, :, :],keep_prob)
    output_f.append(y_f_i)
    # calc g for all in seq
    x_g_i = ring_net.decoding(y_1) 
    #tf.image_summary('images_seq_' + str(i), tf.concat(3, tf.split(3, 12, x_g_i)[0:3]))
    tf.image_summary('images_seq_' + str(i) , tf.slice(x_g_i, [0, 0, 0, 0], [1, -1, -1, 3]))
    output_g.append(x_g_i)
    # calc t for all in seq
    if i != (seq_length - 2):
      y_1, reward, hidden_state = ring_net.lstm_compression(y_1, action[:, i+1, :], hidden_state, keep_prob)
      output_t.append(y_1)
      output_reward.append(reward)
    
  # compact output_f and output_t 
  if seq_length > 1:
    output_f = tf.pack(output_f)
    output_t = tf.pack(output_t)
    output_reward = tf.pack(output_reward)
    output_reward = tf.transpose(output_reward, perm=[1,0,2])

  # compact output g
  output_g = tf.pack(output_g)
  output_g = tf.transpose(output_g, perm=[1,0,2,3,4]) # this will make it look like x (I should check to see if transpose is not flipping or doing anything funny)
  return output_t, output_g, output_f, output_reward

