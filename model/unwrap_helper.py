
"""helper functions to unrap the network for training.
"""
import tensorflow as tf
import numpy as np
import ring_net 
import architecture

FLAGS = tf.app.flags.FLAGS

def lstm_unwrap(state, action, keep_prob_encoding, keep_prob_lstm, seq_length, train_piece, return_hidden):
 
  # first run  
  output_f = []
  y_0 = ring_net.encoding(state[:, 0, :, :, :],keep_prob_encoding) 
  output_f.append(y_0)

  output_t = []
  output_reward = []
  y_1, reward_1, hidden = ring_net.lstm_compression(y_0, action[:, 0, :], None, keep_prob_lstm)
  output_t.append(y_1)
  output_reward.append(reward_1)

  output_g = []
  x_1 = ring_net.decoding(y_1)
  output_g.append(x_1)
  if FLAGS.model != 'lstm_84x84x1':
    tf.image_summary('images_encode_1', x_1[:,:,:,0:3])
  else:
    tf.image_summary('images_encode_1', x_1[:,:,:,:])

  # set reuse to true
  tf.get_variable_scope().reuse_variables()

  # first get encoding
  for i in xrange(seq_length-1):
    # encode
    y_i = ring_net.encoding(state[:,i+1,:,:,:], keep_prob_encoding)
    output_f.append(y_i)
  
    # compress
    if i < 10 or FLAGS.nstep != 1: # possibly increase to 8
      y_1, reward_1, hidden = ring_net.lstm_compression(y_i, action[:, i+1, :], hidden, keep_prob_lstm)
    else:
      y_1, reward_1, hidden = ring_net.lstm_compression(y_1, action[:, i+1, :], hidden, keep_prob_lstm)
    
    output_t.append(y_1)
    output_reward.append(reward_1)
 
    # decode
    x_i_plus = ring_net.decoding(y_1)
    output_g.append(x_i_plus)
    if FLAGS.model != 'lstm_84x84x1':
      tf.image_summary('images_encoding_' + str(i+2), x_i_plus[:,:,:,0:3])
    else:
      tf.image_summary('images_encoding_' + str(i+2), x_i_plus[:,:,:,:])

  # now do the autoencoding part
  output_autoencoder = []
  for i in xrange(seq_length):
    x_i = ring_net.decoding(output_f[i])
    output_autoencoder.append(x_i)
    if FLAGS.model != 'lstm_84x84x1':
      tf.image_summary('images_autoencoding_' + str(i+2), x_i[:,:,:,0:3])
    else:
      tf.image_summary('images_autoencoding_' + str(i+2), x_i[:,:,:,:])

  # compact outputs
  # f
  output_f = tf.pack(output_f)
  output_f = tf.transpose(output_f, perm=[1,0,2])
  # t
  output_t = tf.pack(output_t)
  output_t = tf.transpose(output_t, perm=[1,0,2])
  # reward 
  output_reward = tf.pack(output_reward)
  output_reward = tf.transpose(output_reward, perm=[1,0,2])
  # g
  output_g = tf.pack(output_g)
  output_g = tf.transpose(output_g, perm=[1,0,2,3,4])
  # autoencoder
  output_autoencoder = tf.pack(output_autoencoder)
  output_autoencoder = tf.transpose(output_autoencoder, perm=[1,0,2,3,4])
 
  if return_hidden: 
    return output_f, output_t, output_g, output_reward, output_autoencoder, hidden 
  else: 
    return output_f, output_t, output_g, output_reward, output_autoencoder 

