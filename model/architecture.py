
"""functions used to construct different architectures  
"""

import tensorflow as tf
import numpy as np

FLAGS = tf.app.flags.FLAGS

def _activation_summary(x):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measure the sparsity of activations.

  Args:
    x: Tensor
  Returns:
    nothing
  """
  tensor_name = x.op.name
  tf.histogram_summary(tensor_name + '/activations', x)
  tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer)
  return var


def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  var = _variable_on_cpu(name, shape,
                         tf.truncated_normal_initializer(stddev=stddev))
  if wd:
    weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
    weight_decay.set_shape([])
    tf.add_to_collection('losses', weight_decay)
  return var

def _conv_layer(inputs, kernel_size, stride, num_features, idx):
  with tf.variable_scope('{0}_conv'.format(idx)) as scope:
    input_channels = inputs.get_shape()[3]

    weights = _variable_with_weight_decay('weights', shape=[kernel_size,kernel_size,input_channels,num_features],stddev=0.1, wd=FLAGS.weight_decay)
    biases = _variable_on_cpu('biases',[num_features],tf.constant_initializer(0.1))

    conv = tf.nn.conv2d(inputs, weights, strides=[1, stride, stride, 1], padding='SAME')
    conv_biased = tf.nn.bias_add(conv, biases)
    #Leaky ReLU
    conv_rect = tf.maximum(FLAGS.alpha*conv_biased,conv_biased,name='{0}_conv'.format(idx))
    return conv_rect

def _transpose_conv_layer(inputs, kernel_size, stride, num_features, idx):
  with tf.variable_scope('{0}_trans_conv'.format(idx)) as scope:
    input_channels = inputs.get_shape()[3]
    
    weights = _variable_with_weight_decay('weights', shape=[kernel_size,kernel_size,num_features,input_channels], stddev=0.1, wd=FLAGS.weight_decay)
    biases = _variable_on_cpu('biases',[num_features],tf.constant_initializer(0.1))
    batch_size = tf.shape(inputs)[0]
    output_shape = tf.pack([tf.shape(inputs)[0], tf.shape(inputs)[1]*stride, tf.shape(inputs)[2]*stride, num_features]) 
    conv = tf.nn.conv2d_transpose(inputs, weights, output_shape, strides=[1,stride,stride,1], padding='SAME')
    conv_biased = tf.nn.bias_add(conv, biases)
    #Leaky ReLU
    conv_rect = tf.maximum(FLAGS.alpha*conv_biased,conv_biased,name='{0}_transpose_conv'.format(idx))
    return conv_rect
     

def _fc_layer(inputs, hiddens, idx, flat = False, linear = False):
  with tf.variable_scope('{0}_fc'.format(idx)) as scope:
    input_shape = inputs.get_shape().as_list()
    if flat:
      dim = input_shape[1]*input_shape[2]*input_shape[3]
      inputs_processed = tf.reshape(inputs, [-1,dim])
    else:
      dim = input_shape[1]
      inputs_processed = inputs
    
    weights = _variable_with_weight_decay('weights', shape=[dim,hiddens],stddev=0.01, wd=FLAGS.weight_decay)
    biases = _variable_on_cpu('biases', [hiddens], tf.constant_initializer(0.01))
    if linear:
      return tf.add(tf.matmul(inputs_processed,weights),biases,name=str(idx)+'_fc')
  
    ip = tf.add(tf.matmul(inputs_processed,weights),biases)
    return tf.maximum(FLAGS.alpha*ip,ip,name=str(idx)+'_fc')

def encoding_28x28x4(inputs, keep_prob):
  """Builds encoding part of ring net.
  Args:
    inputs: input to encoder
    keep_prob: dropout layer
  """
  #--------- Making the net -----------
  # x_1 -> y_1 -> y_2 -> x_2
  # this peice x_1 -> y_1
  x_1_image = inputs 
 
  # conv1
  conv1 = _conv_layer(x_1_image, 5, 1, 32, "encode_1")
  # conv2
  conv2 = _conv_layer(conv1, 2, 2, 32, "encode_2")
  # conv3
  conv3 = _conv_layer(conv2, 5, 1, 64, "encode_3")
  # conv4
  conv4 = _conv_layer(conv3, 2, 2, 64, "encode_4")
  # fc5 
  fc5 = _fc_layer(conv4, 512, "encode_5", True, False)
  # dropout maybe
  fc5_dropout = tf.nn.dropout(fc5, keep_prob)
  # y_1 
  y_1 = _fc_layer(fc5_dropout, 64, "encode_6", False, False)
  _activation_summary(y_1)

  return y_1 

def encoding_84x84x4(inputs, keep_prob):
  """Builds encoding part of ring net. 
  Args:
    inputs: input to encoder
    keep_prob: dropout layer
  """
  #--------- Making the net -----------
  # x_1 -> y_1 -> y_2 -> x_2
  # this peice x_1 -> y_1
  x_1_image = inputs 
 
  # conv1
  conv1 = _conv_layer(x_1_image, 8, 3, 32, "encode_1")
  # conv2
  conv2 = _conv_layer(conv1, 4, 2, 64, "encode_2")
  # conv3
  conv3 = _conv_layer(conv2, 3, 1, 128, "encode_3")
  # conv4
  conv4 = _conv_layer(conv3, 1, 1, 64, "encode_4")
  # conv5
  conv5 = _conv_layer(conv4, 3, 1, 128, "encode_5")
  # conv6
  conv6 = _conv_layer(conv5, 1, 1, 64, "encode_6")
  # fc5 
  fc5 = _fc_layer(conv4, 1024, "encode_7", True, False)
  # dropout maybe
  fc5_dropout = tf.nn.dropout(fc5, keep_prob)
  # y_1 
  y_1 = _fc_layer(fc5_dropout, 512, "encode_8", False, False)
  _activation_summary(y_1)

  return y_1 

def encoding_84x84x3(inputs, keep_prob):
  """Builds encoding part of ring net.
  Args:
    inputs: input to encoder
    keep_prob: dropout layer
  """
   #--------- Making the net -----------
  # x_1 -> y_1 -> y_2 -> x_2
  # this peice x_1 -> y_1
  x_1_image = inputs 
 
  # conv1
  conv1 = _conv_layer(x_1_image, 8, 3, 32, "encode_1")
  # conv2
  conv2 = _conv_layer(conv1, 4, 2, 64, "encode_2")
  # conv3
  conv3 = _conv_layer(conv2, 3, 1, 128, "encode_3")
  # conv4
  conv4 = _conv_layer(conv3, 1, 1, 64, "encode_4")
  # conv5
  conv5 = _conv_layer(conv4, 3, 1, 128, "encode_5")
  # conv6
  conv6 = _conv_layer(conv5, 1, 1, 64, "encode_6")
  # fc5 
  fc7 = _fc_layer(conv6, 1024, "encode_7", True, False)
  # dropout maybe
  # y_1 
  y_1 = tf.nn.dropout(fc7, keep_prob)
  _activation_summary(y_1)

  return y_1 

def compression_28x28x4(inputs, keep_prob):
  """Builds compressed dynamical system part of the net.
  Args:
    inputs: input to system
  """
  #--------- Making the net -----------
  # x_1 -> y_1 -> y_2 -> x_2
  # this peice y_1 -> y_2
  y_1 = inputs 
 
  # (start indexing at 10) -- I will change this in a bit
  # fc11
  fc11 = _fc_layer(y_1, 512, "compress_11", False, False)
  # fc12
  fc12 = _fc_layer(fc11, 512, "compress_12", False, False)
  # dropout maybe
  fc12_dropout = tf.nn.dropout(fc12, keep_prob)
  # y_2 
  y_2 = _fc_layer(fc12_dropout, 64, "compress_13", False, False)
  _activation_summary(y_2)

  return y_2 

def compression_84x84x4(inputs, keep_prob):
  """Builds compressed dynamical system part of the net.
  Args:
    inputs: input to system
  """
  #--------- Making the net -----------
  # x_1 -> y_1 -> y_2 -> x_2
  # this peice y_1 -> y_2
  y_1 = inputs 
 
  # (start indexing at 10) -- I will change this in a bit
  # fc11
  fc11 = _fc_layer(y_1, 512, "compress_11", False, False)
  # fc12
  fc12 = _fc_layer(fc11, 512, "compress_12", False, False)
  # fc13
  fc13 = _fc_layer(fc12, 1024, "compress_13", False, False)
  # dropout maybe
  fc13_dropout = tf.nn.dropout(fc13, keep_prob)
  # y_2 
  y_2 = _fc_layer(fc13_dropout, 512, "compress_14", False, False)

  return y_2 

def compression_84x84x3(inputs, keep_prob):
  """Builds compressed dynamical system part of the net.
  Args:
    inputs: input to system
  """
  #--------- Making the net -----------
  # x_1 -> y_1 -> y_2 -> x_2
  # this peice y_1 -> y_2
  y_1 = inputs 
 
  # (start indexing at 10) -- I will change this in a bit
  # fc11
  fc11 = _fc_layer(y_1, 1024, "compress_11", False, False)
  # fc12
  fc12 = _fc_layer(fc11, 1024, "compress_12", False, False)
  # fc13
  fc13 = _fc_layer(fc12, 2048, "compress_13", False, False)
  # dropout maybe
  fc13_dropout = tf.nn.dropout(fc13, keep_prob)
  # y_2 
  y_2 = _fc_layer(fc13_dropout, 1024, "compress_14", False, False)

  return y_2 

def lstm_compression_28x28x4(inputs, hidden_state, keep_prob):
  """Builds compressed dynamical system part of the net.
  Args:
    inputs: input to system
  """
  #--------- Making the net -----------
  # x_1 -> y_1 -> y_2 -> x_2
  # this peice y_1 -> y_2
  num_layers = 2 

  y_1 = inputs

  with tf.variable_scope("compress_LSTM", initializer = tf.random_uniform_initializer(-0.01, 0.01)):
    with tf.device('/cpu:0'):
      lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(64, forget_bias=1.0)
      lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)
      cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * num_layers)
      if hidden_state == None:
        batch_size = inputs.get_shape()[0]
        hidden_state = cell.zero_state(batch_size, tf.float32)

  y2, new_state = cell(y_1, hidden_state)
  
  return y2, new_state

def lstm_compression_84x84x4(inputs, hidden_state, keep_prob):
  """Builds compressed dynamical system part of the net.
  Args:
    inputs: input to system
  """
  #--------- Making the net -----------
  # x_1 -> y_1 -> y_2 -> x_2
  # this peice y_1 -> y_2
  num_layers = 2 

  y_1 = inputs

  with tf.variable_scope("compress_LSTM", initializer = tf.random_uniform_initializer(-0.01, 0.01)):
    with tf.device('/cpu:0'):
      lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(512, forget_bias=1.0)
      lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)
      cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * num_layers)
      if hidden_state == None:
        batch_size = inputs.get_shape()[0]
        hidden_state = cell.zero_state(batch_size, tf.float32)

  y2, new_state = cell(y_1, hidden_state)
  
  return y2, new_state

def lstm_compression_84x84x3(inputs, hidden_state, keep_prob):
  """Builds compressed dynamical system part of the net.
  Args:
    inputs: input to system
  """
  #--------- Making the net -----------
  # x_1 -> y_1 -> y_2 -> x_2
  # this peice y_1 -> y_2
  num_layers = 3

  y_1 = inputs

  with tf.variable_scope("compress_LSTM", initializer = tf.random_uniform_initializer(-0.01, 0.01)):
    with tf.device('/cpu:0'):
      lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(1024, forget_bias=1.0)
      lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)
      cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * num_layers)
      if hidden_state == None:
        batch_size = inputs.get_shape()[0]
        hidden_state = cell.zero_state(batch_size, tf.float32)

  y2, new_state = cell(y_1, hidden_state)
  
  return y2, new_state

def decoding_28x28x4(inputs):
  """Builds decoding part of ring net.
  Args:
    inputs: input to decoder
  """
  #--------- Making the net -----------
  # x_1 -> y_1 -> y_2 -> x_2
  # this peice y_2 -> x_2
  y_2 = inputs 
 
  # fc21
  fc21 = _fc_layer(y_2, 512, "decode_21", False, False)
  # fc23
  fc22 = _fc_layer(fc21, 64*7*7, "decode_22", False, False)
  conv22 = tf.reshape(fc22, [-1, 7, 7, 64])
  # conv23
  conv23 = _transpose_conv_layer(conv22, 2, 2, 64, "decode_23")
  # conv24
  conv24 = _transpose_conv_layer(conv23, 5, 1, 32, "decode_24")
  # conv25
  conv25 = _transpose_conv_layer(conv24, 2, 2, 32, "decode_25")
  # conv26
  conv26 = _transpose_conv_layer(conv25, 5, 1, 4, "decode_26")
  # x_2 
  x_2 = tf.reshape(conv26, [-1, 28, 28, 4])
  return x_2 

def decoding_84x84x4(inputs):
  """Builds decoding part of ring net.
  Args:
    inputs: input to decoder
  """
  #--------- Making the net -----------
  # x_1 -> y_1 -> y_2 -> x_2
  # this peice y_2 -> x_2
  y_2 = inputs 
 
  # fc21
  fc21 = _fc_layer(y_2, 64*14*14, "decode_21", False, False)
  conv22 = tf.reshape(fc21, [-1, 14, 14, 64])
  # conv23
  conv23 = _transpose_conv_layer(conv22, 1, 1, 128, "decode_23")
  # conv24
  conv24 = _transpose_conv_layer(conv23, 3, 1, 64, "decode_24")
  # conv25
  conv25 = _transpose_conv_layer(conv24, 1, 1, 128, "decode_25")
  # conv26
  conv26 = _transpose_conv_layer(conv25, 3, 1, 128, "decode_26")
  # conv25
  conv27 = _transpose_conv_layer(conv26, 4, 2, 256, "decode_27")
  # conv26
  x_2 = _transpose_conv_layer(conv27, 8, 3, 4, "decode_28")
  # x_2 
  _activation_summary(x_2)

  return x_2 

def decoding_84x84x3(inputs):
  """Builds decoding part of ring net.
  Args:
    inputs: input to decoder
  """
  #--------- Making the net -----------
  # x_1 -> y_1 -> y_2 -> x_2
  # this peice y_2 -> x_2
  y_2 = inputs 
 
  # fc21
  fc21 = _fc_layer(y_2, 64*14*14, "decode_21", False, False)
  conv22 = tf.reshape(fc21, [-1, 14, 14, 64])
  # conv23
  conv23 = _transpose_conv_layer(conv22, 1, 1, 128, "decode_23")
  # conv24
  conv24 = _transpose_conv_layer(conv23, 3, 1, 64, "decode_24")
  # conv25
  conv25 = _transpose_conv_layer(conv24, 1, 1, 128, "decode_25")
  # conv26
  conv26 = _transpose_conv_layer(conv25, 3, 1, 128, "decode_26")
  # conv25
  conv27 = _transpose_conv_layer(conv26, 4, 2, 256, "decode_27")
  # conv26
  x_2 = _transpose_conv_layer(conv27, 8, 3, 3, "decode_28")
  # x_2 
  _activation_summary(x_2)

  return x_2 

