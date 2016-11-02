
import os
import numpy as np
import tensorflow as tf
import utils.createTFRecords as createTFRecords
from glob import glob as glb


FLAGS = tf.app.flags.FLAGS

# Constants describing the training process.
tf.app.flags.DEFINE_integer('min_queue_examples', 1000,
                           """ min examples to queue up""")

def read_data(filename_queue, seq_length, shape, num_frames, color, num_actions):
  """ reads data from tfrecord files.

  Args: 
    filename_queue: A que of strings with filenames 

  Returns:
    frames: the frame data in size (batch_size, seq_length, image height, image width, frames)
  """
  reader = tf.TFRecordReader()
  key, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(
    serialized_example,
    features={
      'state':tf.FixedLenFeature([],tf.string),
      'reward':tf.FixedLenFeature([],tf.string),
      'action':tf.FixedLenFeature([],tf.string)
    }) 
  state = tf.decode_raw(features['state'], tf.uint8)
  reward = tf.decode_raw(features['reward'], tf.uint8)
  reward = tf.reshape(reward, [seq_length, 1])
  action = tf.decode_raw(features['action'], tf.uint8)
  action = tf.reshape(action, [seq_length, num_actions])
  if color:
    state = tf.reshape(state, [seq_length, shape[0], shape[1], num_frames*3])
  else:
    state = tf.reshape(state, [seq_length, shape[0], shape[1], num_frames])
  state = tf.to_float(state) 
  reward = tf.to_float(reward) 
  action = tf.to_float(action) 
  #Display the training images in the visualizer.
  return state, reward, action

def _generate_image_label_batch(state, reward, action, batch_size, shuffle=True):
  """Construct a queued batch of images.
  Args:
    image: 4-D Tensor of [seq, height, width, frame_num] 
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.
  Returns:
    images: Images. 5D tensor of [batch_size, seq_lenght, height, width, frame_num] size.
  """

  num_preprocess_threads = 1
  if shuffle:
    #Create a queue that shuffles the examples, and then
    #read 'batch_size' images + labels from the example queue.
    states, rewards, actions = tf.train.shuffle_batch(
      [state, reward, action],
      batch_size=batch_size,
      num_threads=num_preprocess_threads,
      capacity=FLAGS.min_queue_examples + 3 * batch_size,
      min_after_dequeue=FLAGS.min_queue_examples)
  else:
     states, rewards, actions = tf.train.batch(
      [state, reward, action],
      batch_size=batch_size,
      num_threads=num_preprocess_threads,
      capacity=FLAGS.min_queue_examples + 3 * batch_size)
  return states, rewards, actions

def atari_inputs(batch_size, seq_length):
  """Construct video input for ring net. given a video_dir that contains videos this will check to see if there already exists tf recods and makes them. Then returns batchs
  Args:
    batch_size: Number of images per batch.
    seq_length: seq of inputs.
  Returns:
    images: Images. 4D tensor. Possible of size [batch_size, 84x84x4].
  """

  # get list of video file names
  if FLAGS.model in ("lstm_84x84x1"):
    shape = (84,84)
    num_frames = 4
    color = False
  elif FLAGS.model in ("lstm_210x160x3"):
    shape = (210, 160)
    num_frames = 1
    color = True

  print("begining to generate tf records")
  num_actions = createTFRecords.generate_tfrecords(seq_length, shape, num_frames, color)
 
  # get list of tfrecords 
  tfrecord_filename = glb(FLAGS.data_path + '/tfrecords/'+FLAGS.atari_game[:-4] + '/*seq_' + str(seq_length) + '_size_' + str(shape[0]) + 'x' + str(shape[1]) + 'x' + str(num_frames) + '_color_' + str(color) + '.tfrecords') 
  
  filename_queue = tf.train.string_input_producer(tfrecord_filename) 

  state, reward, action = read_data(filename_queue, seq_length, shape, num_frames, color, num_actions)
  
  states, rewards, actions, = _generate_image_label_batch(state, reward, action, batch_size)
  
  if color:
    tf.image_summary('state', states[:,0, :, :, 0:3])
  else:
    tf.image_summary('state', states[:,0, :, :, :])


  return states, rewards, actions 

