
import os
import numpy as np
import tensorflow as tf
import utils.createTFRecords as createTFRecords
import systems.cannon as cannon 
import systems.cannon_createTFRecords as cannon_createTFRecords
from glob import glob as glb


FLAGS = tf.app.flags.FLAGS

# Constants describing the training process.
tf.app.flags.DEFINE_integer('min_queue_examples', 1000,
                           """ min examples to queue up""")

def read_data(filename_queue, seq_length, shape, num_frames, color):
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
      'image':tf.FixedLenFeature([],tf.string)
    }) 
  image = tf.decode_raw(features['image'], tf.uint8)
  if color:
    image = tf.reshape(image, [seq_length, shape[0], shape[1], num_frames*3])
  else:
    image = tf.reshape(image, [seq_length, shape[0], shape[1], num_frames])
  image = tf.to_float(image) 
  #Display the training images in the visualizer.
  return image

def _generate_image_label_batch(image, batch_size, shuffle=True):
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
    frames = tf.train.shuffle_batch(
      [image],
      batch_size=batch_size,
      num_threads=num_preprocess_threads,
      capacity=FLAGS.min_queue_examples + 3 * batch_size,
      min_after_dequeue=FLAGS.min_queue_examples)
  else:
     frames = tf.train.batch(
      [image],
      batch_size=batch_size,
      num_threads=num_preprocess_threads,
      capacity=FLAGS.min_queue_examples + 3 * batch_size)
  return frames

def video_inputs(batch_size, seq_length):
  """Construct video input for ring net. given a video_dir that contains videos this will check to see if there already exists tf recods and makes them. Then returns batchs
  Args:
    batch_size: Number of images per batch.
    seq_length: seq of inputs.
  Returns:
    images: Images. 4D tensor. Possible of size [batch_size, 84x84x4].
  """

  # get list of video file names
  video_filename = glb('../data/videos/'+FLAGS.video_dir+'/*') 

  if FLAGS.model in ("fully_connected_84x84x4", "lstm_84x84x4"):
    shape = (84,84)
    num_frames = 4
    color = False
  if FLAGS.model in ("fully_connected_84x84x3", "lstm_84x84x3"):
    shape = (84, 84)
    num_frames = 1 
    color = True

  print("begining to generate tf records")
  for f in video_filename:
    createTFRecords.generate_tfrecords(f, seq_length, shape, num_frames, color)
 
  # get list of tfrecords 
  tfrecord_filename = glb('../data/tfrecords/'+FLAGS.video_dir+'/*seq_' + str(seq_length) + '_size_' + str(shape[0]) + 'x' + str(shape[1]) + 'x' + str(num_frames) + '_color_' + str(color) + '.tfrecords') 
  
  
  filename_queue = tf.train.string_input_producer(tfrecord_filename) 

  image = read_data(filename_queue, seq_length, shape, num_frames, color)
  
  if color:
    display_image = tf.split(3, 3, image)
    tf.image_summary('images', display_image[0])
  else:
    tf.image_summary('images', image)

  image = tf.div(image, 255.0) 

  frames = _generate_image_label_batch(image, batch_size)
 
  return frames 

def cannon_inputs(batch_size, seq_length):
  """Construct cannon input for ring net. just a 28x28 frame video of a bouncing ball 
  Args:
    batch_size: Number of images per batch.
    seq_length: seq of inputs.
  Returns:
    images: Images. 4D tensor. Possible of size [batch_size, 28x28x4].
  """
  num_samples = 100000
  
  cannon_createTFRecords.generate_tfrecords(num_samples, seq_length)
 
  tfrecord_filename = glb('../data/tfrecords/cannon/*num_samples_' + str(num_samples) + '_seq_length_' + str(seq_length) + '.tfrecords') 
  
  filename_queue = tf.train.string_input_producer(tfrecord_filename) 

  image = read_data(filename_queue, seq_length, (28, 28), 4, False)
  tf.image_summary('images', image)
  
  frames = _generate_image_label_batch(image, batch_size)

  return frames

