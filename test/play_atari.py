import math

import numpy as np
import tensorflow as tf
import cv2

import sys
sys.path.append('../')

import model.ring_net as ring_net
import model.unwrap_helper_test as unwrap_helper_test 
import random
import time

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', '../checkpoints/ring_net_eval_store',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '../checkpoints/train_store_',
                           """Directory where to read model checkpoints.""")

def random_action(num_actions):
  random_action = np.zeros((1, num_actions))
  action_ind = random.randint(0, num_actions-1)
  random_action[0, action_ind] = 1
  return random_action

def evaluate():
  """ Eval the system"""
  with tf.Graph().as_default():
    # make inputs
    state_start, reward_start, action_start, = ring_net.inputs(1, 5) 
    action_size = int(action_start.get_shape()[2])
    action = tf.placeholder(tf.float32, (1, action_size))

    # unwrap it
    output_f, output_t, output_g, output_reward, output_autoencoder, hidden = ring_net.unwrap(state_start, action_start, 1.0, 1.0, 5, "all", return_hidden=True)

    # rename output_t
    y_0 = output_t[:,4,:]
    y_1, reward_1, hidden_1 = ring_net.lstm_compression(y_0, action, hidden,  1.0)
    x_1 = ring_net.decoding(y_1)

    # restore network
    variables_to_restore = tf.all_variables()
    saver = tf.train.Saver(variables_to_restore)
    sess = tf.Session()
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    #ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      saver.restore(sess, ckpt.model_checkpoint_path)
      print("restored file from " + ckpt.model_checkpoint_path)
    else:
      print("no chekcpoint file found from " + FLAGS.checkpoint_dir + FLAGS.model + FLAGS.atari_game + ", this is an error")

    # get frame
    tf.train.start_queue_runners(sess=sess)
    play_action = random_action(6)
    y_1_g, hidden_1_g = sess.run([y_1, hidden_1], feed_dict={action:play_action})

    # Play!!!! 
    for step in xrange(10000):
      print(step)
      #time.sleep(.5)
      # calc generated frame from t
      play_action = random_action(6)
      y_1_g, hidden_1_g = sess.run([y_1, hidden_1],feed_dict={y_0:y_1_g, hidden:hidden_1_g, action:play_action})
      x_1_g = sess.run(x_1,feed_dict={y_1:y_1_g})
      frame = np.uint8(np.minimum(np.maximum(0, x_1_g*255.0), 255))
      frame = frame[0, :, :, :]
      frame = cv2.resize(frame, (500, 500))
      cv2.imshow('frame', frame)
      cv2.waitKey(0)
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    cv2.destroyAllWindows()

       
def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(FLAGS.eval_dir):
    tf.gfile.DeleteRecursively(FLAGS.eval_dir)
  tf.gfile.MakeDirs(FLAGS.eval_dir)
  evaluate()


if __name__ == '__main__':
  tf.app.run()
