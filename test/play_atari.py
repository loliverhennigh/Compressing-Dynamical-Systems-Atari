import math

import numpy as np
import tensorflow as tf
import cv2

import sys
sys.path.append('../')
import systems.cannon as cn
import systems.video as vi 

import model.ring_net as ring_net
import model.unwrap_helper_test as unwrap_helper_test 
import random

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', '../checkpoints/ring_net_eval_store',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '../checkpoints/train_store_',
                           """Directory where to read model checkpoints.""")



shape = (84,84)
frame_num = 4
color = False

def random_action(num_actions):
  random_action = np.zeros((1, num_actions))
  action_ind = random.randint(0, num_actions-1)
  random_action[0, action_ind] = 1
  return random_action

def evaluate():
  """ Eval the system"""
  with tf.Graph().as_default():
    # make inputs
    image = tf.placeholder(tf.uint8, (1, 1, shape[0], shape[1], frame_num))
    action = tf.placeholder(tf.float32, (1, 6))
    x = tf.to_float(image)
    #x = tf.div(x, 255.0)
    # unwrap it
    keep_prob = tf.placeholder("float")

    y_0 = unwrap_helper_test.encoding(x, keep_prob)

    x_1, y_1, reward_1, hidden_state_1 = unwrap_helper_test.lstm_step(y_0,action, None,  keep_prob)

    # set reuse to true 
    tf.get_variable_scope().reuse_variables()

    x_2, y_2, reward_2, hidden_state_2 = unwrap_helper_test.lstm_step(y_1, action, hidden_state_1,  keep_prob)

    # restore network
    variables_to_restore = tf.all_variables()
    saver = tf.train.Saver(variables_to_restore)
    sess = tf.Session()
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir + FLAGS.model + FLAGS.atari_game)
    #ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      saver.restore(sess, ckpt.model_checkpoint_path)
      print("restored file from " + ckpt.model_checkpoint_path)
    else:
      print("no chekcpoint file found from " + FLAGS.checkpoint_dir + FLAGS.model + FLAGS.atari_game + ", this is an error")

    # get frame 
    start_frame = np.zeros((1, 1, shape[0], shape[1], frame_num))
    play_action = random_action(6)

    # eval ounce
    generated_t_x_1, generated_t_y_1, generated_t_reward_1, generated_t_hidden_state_1 = sess.run([x_1, y_1, reward_1, hidden_state_1],feed_dict={keep_prob:1.0, image:start_frame, action:play_action})
    
    # Play!!!! 
    for step in xrange(10000):
      # calc generated frame from t
      play_action = random_action(6)
      generated_t_x_1, generated_t_y_1, generated_t_hidden_state_1 = sess.run([x_2, y_2, hidden_state_2],feed_dict={keep_prob:1.0, y_1:generated_t_y_1, hidden_state_1:generated_t_hidden_state_1, action:play_action})
      frame = np.uint8(np.maximum(0, generated_t_x_1))
      frame = frame[0, :, :, 0:3]
      frame = cv2.resize(frame, (500, 500))
      cv2.imshow('frame', frame)
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
