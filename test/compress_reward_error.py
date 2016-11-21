
import os.path
import time

import numpy as np
import tensorflow as tf

import sys
sys.path.append('../')
import model.ring_net as ring_net
import matplotlib.pyplot as plt

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', '../checkpoints/ring_net_eval_store',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '../checkpoints/train_store_',
                           """Directory where to read model checkpoints.""")

def evaluate():
  """Train ring_net for a number of steps."""
  with tf.Graph().as_default():
    # make inputs
    state, reward, action = ring_net.inputs(1, 15) 

    # possible input dropout 
    input_keep_prob = tf.placeholder("float")
    state_drop = tf.nn.dropout(state, input_keep_prob)

    # possible dropout inside
    keep_prob_encoding = tf.placeholder("float")
    keep_prob_lstm = tf.placeholder("float")

    # unwrap
    reward_2_o = []
    # first step
    x_2, reward_2, hidden_state = ring_net.encode_compress_decode(state[:,0,:,:,:], action[:,1,:], None, keep_prob_encoding, keep_prob_lstm)
    tf.get_variable_scope().reuse_variables()
    # unroll for 9 more steps
    for i in xrange(8):
      x_2, reward_2,  hidden_state = ring_net.encode_compress_decode(state[:,i+1,:,:,:], action[:,i+2,:], hidden_state, keep_prob_encoding, keep_prob_lstm)
    y_1 = ring_net.encoding(state[:,9,:,:,:], keep_prob_encoding)
    y_2, reward_2, hidden_state = ring_net.lstm_compression(y_1, action[:,10,:], hidden_state, keep_prob_lstm)
    x_2 = ring_net.decoding(y_2)

    reward_2_o.append(reward_2)
    # now collect values
    for i in xrange(4):
      y_2, reward_2, hidden_state = ring_net.lstm_compression(y_2, action[:,i+11,:], hidden_state, keep_prob_lstm)
      x_2 = ring_net.decoding(y_2)
      reward_2_o.append(reward_2)
    reward_2_o = tf.pack(reward_2_o)
    reward_2_o = tf.transpose(reward_2_o, perm=[1,0,2])

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
      print("no chekcpoint file found from " + FLAGS.checkpoint_dir + ", this is an error")

    # Start que runner
    tf.train.start_queue_runners(sess=sess)

    reward_g_o, reward_o = sess.run([reward_2_o, reward],feed_dict={keep_prob_encoding:1.0, keep_prob_lstm:1.0, input_keep_prob:1.0})

    print(reward_g_o.shape)
    print(reward_o.shape)

    plt.figure(0)
    plt.plot(reward_g_o[0,:,0], label= "predicted reward") 
    plt.plot(reward_o[0,10:,0], label= "reward") 
    plt.title("reward vs step")
    plt.xlabel("step")
    plt.ylabel("reward")
    plt.legend()
    plt.savefig("compress_reward.png")


def main(argv=None):  # pylint: disable=unused-argument
  evaluate()

if __name__ == '__main__':
  tf.app.run()
