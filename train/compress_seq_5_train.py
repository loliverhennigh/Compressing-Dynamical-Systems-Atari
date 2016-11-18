
import os.path
import time

import numpy as np
import tensorflow as tf

import sys
sys.path.append('../')
import model.ring_net as ring_net

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '../checkpoints/train_store_',
                            """dir to store trained net""")

# save file name
SAVE_DIR = FLAGS.train_dir + '_' +  FLAGS.model + '_' + FLAGS.atari_game + '_compress_' + '_seq_length_5'
RESTORE_DIR = FLAGS.train_dir + '_' +  FLAGS.model + '_' + FLAGS.atari_game + '_compress_' + '_seq_length_3'

def train():
  """Train ring_net for a number of steps."""
  with tf.Graph().as_default():
    # make inputs
    state, reward, action = ring_net.inputs(4, 15) 

    # possible input dropout 
    input_keep_prob = tf.placeholder("float")
    state_drop = tf.nn.dropout(state, input_keep_prob)

    # possible dropout inside
    keep_prob_encoding = tf.placeholder("float")
    keep_prob_lstm = tf.placeholder("float")

    # unwrap
    x_2_o = []
    # first step
    x_2, reward_2, hidden_state = ring_net.encode_compress_decode(state[:,0,:,:,:], action[:,0,:], None, keep_prob_encoding, keep_prob_lstm)
    tf.get_variable_scope().reuse_variables()
    # unroll for 9 more steps
    for i in xrange(8):
      x_2, reward_2,  hidden_state = ring_net.encode_compress_decode(state[:,i+1,:,:,:], action[:,i+1,:], hidden_state, keep_prob_encoding, keep_prob_lstm)
    y_1 = ring_net.encoding(state[:,9,:,:,:], keep_prob_encoding)
    y_2, reward_2, hidden_state = ring_net.lstm_compression(y_1, action[:,9,:], hidden_state, keep_prob_lstm)
    x_2 = ring_net.decoding(y_2)

    x_2_o.append(x_2)
    # now collect values
    for i in xrange(4):
      y_2, reward_2, hidden_state = ring_net.lstm_compression(y_2, action[:,i+10,:], hidden_state, keep_prob_lstm)
      x_2 = ring_net.decoding(y_2)
      x_2_o.append(x_2)
      tf.image_summary('images_gen_' + str(i), x_2)
    x_2_o = tf.pack(x_2_o)
    x_2_o = tf.transpose(x_2_o, perm=[1,0,2,3,4])

    # error
    error = tf.nn.l2_loss(state[:,10:15,:,:,:] - x_2_o)
    tf.scalar_summary('loss', error)

    # train (hopefuly)
    train_op = ring_net.train(error, 1e-5)
    
    # List of all Variables
    variables = tf.all_variables()

    # Build a saver
    saver = tf.train.Saver(tf.all_variables(), max_to_keep=1)   

    # Summary op
    summary_op = tf.merge_all_summaries()
 
    # Start running operations on the Graph.
    sess = tf.Session()

    # init from seq 1 model
    print("init from " + RESTORE_DIR)
    saver_restore = tf.train.Saver(variables)
    ckpt = tf.train.get_checkpoint_state(RESTORE_DIR)
    saver_restore.restore(sess, ckpt.model_checkpoint_path)

    # Start que runner
    tf.train.start_queue_runners(sess=sess)

    # Summary op
    graph_def = sess.graph.as_graph_def(add_shapes=True)
    summary_writer = tf.train.SummaryWriter(SAVE_DIR, graph_def=graph_def)

    for step in xrange(100000):
      t = time.time()
      _ , loss_value = sess.run([train_op, error],feed_dict={keep_prob_encoding:1.0, keep_prob_lstm:1.0, input_keep_prob:1.0})
      elapsed = time.time() - t

      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      if step%100 == 0:
        print("loss value at " + str(loss_value))
        print("time per batch is " + str(elapsed))
        summary_str = sess.run(summary_op, feed_dict={keep_prob_encoding:1.0, keep_prob_lstm:1.0, input_keep_prob:1.0})
        summary_writer.add_summary(summary_str, step) 

      if step%1000 == 0:
        checkpoint_path = os.path.join(SAVE_DIR, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)  
        print("saved to " + SAVE_DIR)

def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(SAVE_DIR):
    tf.gfile.DeleteRecursively(SAVE_DIR)
  tf.gfile.MakeDirs(SAVE_DIR)
  train()

if __name__ == '__main__':
  tf.app.run()
