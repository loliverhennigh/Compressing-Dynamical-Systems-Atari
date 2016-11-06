
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

# learning curriculum
CURRICULUM_STEPS = [1000000, 150000, 150000]
CURRICULUM_SEQ = [1, 10, 15]
CURRICULUM_TRAIN_PEICE = ["autoencoder", "all", "all"]
CURRICULUM_BATCH_SIZE = [50, 8, 6]
CURRICULUM_LEARNING_RATE = [1e-4, 1e-4, 1e-4]

# save file name
SAVE_DIR = FLAGS.train_dir + '_' +  FLAGS.model + '_' + FLAGS.atari_game

def train(iteration):
  """Train ring_net for a number of steps."""
  with tf.Graph().as_default():
    # make inputs
    state, reward, action = ring_net.inputs(CURRICULUM_BATCH_SIZE[iteration], CURRICULUM_SEQ[iteration]) 

    # possible input dropout 
    input_keep_prob = tf.placeholder("float")
    state_drop = tf.nn.dropout(state, input_keep_prob)

    # possible dropout inside
    keep_prob_encoding = tf.placeholder("float")
    keep_prob_lstm = tf.placeholder("float")

    # create and unrap network
    output_f, output_t, output_g, output_reward, output_autoencoder = ring_net.unwrap(state_drop, action, keep_prob_encoding, keep_prob_lstm, CURRICULUM_SEQ[iteration], CURRICULUM_TRAIN_PEICE[iteration])

    # calc error
    error = ring_net.loss(state, reward, output_f, output_t, output_g, output_reward, output_autoencoder, CURRICULUM_TRAIN_PEICE[iteration])
    error = tf.div(error, CURRICULUM_SEQ[iteration])

    # train (hopefuly)
    train_op = ring_net.train(error, CURRICULUM_LEARNING_RATE[iteration])
    
    # List of all Variables
    variables = tf.all_variables()
    #for i, variable in enumerate(variables):
    #  print '----------------------------------------------'
    #  print variable.name[:variable.name.index(':')]

    # Build a saver
    saver = tf.train.Saver(tf.all_variables())   

    # Summary op
    summary_op = tf.merge_all_summaries()
 
    # Build an initialization operation to run below.
    if iteration == 0:
      init = tf.initialize_all_variables()

    # Start running operations on the Graph.
    sess = tf.Session()

    # init if this is the very time training
    if iteration == 0: 
      print("init network from scratch")
      sess.run(init)

    # restore if iteration is not 0
    if iteration != 0:
      variables_to_restore = tf.all_variables()
      autoencoder_variables = [variable for i, variable in enumerate(variables_to_restore) if ("compress" not in variable.name[:variable.name.index(':')]) and ("RNN" not in variable.name[:variable.name.index(':')])]
      rnn_variables = [variable for i, variable in enumerate(variables_to_restore) if ("compress" in variable.name[:variable.name.index(':')]) or ("RNN" in variable.name[:variable.name.index(':')])]
     
      ckpt = tf.train.get_checkpoint_state(SAVE_DIR)
      autoencoder_saver = tf.train.Saver(autoencoder_variables)
      print("restoring autoencoder part of network form " + ckpt.model_checkpoint_path)
      autoencoder_saver.restore(sess, ckpt.model_checkpoint_path)
      print("restored file from " + ckpt.model_checkpoint_path)

      if CURRICULUM_SEQ[iteration-1] == 1:
        print("init compression part of network from scratch")
        rnn_init = tf.initialize_variables(rnn_variables)
        sess.run(rnn_init)
      else:
        rnn_saver = tf.train.Saver(rnn_variables)
        print("restoring compression part of network")
        rnn_saver.restore(sess, ckpt.model_checkpoint_path)
        print("restored file from " + ckpt.model_checkpoint_path)
        
    # Start que runner
    tf.train.start_queue_runners(sess=sess)

    # Summary op
    graph_def = sess.graph.as_graph_def(add_shapes=True)
    summary_writer = tf.train.SummaryWriter(SAVE_DIR, graph_def=graph_def)

    for step in xrange(CURRICULUM_STEPS[iteration]):
      t = time.time()
      _ , loss_value = sess.run([train_op, error],feed_dict={keep_prob_encoding:1.0, keep_prob_lstm:1.0, input_keep_prob:.8})
      elapsed = time.time() - t

      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      if step%100 == 0:
        print("loss value at " + str(loss_value))
        print("time per batch is " + str(elapsed))
        summary_str = sess.run(summary_op, feed_dict={keep_prob_encoding:1.0, keep_prob_lstm:1.0, input_keep_prob:.8})
        summary_writer.add_summary(summary_str, step) 

      if step%1000 == 0:
        checkpoint_path = os.path.join(SAVE_DIR, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)  
        print("saved to " + SAVE_DIR)

def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(SAVE_DIR):
    tf.gfile.DeleteRecursively(SAVE_DIR)
  tf.gfile.MakeDirs(SAVE_DIR)
  for i in xrange(len(CURRICULUM_STEPS)):
    train(i)

if __name__ == '__main__':
  tf.app.run()
