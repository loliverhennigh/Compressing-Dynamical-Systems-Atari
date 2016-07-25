import math

import numpy as np
import tensorflow as tf
import cv2


import sys
sys.path.append('../')
import systems.cannon as cn
import systems.video as vi 

import model.ring_net as ring_net

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', '../checkpoints/ring_net_eval_store',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '../checkpoints/train_store_',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_string('video_name', 'color_video.mov',
                           """name of the video you are saving""")

assert(FLAGS.model in ("fully_connected_84x84x4", "fully_connected_84x84x3", "lstm_84x84x4", "lstm_84x84x3"), "need to use a model thats 84x84, sorry")

fourcc = cv2.cv.CV_FOURCC('m', 'p', '4', 'v') 
video = cv2.VideoWriter()
success = video.open(FLAGS.video_name, fourcc, 4, (84, 252), True)

NUM_FRAMES = 20 

def evaluate():
  """ Eval the system"""
  with tf.Graph().as_default():
    # make inputs
    x = ring_net.inputs(1, NUM_FRAMES) 
    # unwrap it
    keep_prob = tf.placeholder("float")
    output_t, output_g, output_f = ring_net.unwrap(x, keep_prob, NUM_FRAMES) 

    # restore network
    variables_to_restore = tf.all_variables()
    saver = tf.train.Saver(variables_to_restore)
    sess = tf.Session()
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir + FLAGS.model + FLAGS.system)
    if ckpt and ckpt.model_checkpoint_path:
      saver.restore(sess, ckpt.model_checkpoint_path)
      print("restored file from " + ckpt.model_checkpoint_path)
    else:
      print("no chekcpoint file found, this is an error")

    # start que runner
    tf.train.start_queue_runners(sess=sess)

    # eval ounce
    generated_seq, hidden_states, inputs = sess.run([output_g, output_f, x],feed_dict={keep_prob:1.0})
    generated_seq = generated_seq[0]
    inputs = inputs[0]
 
    # make video
    hidden_im = np.zeros((100,100,3))
    for step in xrange(NUM_FRAMES-1):
      # calc image from y_2
      new_im = np.concatenate((generated_seq[step, :, :, 0:3].squeeze()/np.amax(generated_seq[step, :, :, 0:3]), inputs[step,:,:,0:3].squeeze()/np.amax(inputs[step,:,:,0:3]), generated_seq[step, :, :, 0:3].squeeze() - inputs[step, :, :, 0:3].squeeze()), axis=0)
      new_im = np.uint8(np.abs(new_im * 255))
      video.write(new_im)
    print('saved to ' + FLAGS.video_name)
    video.release()
    cv2.destroyAllWindows()
       
def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(FLAGS.eval_dir):
    tf.gfile.DeleteRecursively(FLAGS.eval_dir)
  tf.gfile.MakeDirs(FLAGS.eval_dir)
  evaluate()


if __name__ == '__main__':
  tf.app.run()
