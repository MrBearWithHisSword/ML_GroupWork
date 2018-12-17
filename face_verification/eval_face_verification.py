import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging
# from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
from inception_resnet_v2 import inception_resnet_v2, inception_resnet_v2_arg_scope
from hyperparams import Hyperparameters
from tqdm import tqdm
import time
import os
from data_preprocessing import *
import matplotlib.pyplot as plt
plt.style.use('ggplot')
slim = tf.contrib.slim

# State where the log dir is.
log_dir = Hyperparameters.log_dir

# Create evaluation log directory to visualize the validation process
log_eval = Hyperparameters.log_eval

# State where the validation data is.
data_dir = Hyperparameters.tfrecord_dir

# Get the latest_checkpoint file
checkpoint_file = tf.train.latest_checkpoint(log_dir)


def distance(E1, E2):
    dis = tf.reduce_sum((E1-E2)**2, -1, keepdims=True)
    return dis


def run():
    # Create log_dir for evaluation information
    if not os.path.exists(log_eval):
        os.makedirs(log_eval)

    # Build Graph
    with tf.Graph().as_default():
        tf.logging.set_verbosity(tf.logging.INFO)
        # Get the dataset
        dataset = get_split(Hyperparameters.tfrecord_dir, split_name='train')
        dataset = dataset.shuffle(buffer_size=1000, seed=Hyperparameters.random_seed)
        dataset = dataset.repeat(Hyperparameters.epoch_num)
        dataset = dataset.batch(Hyperparameters.batch_size)
        dataset = dataset.prefetch(buffer_size=Hyperparameters.prefetch_buffer_size)
        iterator = dataset.make_initializable_iterator()
        batch_of_pairs = iterator.get_next()
        # steps per epoch
        num_steps_per_epoch = Hyperparameters.num_batchs_per_eval_epoch
        # Create inference model.
        with slim.arg_scope(inception_resnet_v2_arg_scope()):
            _, end_points = inception_resnet_v2(batch_of_pairs['anchor_raw'],
                                                num_classes=None,
                                                create_aux_logits=False,
                                                reuse=tf.AUTO_REUSE,
                                                is_training=False)
            E1 = end_points['global_pool']
            E1 = slim.flatten(E1)
            E1 = slim.dropout(E1, 0.8, is_training=True, scope='Dropout')
            _, end_points = inception_resnet_v2(batch_of_pairs['img_raw'],
                                                num_classes=None,
                                                create_aux_logits=False,
                                                reuse=True,
                                                is_training=False)
            E2 = end_points['global_pool']
            E2 = slim.flatten(E2)
            E2 = slim.dropout(E2, 0.8, is_training=False, scope='Dropout')
            dis = distance(E1, E2)
            logits = slim.fully_connected(dis, 2, activation_fn=None, scope='Logits')
            predictions = tf.argmax(tf.nn.softmax(logits, name='Predictions'), 1)
        # Get all the variables to restore from the checkpoint file and create the saver function to restore
        variables_to_restore = slim.get_variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        def restore_fn(sess):
            return saver.restore(sess, checkpoint_file)
        # Define the metrics to track
        accuracy, accuracy_update = tf.contrib.metrics.streaming_accuracy(predictions, batch_of_pairs['label'])
        metrics_op = tf.group(accuracy_update)
        # Create the global step and an increment_op for monitoring
        global_step = tf.train.get_or_create_global_step()
        global_step_op = tf.assign(global_step, global_step + 1)   # no apply_gradient method so manually increasing the global_step
        # Create a evaluation step function
        def eval_step(sess, metrics_op, global_step):
            '''
            Simply takes in a session, runs the metrics op and some logging information.
            '''
            start_time = time.time()
            _, global_step_count, accuracy_value = sess.run([metrics_op, global_step_op, accuracy])
            time_elapsed = time.time() - start_time

            # Log some information
            logging.info('Global Step %s: Streaming Accuracy: %.4f (%.2f sec/step)', global_step_count, accuracy_value,
                         time_elapsed)

            return accuracy_value
        # Define some scalar quantities to monitor
        tf.summary.scalar('Validation_Accuracy', accuracy)
        my_summary_op = tf.summary.merge_all()
        # Get  supervisor
        sv = tf.train.Supervisor(logdir=log_eval, summary_op=None, saver=None, init_fn=restore_fn)
        # Run the session
        with tf.Session() as sess:
            for epoch in tqdm(range(Hyperparameters.eval_epoch_num)):
                logging.info('Epoch: %s/%s', epoch, Hyperparameters.eval_epoch_num)
                logging.info('Current Streaming Accuracy: %.4f', sess.run(accuracy))
                for step in tqdm(range(Hyperparameters.num_batchs_per_eval_epoch)):
                    if step % 10 == 0:
                        eval_step(sess, metrics_op=metrics_op, global_step=sv.global_step)
                        summaries = sess.run(my_summary_op)
                        sv.summary_computed(sess, summaries)
                    else:
                        eval_step(sess, metrics_op=metrics_op, global_step=sv.global_step)
            # At the end of all the evaluation, show the final accuracy
            logging.info('Final Streaming Accuracy: %.4f', sess.run(accuracy))

            # # Visualize some of the last batch's images just to see what our model has predicted
            # anchors = batch_of_pairs['anchor_raw']
            # imgs = batch_of_pairs['img_raw']
            # labels = batch_of_pairs['label']
            # anchors, imgs, labels, predictions = sess.run([anchors, imgs, labels, predictions])
            # for i in range(10):
            #     anchor, img, label, prediction = anchors[i], imgs[i], labels[i], predictions[i]
            #     text = 'Prediction: %s \n Ground Truth: %s' % (prediction, label_name)
            #     # img_plot = plt.imshow(img)
            #
            #     # Set up the plot and hide axes
            #     plt.title(text)
            #     plt.imshow(img)
            #     plt.figure()
            #     plt.imshow(anchor)
            #     # img_plot.axes.get_yaxis().set_ticks([])
            #     # img_plot.axes.get_xaxis().set_ticks([])
            #     plt.show()
            #
            # logging.info(
            #     'Model evaluation has completed! Visit TensorBoard for more information regarding your evaluation.')


if __name__ == '__main__':
    run()
