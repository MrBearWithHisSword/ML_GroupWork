import tensorflow as tf
# from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step

from tensorflow.python.platform import tf_logging as logging
from inception_resnet_v2 import inception_resnet_v2, inception_resnet_v2_arg_scope
import os
import time
from tqdm import tqdm
from data_preprocessing import get_split
from hyperparams import Hyperparameters
slim = tf.contrib.slim

#================Dataset Information==============
data_dir = './data/'

# State where your log file is at.
log_dir = Hyperparameters.log_dir

# State where your checkpoint file is.(for transfer learning)
checkpoint_file = Hyperparameters.checkpoint_file


# Create item descriptions
items_to_descriptions = {
    'anchor': 'A 3-channel RGB colored face image.',
    'img': 'Another 3-channel RGB colored face image.',
    'label': '这两张人脸是否属于同一个人，若是则为1，否则为0'
}

#===============Data Loading====================


def distance(E1, E2):
    dis = tf.reduce_sum((E1-E2)**2, -1, keepdims=True)
    return dis

def run():
    # Create the log diretory.
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    # Training Process
    # Build Graph
    with tf.Graph().as_default() as graph:
        # Log
        tf.logging.set_verbosity(tf.logging.INFO)
        # Dataset
        dataset = get_split(Hyperparameters.tfrecord_dir, split_name='train')
        dataset = dataset.shuffle(buffer_size=1000, seed=Hyperparameters.random_seed)
        dataset = dataset.repeat(Hyperparameters.epoch_num)
        dataset = dataset.batch(Hyperparameters.batch_size)
        dataset = dataset.prefetch(buffer_size=Hyperparameters.prefetch_buffer_size)
        iterator = dataset.make_initializable_iterator()
        batch_of_pairs = iterator.get_next()
        # step num
        num_steps_per_epoch = Hyperparameters.num_batchs_per_train_epoch
        decay_steps = Hyperparameters.num_epochs_before_decay * num_steps_per_epoch
        # Create the model inference
        with slim.arg_scope(inception_resnet_v2_arg_scope()):
            _, end_points = inception_resnet_v2(batch_of_pairs['anchor_raw'],
                                                num_classes=None,
                                                create_aux_logits=False,
                                                reuse=tf.AUTO_REUSE,
                                                is_training=True)
            E1 = end_points['global_pool']
            E1 = slim.flatten(E1)
            E1 = slim.dropout(E1, 0.8, is_training=True, scope='Dropout')
            _, end_points = inception_resnet_v2(batch_of_pairs['img_raw'],
                                                num_classes=None,
                                                create_aux_logits=False,
                                                reuse=True,
                                                is_training=True)
            E2 = end_points['global_pool']
            E2 = slim.flatten(E2)
            E2 = slim.dropout(E2, 0.8, is_training=True, scope='Dropout')
            dis = distance(E1, E2)
            logits = slim.fully_connected(dis, 2, activation_fn=None, scope='Logits')
            predictions = tf.argmax(tf.nn.softmax(logits, name='Predictions'), 1)
        # Define scope for restoration
        exclude = ['InceptionResnetV2/Logits', 'InceptionResnetV2/AuxLogits', 'InceptionResnetV2/Predictions',
                   'Logits/biases', 'Logits/weights']
        variables_to_restore = slim.get_variables_to_restore(exclude=exclude)
        # Perform one-hot-encoding of the labels
        one_hot_labels = slim.one_hot_encoding(batch_of_pairs['label'], 2)
        loss = tf.losses.softmax_cross_entropy(onehot_labels=one_hot_labels, logits=logits)
        total_loss = tf.losses.get_total_loss()  # obtain the regularization losses as well
        # Create the global step for monitoring the learning_rate and training.
        # global_step = get_or_create_global_step()
        global_step = tf.train.get_or_create_global_step()
        # Define your exponentially decaying learning rate
        lr = tf.train.exponential_decay(
            learning_rate=Hyperparameters.initial_learning_rate,
            global_step=global_step,
            decay_steps=decay_steps,
            decay_rate=Hyperparameters.learning_rate_decay_factor,
            staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        # Create the train_op.
        train_op = slim.learning.create_train_op(total_loss, optimizer)
        # State the metrics that you want to predict. We get a predictions that is not one_hot_encoded.
        accuracy, accuracy_update = tf.contrib.metrics.streaming_accuracy(predictions, batch_of_pairs['label'])
        metrics_op = accuracy_update
        # Now finally create all the summaries you need to monitor and group them into one summary op.
        tf.summary.scalar('losses/Total_Loss', total_loss)
        tf.summary.scalar('accuracy', accuracy)
        tf.summary.scalar('learning_rate', lr)
        my_summary_op = tf.summary.merge_all()
        def train_step(sess, train_op, global_step):
            '''
            Simply runs a session for the three arguments provided and gives a logging on the time elapsed for each global step
            '''
            #Check the time for each sess run
            start_time = time.time()
            total_loss, global_step_count, _ = sess.run([train_op, global_step, metrics_op])
            time_elapsed = time.time() - start_time

            #Run the logging to print some results
            logging.info('global step %s: loss: %.4f (%.2f sec/step)', global_step_count, total_loss, time_elapsed)

            return total_loss, global_step_count

        # Now we create a saver function that actually restores the variables from a checkpoint file in a sess
        saver = tf.train.Saver(variables_to_restore)

        def restore_fn(sess):
            return saver.restore(sess, checkpoint_file)

        # Define your supervisor for running a managed session. Do not run the summary_op automatically or else it will consume too much memory
        sv = tf.train.Supervisor(logdir=log_dir, summary_op=None, init_fn=restore_fn)

        # Run the managed session
        with sv.managed_session() as sess:
            # for epoch in tqdm(range(Hyperparameters.epoch_num)):
            for epoch in tqdm(range(1)):
                # At the start of every epoch, show the vital information:
                sess.run(iterator.initializer)
                logging.info('Epoch %s/%s', epoch, Hyperparameters.epoch_num)
                learning_rate_value, accuracy_value = sess.run([lr, accuracy])
                logging.info('Current Learning Rate: %s', learning_rate_value)
                logging.info('Current Streaming Accuracy: %s', accuracy_value)
                # Log the summaries every 10 step.
                # for step in tqdm(range(Hyperparameters.num_batchs_per_train_epoch)):
                for step in tqdm(range(1)):
                    if step % 10 == 0:
                        loss, _ = train_step(sess, train_op, sv.global_step)
                        summaries = sess.run(my_summary_op)
                        sv.summary_computed(sess, summaries)
                    # If not, simply run the training step
                    else:
                        loss, _ = train_step(sess, train_op, sv.global_step)
            # We log the final training loss and accuracy
            logging.info('Final Loss: %s', loss)
            logging.info('Final Accuracy: %s', sess.run(accuracy))

            # Once all the training has been done, save the log files and checkpoint model
            logging.info('Finished training! Saving model to disk now.')
            # saver.save(sess, "./flowers_model.ckpt")
            sv.saver.save(sess, sv.save_path, global_step=sv.global_step)


if __name__ == '__main__':
    run()
