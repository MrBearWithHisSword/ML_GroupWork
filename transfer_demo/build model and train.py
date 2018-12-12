import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
from tensorflow.contrib import slim



def regression_model(inputs, is_training=True, scope='deep_regression'):
    with tf.variable_scope(scope, 'deep_regression', [inputs]):
        end_points = {}
        with slim.arg_scope([slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            weights_regularizer=slim.l2_regularizer(0.01)):
            net = slim.fully_connected(inputs, 32, scope='fc1')
            end_points['fc1'] = net

            net = slim.dropout(net, 0.8, is_training=is_training)

            net = slim.fully_connected(net, 16, scope='fc2')
            end_points['fc2'] = net

            predictions = slim.fully_connected(net, 1, activation_fn=None, scope='prediction')
            end_points['out'] = predictions

            return predictions, end_points


with tf.Graph().as_default():
    inputs = tf.placeholder(tf.float32, shape=(None, 1))
    outpus = tf.placeholder(tf.float32, shape=(None, 1))

    prediction, end_points = regression_model(inputs)

    # Print name and shape of each tensor.
    print('Layers')
    for k, v in end_points.items():
        print('name = {}, shape = {}'.format(v.name, v.get_shape()))

    # Print name and shape of parameter nodes
    print('\nParamers')
    for v in slim.get_model_variables():
        print('name = {}, shape = {}'.format(v.name, v.get_shape()))


def produce_batch(batch_size, noise=0.3):
    xs = np.random.random(size=[batch_size, 1]) * 10
    ys = np.sin(xs) + np.random.normal(size=[batch_size, 1], scale=noise)
    return [xs.astype(np.float32), ys.astype(np.float32)]


x_train, y_train = produce_batch(200)
x_test, y_test = produce_batch(200)
# plt.scatter(x_train, y_train)
# plt.show()

def convert_data_to_tensors(x, y):
    inputs = tf.constant(x)
    inputs.set_shape([None, 1])

    outputs = tf.constant(y)
    outputs.set_shape([None, 1])
    return inputs, outputs


ckpt_dir = '/tmp/regression_model/'

with tf.Graph().as_default():
        tf.logging.set_verbosity(tf.logging.INFO)

        inputs, targets = convert_data_to_tensors(x_train, y_train)

        predictions, nodes = regression_model(inputs, is_training=True)

        loss = tf.losses.mean_squared_error(labels=targets, predictions=predictions)

        total_loss = slim.losses.get_total_loss()

        optimizer = tf.train.AdamOptimizer(learning_rate=0.005)
        train_op = slim.learning.create_train_op(total_loss, optimizer)

        final_loss = slim.learning.train(
            train_op,
            logdir=ckpt_dir,
            number_of_steps=5000,
            save_summaries_secs=5,
            log_every_n_steps=500
        )

print("Finished training. Last batch loss:", final_loss)
print("Checkpoint saved in %s" % ckpt_dir)

