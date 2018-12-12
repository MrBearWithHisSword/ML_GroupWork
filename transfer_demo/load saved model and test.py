import tensorflow as tf
from tensorflow.contrib import slim
import matplotlib.pyplot as plt
import numpy as np


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


def produce_batch(batch_size, noise=0.3):
    xs = np.random.random(size=[batch_size, 1]) * 10
    ys = np.sin(xs) + np.random.normal(size=[batch_size, 1], scale=noise)
    return [xs.astype(np.float32), ys.astype(np.float32)]


x_train, y_train = produce_batch(200)
x_test, y_test = produce_batch(200)

log_dir = '/tmp/regression_model/'

with tf.Graph().as_default():
    # inputs, targets = tf.convert_to_tensor(x_test, y_test)
    inputs = tf.convert_to_tensor(x_test)
    targets = tf.convert_to_tensor(y_test)
    predictions, end_points = regression_model(inputs, is_training=False)

    # Make a session which restores the old parameters from a checkpoint.
    sv = tf.train.Supervisor(logdir=log_dir)
    with sv.managed_session() as sess:
        inputs, predictions, targets = sess.run([inputs, predictions, targets])

    variables = slim.get_model_variables()
    for v in variables:
        print("name = {}, shape = {}".format(v.name, v.get_shape()))
plt.scatter(inputs, targets, c='r')
plt.scatter(inputs, predictions, c='b')
plt.title('red=true, blue=predicted')
plt.show()