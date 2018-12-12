import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.contrib import slim

log_dir = '/tmp/regression_model/fine-tuning_model/'

# Model
def fine_tuned_model(inputs, is_training=True, scope='my_model'):
    with tf.variable_scope(scope, 'my_model', values=[inputs]):
        with slim.arg_scope([slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            weights_regularizer=slim.l2_regularizer(0.02)):
            # here I changed the weights_regularizer. Besides, I removed the end_points tracker.
            net = slim.fully_connected(inputs, 32, scope='fc1')
            net = slim.dropout(net, 0.8, is_training=is_training)
            predictions = slim.fully_connected(net, 1, activation_fn=None, scope='prediction')
            return predictions


# Data
def produce_batch(batch_size, noise=0.3):
    xs = np.random.random(size=[batch_size, 1]) * 10
    ys = np.sin(xs) + np.random.normal(size=[batch_size, 1], scale=noise)
    return [xs.astype(np.float32), ys.astype(np.float32)]

x_test, y_test = produce_batch(200)


# Graph
with tf.Graph().as_default():
    x = tf.placeholder(dtype=tf.float32, shape=[None, 1])
    y = tf.placeholder(dtype=tf.float32, shape=[None, 1])
    preds = fine_tuned_model(inputs=x, is_training=False)

    sv = tf.train.Supervisor(logdir=log_dir)
    with sv.managed_session() as sess:
        inputs, predictions, targets = sess.run([x, preds, y],
                                                feed_dict={x: x_test, y: y_test})

plt.scatter(inputs, targets, c='r')
plt.scatter(inputs, predictions, c='b')
plt.title('red=true, blue=predicted')
plt.show()
