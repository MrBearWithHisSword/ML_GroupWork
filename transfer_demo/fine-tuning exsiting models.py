import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np
import matplotlib.pyplot as plt

log_dir = '/tmp/regression_model/fine-tuning_model/'

ckpt_dir = '/tmp/regression_model/model.ckpt'

"""Here is the code that retrieve the whole model."""
# def produce_batch(batch_size, noise=0.3):
#     xs = np.random.random(size=[batch_size, 1]) * 10
#     ys = np.sin(xs) + np.random.normal(size=[batch_size, 1], scale=noise)
#     return [xs.astype(np.float32), ys.astype(np.float32)]
#
#
# x_train, y_train = produce_batch(200)
# x_test, y_test = produce_batch(200)
#
# # Reset graph
# tf.reset_default_graph()
#
# # Data
# x_train = tf.convert_to_tensor(x_train)
# y_train = tf.convert_to_tensor(y_train)
#
# # Graph
# inputs = tf.placeholder(dtype=tf.float32, shape=[None, 1])
# # outputs = tf.placeholder(dtype=tf.float32, shape=[None, 1])
#
# with tf.variable_scope('deep_regression', [inputs]):
#     end_points = {}
#     with slim.arg_scope([slim.fully_connected],
#                         activation_fn=tf.nn.relu,
#                         weights_regularizer=slim.l2_regularizer(0.01)):
#         net = slim.fully_connected(inputs, 32, scope='fc1')
#         end_points['fc1'] = net
#
#         net = slim.dropout(net, 0.8, is_training=True)
#
#         net = slim.fully_connected(net, 16, scope='fc2')
#         end_points['fc2'] = net
#
#         predictions = slim.fully_connected(net, 1, activation_fn=None, scope='prediction')
#         end_points['out'] = predictions
# outputs = predictions
#
# # variables to restore
# variables_to_restore = slim.get_model_variables()
# # got the variable's name in the saved model
# def name_in_checkpoint(var):
#     """
#     :param var: the variable's name of your new model
#     :return:  the corresponding variable's name of the pre-trained model
#     """
#
#     # # demo
#     # if 'weights' in var.op.name:
#     #     return var.op.name.replace("weights", "params1")
#
#     # 因为这里我的模型的变量名和checkpoint中的完全相同,不需要转换
#     return var.op.name
#
#
# variables_to_restore = {name_in_checkpoint(var): var for var in variables_to_restore}
# restorer = tf.train.Saver(variables_to_restore)
#
# # restore the variables.
# with tf.Session() as sess:
#     restorer.restore(sess, ckpt_dir)
#     print('Variables restored.')
#     for v in slim.get_model_variables():
#         print("name = {}, val: = {}".format(v.name, v.eval()))
#
#     targets = outputs
#     # inputs = x_train
#     inputs, preds, targets = sess.run([inputs, outputs, y_train], )
#
#
# # testing the restored model.
# # with tf.Session() as sess:
# #     sess.run(tf.global_variables_initializer())
# #     targets = outputs
# #     inputs, preds, targets = sess.run([inputs, outputs, y_train], feed_dict={inputs: x_train})
# plt.scatter(inputs, targets, c='r')
# plt.scatter(inputs, preds, c='b')
# plt.title('red=true, blue=predicted')
# plt.show()

"""Here is the code that fine-tuning the model."""
# Fine-tuning
# Reset graph
tf.reset_default_graph()


# Build new model
def fine_tuning_model(inputs, is_training=True, scope='my_model'):
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


# Got the variable's name in the checkpoint.
def name_in_checkpoint(var):
    if "my_model" in var.op.name:
        return var.op.name.replace("my_model", "deep_regression")


# Get uninitialized variables.
def guarantee_initialized_variables(session, list_of_variables=None):
    if list_of_variables is None:
        list_of_variables = tf.all_variables()
    uninitialized_variables = list(tf.get_variable(name) for name in
                                   session.run(tf.report_uninitialized_variables(list_of_variables)))
    session.run(tf.initialize_variables(uninitialized_variables))
    return uninitialized_variables


# Construct graph
with tf.Graph().as_default():
    x_train, y_train = produce_batch(200)
    x_test, y_test = produce_batch(200)
    x_train = tf.convert_to_tensor(x_train)
    y_train = tf.convert_to_tensor(y_train)

    preds = fine_tuning_model(x_train, is_training=True)

    # Define the loss
    loss = tf.losses.mean_squared_error(predictions=preds, labels=y_train)
    total_loss = slim.losses.get_total_loss()
    # # Specify the optimizer and create the train op.
    # optimizer = tf.train.AdamOptimizer(learning_rate=0.005)
    # train_op = slim.learning.create_train_op(total_loss=total_loss,
    #                                          optimizer=optimizer)

    # Variables' initializer
    initializer = tf.global_variables_initializer()

    # # Show all the variables in the model
    # vars = slim.get_model_variables()
    # for v in vars:
    #     print("name: {}, shape: {}".format(v.name, v.get_shape()))

    # Get the variables which need to be restored
    variables_to_restore = slim.get_variables(scope="my_model/fc1",)
    for v in variables_to_restore:
        print("name: {}, shape: {}".format(v.name, v.get_shape()))
    variables_to_restore = {name_in_checkpoint(var): var for var in variables_to_restore}
    for ckpt_name, var in variables_to_restore.items():
        print("ckpt_name: {}".format(ckpt_name))

    # Create the saver which will be used to restore the variables.
    restorer = tf.train.Saver(variables_to_restore)

    # Specify the optimizer and create the train op.
    optimizer = tf.train.AdamOptimizer(learning_rate=0.005)
    train_op = slim.learning.create_train_op(total_loss=total_loss,
                                             optimizer=optimizer)

    # Create the Session to: initialize variables, run the restorer, fine-tuning the new model
    # and Save it to the log_dir.
    with tf.Session() as sess:
        # Initialize all the variables.
        sess.run(initializer)

        # # Check current model's variables.
        # print("Current model's variables:")
        # vars = slim.get_model_variables()
        # for v in vars:
        #     print("name: {}, val: {}".format(v.name, v.eval()))

        # Restore the specified variables.
        restorer.restore(sess, ckpt_dir)
        print('Specified variables restored.')
        for name, var in variables_to_restore.items():
            print("name: {}, val: {}".format(name, var.eval()))

        # Check the new model's variables.
        print("Current model's variables:")
        vars = slim.get_model_variables()
        for v in vars:
            print("name: {}, val: {}".format(v.name, v.eval()))

    # Fine-tuning the new model
    final_loss = slim.learning.train(train_op=train_op,
                                     logdir=log_dir,
                                     number_of_steps=5000,
                                     log_every_n_steps=500)
    print('Finished fine-tuning. Last loss:', final_loss)
    print("Checkpoint saved in %s" % log_dir)

