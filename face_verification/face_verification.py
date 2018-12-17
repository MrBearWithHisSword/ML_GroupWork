import os
import tensorflow as tf
from inception_resnet_v2 import inception_resnet_v2, inception_resnet_v2_arg_scope
from data_preprocessing import *
import matplotlib.pyplot as plt
from hyperparams import Hyperparameters
plt.style.use('ggplot')
slim = tf.contrib.slim

checkpoint_file = tf.train.latest_checkpoint(Hyperparameters.log_dir)

anchor_path = '/home/hcshi/Class_Assignment/01_FaceVerification/webface/4000704/001.jpg'
img_path = '/home/hcshi/Class_Assignment/01_FaceVerification/webface/4000704/002.jpg'

def distance(E1, E2):
    dis = tf.reduce_sum((E1-E2)**2, -1, keepdims=True)
    return dis

def predict(anchor_path, img_path):
    with tf.Graph().as_default() as graph:
        # Read
        anchor_string = tf.gfile.FastGFile(anchor_path, 'rb').read()
        img_string = tf.gfile.FastGFile(img_path, 'rb').read()
        anchor = tf.image.decode_jpeg(anchor_string, channels=3)
        img = tf.image.decode_jpeg(img_string, channels=3)
        # Resize
        anchor = tf.image.resize_images(anchor, [299, 299])
        img = tf.image.resize_images(img, [299, 299])
        # Normalization
        anchor_norm = tf.image.per_image_standardization(anchor)
        img_norm = tf.image.per_image_standardization(img)
        # Batch up the input
        anchor_norm = tf.expand_dims(anchor_norm, 0)
        img_norm = tf.expand_dims(img_norm, 0)
        # Build graph
        with slim.arg_scope(inception_resnet_v2_arg_scope()):
            _, end_points = inception_resnet_v2(anchor_norm,
                                                num_classes=None,
                                                create_aux_logits=False,
                                                reuse=tf.AUTO_REUSE,
                                                is_training=False)
            E1 = end_points['global_pool']
            E1 = slim.flatten(E1)
            E1 = slim.dropout(E1, 0.8, is_training=True, scope='Dropout')
            _, end_points = inception_resnet_v2(img_norm,
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
        # Run session
        with tf.Session(graph=graph) as sess:
            sess.run(tf.global_variables_initializer())
            try:
                sess.run(restore_fn(sess))
            except ValueError:
                print("Can't load model's trained parameters form save_path when it is None.")
            predictions = sess.run(predictions)
            print(predictions)
        anchor_norm = tf.squeeze(anchor_norm)
        img_norm = tf.squeeze(img_norm)
        with tf.Session() as sess:
            anchor_norm = sess.run(anchor_norm)
            img_norm = sess.run(img_norm)
        # plt.imshow(anchor)
        # plt.figure()
        # plt.imshow(img)
        # plt.figure()
        plt.imshow(anchor_norm)
        plt.figure()
        plt.imshow(img_norm)
        plt.show()

if __name__ == '__main__':
    predict(anchor_path, img_path)

