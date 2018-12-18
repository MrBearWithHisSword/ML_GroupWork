import os
import sys
import getopt
import tensorflow as tf
from inception_resnet_v2 import inception_resnet_v2, inception_resnet_v2_arg_scope
from data_preprocessing import *
import numpy as np
# import matplotlib.pyplot as plt
from hyperparams import Hyperparameters
# plt.style.use('ggplot')
slim = tf.contrib.slim

checkpoint_file = tf.train.latest_checkpoint(Hyperparameters.log_dir)

# anchor_path = '/home/shihaochen/webface/4421617/001.jpg'
# img_path = '/home/shihaochen/webface/4421617/002.jpg'


def distance(E1, E2):
    dis = tf.reduce_sum((E1-E2)**2, -1, keepdims=True)
    return dis


def predict(anchor_path, img_path):
    with tf.Graph().as_default() as graph:
        # Read
        anchor_string = tf.gfile.GFile(anchor_path, 'rb').read()
        img_string = tf.gfile.GFile(img_path, 'rb').read()
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
                restore_fn(sess)
            except ValueError:
                print("Error: Can't load model's trained parameters form save_path when it is None.")
                print("If you want to test our model with random initialized params, please delete or note the 76 line of this script.")
                print("Error: 如果你看到这条信息，说明我们的模型没有被成功加载，请在hyperparams.py中检查log_dir目录下保存的模型或与我联系。"
                      "如果你只是希望用随机初始化的参数测试模型是否能跑通，请注释掉这个脚本里的第76行（sys.exit())并重新执行。")
                # sys.exit(3)
            predictions = sess.run(predictions)
            predictions = np.squeeze(predictions)
            print("the predictions is:{}".format(predictions))
            return predictions
        # anchor_norm = tf.squeeze(anchor_norm)
        # img_norm = tf.squeeze(img_norm)
        # with tf.Session() as sess:
        #     anchor_norm = sess.run(anchor_norm)
        #     img_norm = sess.run(img_norm)
        # plt.imshow(anchor)
        # plt.figure()
        # plt.imshow(img)
        # plt.figure()
        # plt.imshow(anchor_norm)
        # plt.figure()
        # plt.imshow(img_norm)
        # plt.show()


# Using getopt module
def main(argv):
    inputfile = []
    outputfile = ''
    preds = int(0)
    # Check file_path & Get the two img.
    try:
        opts, args = getopt.getopt(argv, "i:o:")
    except getopt.GetoptError:
        print("You should run command in shell like: python <this file's name>  -i <img_1> -i <img_2> -o <output_file>")
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-i':
            inputfile.append(arg)
        elif opt == '-o':
            outputfile = arg
    anchor_path = inputfile[0]
    img_path = inputfile[1]
    # Predict
    print("your img1_path is: {}".format(anchor_path))
    print("your img2_path is: {}".format(img_path))
    prediction = predict(anchor_path, img_path)

    # Write the answer in to output_file.
    # Here we add the answer to the end of the file, for your convenience to check the result.
    # 我们在这里直接将预测的结果追加到输出文件的，这样或许可以让你更方便地批量检查预测结果。
    with open(outputfile, 'a') as f:
        f.write(str(prediction) + '\n')
    # print("Your input file is:{}".format(inputfile))
    print("For your convenience, the answer is written in :{}".format(outputfile))
    print("为方便您check the answer, 我们已经将结果追加到文件'{}'中去。其中每一行为一次对比的预测结果。".format(outputfile))


if __name__ == '__main__':
    # print(checkpoint_file)
    main(sys.argv[1:])


