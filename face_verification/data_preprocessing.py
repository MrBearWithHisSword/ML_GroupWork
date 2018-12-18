import os
import tensorflow as tf
from hyperparams import Hyperparameters
import time
import numpy as np
# import matplotlib.pyplot as plt


#=========================Preprocessing=========================


pair_feature_description = {
    'anchor_raw': tf.FixedLenFeature([], tf.string),
    'img_raw': tf.FixedLenFeature([], tf.string),
    'label': tf.FixedLenFeature([], tf.int64),
}


def _parse_pair_function(example_proto):
    parsed_pair = tf.parse_single_example(example_proto, pair_feature_description)
    parsed_pair['anchor_raw'] = tf.image.decode_jpeg(parsed_pair['anchor_raw'], channels=3)
    parsed_pair['img_raw'] = tf.image.decode_jpeg(parsed_pair['img_raw'], channels=3)
    # parsed_pair['anchor_raw'] = tf.image.resize_nearest_neighbor(parsed_pair['anchor_raw'],
    #                                                              [Hyperparameters.img_height, Hyperparameters.img_width])
    parsed_pair['anchor_raw'] = tf.image.resize_images(parsed_pair['anchor_raw'],
                                                       [Hyperparameters.img_height, Hyperparameters.img_width])
    parsed_pair['img_raw'] = tf.image.resize_images(parsed_pair['img_raw'],
                                                    [Hyperparameters.img_height, Hyperparameters.img_width])
    parsed_pair['anchor_raw'] = tf.image.per_image_standardization(parsed_pair['anchor_raw'])
    parsed_pair['img_raw'] = tf.image.per_image_standardization(parsed_pair['img_raw'])
    
    return parsed_pair


def get_split(tfrecord_dir, split_name):
    filenames = []
    file_list = os.listdir(tfrecord_dir)
    for file_name in file_list:
        if split_name in file_name:
            filenames.append(tfrecord_dir + file_name)
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(_parse_pair_function, Hyperparameters.parallel_calls)
    return dataset

# Test
if __name__ == '__main__':
    dataset = get_split(Hyperparameters.tfrecord_dir, split_name='train')
    # dataset = dataset.shuffle(buffer_size=1000, seed=Hyperparameters.random_seed)
    dataset = dataset.shuffle(buffer_size=500, seed=10)
    # dataset = dataset.repeat(Hyperparameters.epoch_num)
    dataset = dataset.batch(Hyperparameters.batch_size)
    dataset = dataset.prefetch(buffer_size=Hyperparameters.prefetch_buffer_size)
    iterator = dataset.make_initializable_iterator()
    batch_of_pairs = iterator.get_next()
    print(batch_of_pairs)
    batch_num = 0
    with tf.Session() as sess:
        sess.run(iterator.initializer)
        time_pre = time.time()
        for i in range(2):
            try:
                pair_batch = sess.run(batch_of_pairs)
                # batch_num = batch_num + 1
                # print(batch_num)
                # print(batch_num*32)
                # time_cur = time.time()
                # print('use time: {}'.format(time_cur - time_pre))
                # time_pre = time_cur
                # anchor = pair_batch['anchor_raw'][0]
                # img = pair_batch['img_raw'][0]
                label = pair_batch['label']
                print(label)
                print(label.shape)
                # plt.imshow(anchor.astype(np.uint8))
                # plt.figure()
                # plt.imshow(img.astype(np.uint8))
                # plt.show()
            except tf.errors.OutOfRangeError:
                print('End of Epoch')
