import os
import tensorflow as tf
from hyperparams import Hyperparameters
import numpy as np
import matplotlib.pyplot as plt


#==================================utils=============================
def int64_feature(values):
  """Returns a TF-Feature of int64s.

  Args:
    values: A scalar or list of values.

  Returns:
    A TF-Feature.
  """
  if not isinstance(values, (tuple, list)):
    values = [values]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def bytes_feature(values):
  """Returns a TF-Feature of bytes.

  Args:
    values: A string.

  Returns:
    A TF-Feature.
  """
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def float_feature(values):
  """Returns a TF-Feature of floats.

  Args:
    values: A scalar of list of values.

  Returns:
    A TF-Feature.
  """
  if not isinstance(values, (tuple, list)):
    values = [values]
  return tf.train.Feature(float_list=tf.train.FloatList(value=values))


file_pattern = 'face_%s_'

pair_feature_description = {
    'anchor_raw': tf.FixedLenFeature([], tf.string),
    'img_raw': tf.FixedLenFeature([], tf.string),
    'label': tf.FixedLenFeature([], tf.int64),
}


def _parse_pair_function(example_proto):
    parsed_pair = tf.parse_single_example(example_proto, pair_feature_description)
    parsed_pair['anchor_raw'] = tf.image.decode_jpeg(parsed_pair['anchor_raw'])
    parsed_pair['img_raw'] = tf.image.decode_jpeg(parsed_pair['img_raw'])
    # parsed_pair['anchor_raw'] = tf.image.resize_nearest_neighbor(parsed_pair['anchor_raw'],
    #                                                              [Hyperparameters.img_height, Hyperparameters.img_width])
    parsed_pair['anchor_raw'] = tf.image.resize_images(parsed_pair['anchor_raw'],
                                                       [Hyperparameters.img_height, Hyperparameters.img_width])
    parsed_pair['img_raw'] = tf.image.resize_images(parsed_pair['img_raw'],
                                                    [Hyperparameters.img_height, Hyperparameters.img_width])
    return parsed_pair

def get_split(tfrecord_dir, file_pattern, split_name):
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
    dataset = get_split(Hyperparameters.tfrecord_dir, file_pattern, split_name='train')
    dataset = dataset.shuffle(buffer_size=1000, seed=Hyperparameters.random_seed)
    dataset = dataset.batch(Hyperparameters.batch_size)
    dataset = dataset.prefetch(buffer_size=2)
    iterator = dataset.make_initializable_iterator()
    batch_of_pairs = iterator.get_next()
    print(batch_of_pairs)
    with tf.Session() as sess:
        sess.run(iterator.initializer)
        for i in range(5):
            try:
                pair_batch = sess.run(batch_of_pairs)
                anchor = pair_batch['anchor_raw'][0]
                img = pair_batch['img_raw'][0]
                label = pair_batch['label'][0]
                print(label)
                plt.imshow(anchor.astype(np.uint8))
                plt.figure()
                plt.imshow(img.astype(np.uint8))
                plt.show()
            except tf.errors.OutOfRangeError:
                print('End of Epoch')
