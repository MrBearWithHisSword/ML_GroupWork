import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

slim = tf.contrib.slim


# # show image
# filename = '/home/hcshi/Class_Assignment/01_FaceVerification/webface/0000100/001.jpg'
#
# image_raw = tf.gfile.FastGFile(filename, 'rb').read()
# img = tf.image.decode_jpeg(image_raw)
# img_ = img
# img = tf.expand_dims(img, 0)
# img = tf.image.resize_nearest_neighbor(img, [299, 299])
# img = tf.squeeze(img)
#
# with tf.Session() as sess:
#     print(img.eval().shape)
#     print(img.eval().dtype)
#     print(img.eval())
#     plt.figure(1)
#     plt.imshow(img.eval())
#     plt.figure(2)
#     plt.imshow(img_.eval())
#     plt.show()

# ==================================utils=============================
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


# train_rate = 0.8
#
# # cwd = os.getcwd()
# data_dir = '/home/hcshi/Class_Assignment/01_FaceVerification/webface/'
# train_filename = '/home/hcshi/Class_Assignment/01_FaceVerification/code/inception_res_v2/fake_data/face_train'
# test_filename = '/home/hcshi/Class_Assignment/01_FaceVerification/code/inception_res_v2/fake_data/face_test'
# labels_filename = '/home/hcshi/Class_Assignment/01_FaceVerification/code/inception_res_v2/fake_data/lables.txt'
#
#
# def write_to_tfrecord(data_dir=data_dir, train_rate=train_rate, height=299, width=299):
#     with tf.Graph().as_default():
#         img_placeholder = tf.placeholder(dtype=tf.uint8)
#         encoded_img = tf.image.encode_jpeg(img_placeholder)
#
#         person_list = os.listdir(data_dir)
#         person_num = len(person_list)
#         train_person_num = int(person_num * train_rate)
#         # write training_tfrecord
#         offset = 0
#         for class_id in range(3):
#             tfrecord_writer = tf.python_io.TFRecordWriter(train_filename + '_' + str(offset) + '.tfrecord')
#             # offset = offset + 1
#             print(class_id)
#             person_dir = data_dir + '/' + person_list[class_id] + '/'
#             for face_id in os.listdir(person_dir):
#                 img_path = person_dir + face_id
#                 img = Image.open(img_path)
#                 img = img.resize((height, width))
#                 img = np.array(img).astype(np.uint8)
#                 with tf.Session() as sess:
#                     jpg_string = sess.run(encoded_img,
#                                           feed_dict={img_placeholder: img})
#                 example = tf.train.Example(features=tf.train.Features(feature={
#                     'image/encoded': bytes_feature(jpg_string),
#                     'image/format': bytes_feature(b'jpg'),
#                     'image/class/label': int64_feature(class_id),
#                     # 'image/height': int64_feature(height),
#                     # 'image/width': int64_feature(width)
#                 }))
#                 tfrecord_writer.write(example.SerializeToString())
#         # Write testing_tfrecord
#         offset = 0
#         for class_id in range(train_person_num, train_person_num + 2):
#             tfrecord_writer = tf.python_io.TFRecordWriter(test_filename + '_' + str(offset) + '.tfrecord')
#             # offset = offset + 1
#             print(class_id)
#             person_dir = data_dir + '/' + person_list[class_id] + '/'
#             print(person_dir)
#             print(class_id)
#             for face_id in os.listdir(person_dir):
#                 img_path = person_dir + face_id
#                 img = Image.open(img_path)
#                 img = img.resize((height, width))
#                 img = np.array(img).astype(np.uint8)
#                 with tf.Session() as sess:
#                     jpg_string = sess.run(encoded_img,
#                                           feed_dict={img_placeholder: img})
#                 example = tf.train.Example(features=tf.train.Features(feature={
#                     'image/encoded': bytes_feature(jpg_string),
#                     'image/format': bytes_feature(b'jpg'),
#                     'image/class/label': int64_feature(class_id),
#                     # 'image/height': int64_feature(height),
#                     # 'image/width': int64_feature(width)
#                 }))
#                 tfrecord_writer.write(example.SerializeToString())
#
#
# def write_labels_txt(data_dir, lables_filename):
#     class_to_person_id = {}
#     person_list = os.listdir(data_dir)
#     person_num = len(person_list)
#     for class_id in range(person_num):
#         person_id = person_list[class_id]
#         class_to_person_id[class_id] = person_id
#
#     f = open(labels_filename, 'w')
#     for class_id, person_id in class_to_person_id.items():
#         f.writelines(str(class_id) + ':' + person_id + '\n')
#     f.close()
#
# class_to_person_id = {}
# person_list = os.listdir(data_dir)
# person_num = len(person_list)
# for class_id in range(person_num):
#     person_id = person_list[class_id]
#     class_to_person_id[class_id] = person_id
#
# # ================================get dataset====================================
# file_pattern = 'face_%s.tfrecord'
# dataset_dir = './fake_data/'
# def get_split(split_name, dataset_dir, file_pattern=file_pattern, file_pattern_for_counting='face'):
#     """
#
#     :param split_name:
#     :param dataset_dir:
#     :param file_pattern:
#     :param file_pattern_for_counting:
#     :return:
#     """
#     # First check whether the split_name is train or test
#     if split_name not in ['train', 'test']:
#         raise ValueError('The split_name %s is not recognized. Please input either train or test'
#                          ' as the split_name' % (split_name))
#
#     # Create the full path for a general file_pattern to locate the tfrecord_files
#     file_pattern_path = os.path.join(dataset_dir, file_pattern % (split_name))
#
#     # Count the total number of examples in all of these shard
#     num_samples = 0
#     file_pattern_for_counting = file_pattern_for_counting + '_' + split_name
#     # file_pattern_for_counting = file_pattern_for_counting
#     tfrecords_to_count = [os.path.join(dataset_dir, file) for file in os.listdir(dataset_dir) if
#                           file.startswith(file_pattern_for_counting)]
#     tfrecords_to_count.sort()
#     print(tfrecords_to_count)
#     for tfrecord_file in tfrecords_to_count:
#         for record in tf.python_io.tf_record_iterator(tfrecord_file):
#             num_samples += 1
#             print(num_samples)
#
#     # Create a reader, which must be a TFRecord reader in this case
#     reader = tf.TFRecordReader
#
#     # Create the keys_to_features dictionary for the decoder
#     keys_to_features = {
#         'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
#         'image/format': tf.FixedLenFeature((), tf.string, default_value='jpg'),
#         'image/class/label': tf.FixedLenFeature(
#             [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
#     }
#
#     # Create the items_to_handlers dictionary for the decoder.
#     items_to_handlers = {
#         'image': slim.tfexample_decoder.Image(),
#         'label': slim.tfexample_decoder.Tensor('image/class/label'),
#     }
#
#     # Start to create the decoder
#     decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)
#
#     # Create the labels_to_name file
#     labels_to_name_dict = class_to_person_id
#
#     # Actually create the dataset
#     dataset = slim.dataset.Dataset(
#         data_sources=file_pattern_path,
#         decoder=decoder,
#         reader=reader,
#         num_readers=4,
#         num_samples=None,
#         # num_classes=num_classes,
#         labels_to_name=labels_to_name_dict,
#         items_to_descriptions=None)
#
#     return dataset
#
# image_size = 299
# def load_batch(dataset, batch_size, height=image_size, width=image_size, is_training=True):
#     # First create the data_provider object
#     data_provider = slim.dataset_data_provider.DatasetDataProvider(
#         dataset,
#         common_queue_capacity=24 + 3 * batch_size,
#         common_queue_min=24)
#
#     # Obtain the raw image using the get method
#     raw_image, label = data_provider.get(['image', 'label'])
#     return raw_image, label


# ==================================Write==============================================
# # tf.enable_eager_execution()
# cat_in_snow = '/home/hcshi/Class_Assignment/01_FaceVerification/webface/0000045/001.jpg'
# williamsburg_bridge = '/home/hcshi/Class_Assignment/01_FaceVerification/webface/0000045/002.jpg'
#
#
# image_labels = {
#     cat_in_snow: 0,
#     williamsburg_bridge: 1,
# }
# image_string = open(cat_in_snow, 'rb').read()
# label = image_labels[cat_in_snow]
#
#
# def image_example(image_string, label):
#     # image_shape = tf.image.decode_jpeg(image_string).shape
#     feature = {
#       # 'height': int64_feature(image_shape[0]),
#       # 'width': int64_feature(image_shape[1]),
#       # 'depth': int64_feature(image_shape[2]),
#       'label': int64_feature(label),
#       'image_raw': bytes_feature(image_string),
#     }
#     return tf.train.Example(features=tf.train.Features(feature=feature))
# #
# # for line in str(image_example(image_string, label)).split('\n')[:15]:
# #     print(line)
#
# with tf.python_io.TFRecordWriter('images.tfrecords') as writer:
#   for filename, label in image_labels.items():
#     image_string = open(filename, 'rb').read()
#     # print(image_string)
#     tf_example = image_example(image_string, label)
#     writer.write(tf_example.SerializeToString())

# =======================================Read====================================
raw_image_dataset = tf.data.TFRecordDataset('images.tfrecords')
# Create a dictionary describing the features.
image_feature_description = {
    # 'height': tf.FixedLenFeature([], tf.int64),
    # 'width': tf.FixedLenFeature([], tf.int64),
    # 'depth': tf.FixedLenFeature([], tf.int64),
    'label': tf.FixedLenFeature([], tf.int64),
    'image_raw': tf.FixedLenFeature([], tf.string),
}


def _parse_image_function(example_proto):
    # Parse the input tf.Example proto using the dictionary above.
    return tf.parse_single_example(example_proto, image_feature_description)


parsed_image_dataset = raw_image_dataset.map(_parse_image_function)

# =======================================Recover==================================

# for image_features in parsed_image_dataset:
#     image_raw = image_features['image_raw'].numpy()
#     image_raw = tf.image.decode_jpeg(image_raw)
#     print('image_raw:', image_raw)
#     plt.imshow(image_raw)
#     plt.show()

from hyperparams import Hyperparameters

data_dir = '/home/hcshi/Class_Assignment/01_FaceVerification/webface/'


def select_random_negative(data_dir, anchor_id, is_training_set):
    person_list = os.listdir(data_dir)
    # Select negative person_id
    train_rate = Hyperparameters.split_train_rate
    train_num = int(train_rate * len(person_list))
    if is_training_set:
        negative_id = person_list[np.random.randint(train_num)]
        while negative_id == anchor_id:
            negative_id = person_list[np.random.randint(train_num)]
    else:
        negative_id = person_list[np.random.randint(train_num, len(person_list))]
        while negative_id == anchor_id:
            negative_id = person_list[np.random.randint(train_num)]
    negative_path = data_dir + '/' + negative_id + '/'
    negative_example_list = os.listdir(negative_path)
    negative_img = negative_example_list[np.random.randint(len(negative_example_list))]
    negative_img = negative_path + negative_img
    img = Image.open(negative_img)
    img = np.array(img.resize((Hyperparameters.img_height, Hyperparameters.img_width))).astype(np.uint8)
    return img


def select_all_positive(data_dir, anchor_id, anchor_img):
    all_positive = []
    positive_id = anchor_id
    positive_path = data_dir + '/' + positive_id + '/'
    for positive_img in os.listdir(positive_path):
        if positive_img == anchor_img:
            continue
        positive_img = positive_path + positive_img
        img = Image.open(positive_img)
        img = np.array(img.resize((Hyperparameters.img_height, Hyperparameters.img_width))).astype(np.uint8)
        all_positive.append(img)
        # image_string = open(positive_img, 'rb').read()
        # all_positive.append(image_string)
    return all_positive


def pair_example(anchor_string, img_string, label):
    feature = {
        # 'height': int64_feature(image_shape[0]),
        # 'width': int64_feature(image_shape[1]),
        # 'depth': int64_feature(image_shape[2]),
        'anchor_raw': bytes_feature(anchor_string),
        'img_raw': bytes_feature(img_string),
        'label': int64_feature(label)
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


# for line in str(pair_example(image_string, image_string, 1)).split('\n')[:]:
#     print(line)


def get_image_string(img_path):
    img = Image.open(img_path)
    img = np.array(img.resize((Hyperparameters.img_height, Hyperparameters.img_width))).astype(np.uint8)
    with tf.Session() as sess:
        img_string = sess.run(tf.image.encode_jpeg(img))
    return img_string


# ==========================Write=====================
# with tf.python_io.TFRecordWriter('images.tfrecords') as writer:
#     person_list = os.listdir(data_dir)
#     for i in range(1):
#         person_id = person_list[i]
#         person_path = data_dir + '/' + person_id + '/'
#         all_positive = []
#         anchor_list =  os.listdir(person_path)
#         for anchor_id in range(len(anchor_list)):
#             anchor_string = get_image_string(person_path + anchor_list[anchor_id])
#             # anchor_img =Image.open(person_path + anchor_list[anchor_id])
#             # anchor_img = np.array(anchor_img.resize((299,299))).astype(np.uint8)
#             # anchor_string = tf.image.encode_jpeg(anchor_img)
#             for img_id in range(anchor_id+1, len(anchor_list)):
#                 img_string = get_image_string(person_path + anchor_list[img_id])
#                 # img = Image.open(person_path + anchor_list[img_id])
#                 # img = np.array(img.resize((299,299))).astype(np.uint8)
#                 # img_string = tf.image.encode_jpeg(img)
#             tf_example = pair_example(anchor_string, img_string, 1)
#             writer.write(tf_example.SerializeToString())


# ==========================Read=========================
# raw_pair_dataset = tf.data.TFRecordDataset('images.tfrecords')
# image_feature_description = {
#     'anchor_raw': tf.FixedLenFeature([], tf.string),
#     'img_raw': tf.FixedLenFeature([], tf.string),
#     'label': tf.FixedLenFeature([], tf.int64),
# }
#
#
# def _parse_pair_function(example_proto):
#     # Parse the input tf.Example proto using the dictionary above.
#     return tf.parse_single_example(example_proto, image_feature_description)
#
#
# parsed_pair_dataset = raw_pair_dataset.map(_parse_image_function)
# =========================Retrive=======================
# for pair_features in parsed_pair_dataset:
#     image_raw = pair_features['img_raw'].numpy()
#     anchor_raw = pair_features['anchor_raw'].numpy()
#     image_raw = tf.image.decode_jpeg(image_raw)
#     anchor_raw = tf.image.decode_jpeg(anchor_raw)
#     label = pair_features['label']
#     print("label:{}".format(label))
#     plt.imshow(image_raw)
#     plt.figure()
#     plt.imshow(anchor_raw)
#     plt.show()
# ===============not eager================================
# iterator = parsed_pair_dataset.make_one_shot_iterator()
# next_element = iterator.get_next()
# with tf.Session() as sess:
#     for i in range(3):
#         pair = next_element
#         anchor_raw = pair['anchor_raw']
#         img_raw = pair['img_raw']
#         label = pair['label']
#         anchor = tf.image.decode_jpeg(anchor_raw)
#         img = tf.image.decode_jpeg(img_raw)
#         anchor, img, label = sess.run([anchor, img, label])
#         plt.imshow(anchor)
#         plt.figure()
#         plt.imshow(img)
#         plt.show()
#         print('label:{}'.format(label))
#
#

train_pair_num = 0
train_file_offset = 0
validation_file_offset = 0
validation_pair_num = 0
train_filename = Hyperparameters.train_filename
validation_filename = Hyperparameters.validation_filename


def write_to_tfrecord(data_dir, train_rate,
                      train_file_offset=train_file_offset,
                      validation_file_offset=validation_file_offset,
                      train_pair_num=train_pair_num,
                      validation_pair_num=validation_pair_num):
    person_list = os.listdir(data_dir)
    train_person_num = int(train_rate * len(person_list))
    # write train_file
    for i in range(train_person_num):
        print('write {}th person\'s face into tf_record. Total: {}'.format(i, train_person_num))
        if i % 300 == 0:
            writer = tf.python_io.TFRecordWriter(train_filename + '_' + str(train_file_offset) + '.tfrecord')
            train_file_offset = train_file_offset + 1
        person_id = person_list[i]
        person_path = data_dir + '/' + person_id + '/'
        positive_num = 0
        anchor_list = os.listdir(person_path)
        for anchor_id in range(len(anchor_list)):
            anchor_string = get_image_string(person_path + anchor_list[anchor_id])
            # search all positive
            for img_id in range(anchor_id + 1, len(anchor_list)):
                img_string = get_image_string(person_path + anchor_list[img_id])
                tf_example = pair_example(anchor_string, img_string, 1)
                positive_num = positive_num + 1
                train_pair_num = train_pair_num + 1
                writer.write(tf_example.SerializeToString())
            # select same num of negative
            for num in range(positive_num):
                negative_img = select_random_negative(data_dir=data_dir,
                                                      anchor_id=anchor_list[anchor_id],
                                                      is_training_set=True)
                with tf.Session() as sess:
                    negative_string = sess.run(tf.image.encode_jpeg(negative_img))
                tf_example = pair_example(anchor_string, negative_string, 0)
                train_pair_num = train_pair_num + 1
                writer.write(tf_example.SerializeToString())
    # write validation file
    for i in range(train_person_num, len(person_list)):
        print('write {}th person\'s face into tf_record. Total: {}'.format(i, len(person_list)-train_person_num))
        if i % 300 == 0:
            writer = tf.python_io.TFRecordWriter(
                validation_filename + '_' + str(validation_file_offsetoffset) + '.tfrecord')
            validation_file_offset = validation_file_offset + 1
        person_id = person_list[i]
        person_path = data_dir + '/' + person_id + '/'
        positive_num = 0
        anchor_list = os.listdir(person_path)
        for anchor_id in range(len(anchor_list)):
            anchor_string = get_image_string(person_path + anchor_list[anchor_id])
            # search all positive
            for img_id in range(anchor_id + 1, len(anchor_list)):
                img_string = get_image_string(person_path + anchor_list[img_id])
                tf_example = pair_example(anchor_string, img_string, 1)
                positive_num = positive_num + 1
                validation_pair_num = validation_pair_num + 1
                writer.write(tf_example.SerializeToString())
            # select same num of negative
            for num in range(positive_num):
                negative_img = select_random_negative(data_dir=data_dir,
                                                      anchor_id=anchor_list[anchor_id],
                                                      is_training_set=False)
                with tf.Session() as sess:
                    negative_string = sess.run(tf.image.encode_jpeg(negative_img))
                tf_example = pair_example(anchor_string, negative_string, 0)
                validation_pair_num = validation_pair_num + 1
                writer.write(tf_example.SerializeToString())

    return train_pair_num, validation_pair_num,


train_pair_num, validation_pair_num = write_to_tfrecord(data_dir=data_dir,
                                                        train_rate=Hyperparameters.split_train_rate,
                                                        train_file_offset=train_file_offset,
                                                        validation_file_offset=validation_file_offset,
                                                        train_pair_num=train_pair_num,
                                                        validation_pair_num=validation_pair_num)

print('Finished.')
print('train_pair_num: {} \nvalidation_pair_num: {}'.format(train_pair_num, validation_pair_num))


# write training_tfrecord
#         offset = 0
#         for class_id in range(3):
#             tfrecord_writer = tf.python_io.TFRecordWriter(train_filename + '_' + str(offset) + '.tfrecord')
#             # offset = offset + 1
#             print(class_id)
#             person_dir = data_dir + '/' + person_list[class_id] + '/'
#             for face_id in os.listdir(person_dir):
#                 img_path = person_dir + face_id
#                 img = Image.open(img_path)
#                 img = img.resize((height, width))
#                 img = np.array(img).astype(np.uint8)
#                 with tf.Session() as sess:
#                     jpg_string = sess.run(encoded_img,
#                                           feed_dict={img_placeholder: img})
#                 example = tf.train.Example(features=tf.train.Features(feature={
#                     'image/encoded': bytes_feature(jpg_string),
#                     'image/format': bytes_feature(b'jpg'),
#                     'image/class/label': int64_feature(class_id),
#                     # 'image/height': int64_feature(height),
#                     # 'image/width': int64_feature(width)
#                 }))
#                 tfrecord_writer.write(example.SerializeToString())


# if __name__ == '__main__':
# write_labels_txt(data_dir=data_dir, lables_filename=labels_filename)
# write_to_tfrecord(data_dir=data_dir,
#                   train_rate=train_rate,
#                   height=299,
#                   width=299)
# dataset = get_split('train', dataset_dir, file_pattern)
# raw_image, label = load_batch(dataset, 1, 299, 299)
# sess = tf.Session()
# print(sess.run(raw_image))
# print(get_split('train', dataset_dir=dataset_dir, file_pattern=file_pattern))
