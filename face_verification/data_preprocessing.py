import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

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



train_rate = 0.8

# cwd = os.getcwd()
data_dir = '/home/hcshi/Class_Assignment/01_FaceVerification/webface/'
train_filename = '/home/hcshi/Class_Assignment/01_FaceVerification/code/inception_res_v2/data/face_train'
test_filename = '/home/hcshi/Class_Assignment/01_FaceVerification/code/inception_res_v2/data/face_test'
labels_filename = '/home/hcshi/Class_Assignment/01_FaceVerification/code/inception_res_v2/data/lables.txt'

def write_to_tfrecord(data_dir=data_dir, train_rate=train_rate, height=299, width=299):
    with tf.Graph().as_default():
        img_placeholder = tf.placeholder(dtype=tf.uint8)
        encoded_img = tf.image.encode_jpeg(img_placeholder)

        person_list = os.listdir(data_dir)
        person_num = len(person_list)
        train_person_num = int(person_num * train_rate)
        # write training_tfrecords
        offset = 0
        for class_id in range(train_person_num):
            if(class_id % 1000 == 0):
                tfrecord_writer = tf.python_io.TFRecordWriter(train_filename+'_'+str(offset)+'.tfrecord')
                offset = offset + 1
            print(class_id)
            person_dir = data_dir + '/' + person_list[class_id] + '/'
            for face_id in os.listdir(person_dir):
                img_path = person_dir + face_id
                img = Image.open(img_path)
                img = img.resize((height, width))
                img = np.array(img).astype(np.uint8)
                with tf.Session() as sess:
                    jpg_string = sess.run(encoded_img,
                                          feed_dict={img_placeholder: img})
                example = tf.train.Example(features=tf.train.Features(feature={
                    'image/encoded': bytes_feature(jpg_string),
                    'image/format': bytes_feature(b'jpg'),
                    'image/class/label': int64_feature(class_id),
                    # 'image/height': int64_feature(height),
                    # 'image/width': int64_feature(width)
                }))
                tfrecord_writer.write(example.SerializeToString())
        # Write testing_tfrecords
        offset = 0
        for class_id in range(train_person_num, person_num):
            if (class_id % 1000 == 0):
                tfrecord_writer = tf.python_io.TFRecordWriter(test_filename + '_' + str(offset) + '.tfrecord')
                offset = offset + 1
            print(class_id)
            person_dir = data_dir + '/' + person_list[class_id] + '/'
            print(person_dir)
            print(class_id)
            for face_id in os.listdir(person_dir):
                img_path = person_dir + face_id
                img = Image.open(img_path)
                img = img.resize((height, width))
                img = np.array(img).astype(np.uint8)
                with tf.Session() as sess:
                    jpg_string = sess.run(encoded_img,
                                          feed_dict={img_placeholder: img})
                example = tf.train.Example(features=tf.train.Features(feature={
                    'image/encoded': bytes_feature(jpg_string),
                    'image/format': bytes_feature(b'jpg'),
                    'image/class/label': int64_feature(class_id),
                    # 'image/height': int64_feature(height),
                    # 'image/width': int64_feature(width)
                }))
                tfrecord_writer.write(example.SerializeToString())


def write_labels_txt(data_dir, lables_filename):
    class_to_person_id = {}
    person_list = os.listdir(data_dir)
    person_num = len(person_list)
    for class_id in range(person_num):
        person_id = person_list[class_id]
        class_to_person_id[class_id] = person_id

    f = open(labels_filename, 'w')
    for class_id, person_id in class_to_person_id.items():
        f.writelines(str(class_id)+':'+person_id+'\n')
    f.close()

def read_and_decode(filename):
    pass


if __name__ == '__main__':
    write_labels_txt(data_dir=data_dir, lables_filename=labels_filename)
    write_to_tfrecord(data_dir=data_dir,
                      train_rate=train_rate,
                      height=299,
                      width=299)



            # print(type(label))
            # print(img.shape)
            # print(type(img))
            # plt.imshow(img)
            # plt.show()



# for person_id in os.listdir(data_dir):
#     person_dir = data_dir + "/" + person_id + "/"
#     person_face_num = len(os.listdir(person_dir))
#     print(person_face_num)
#     for face_id in os.listdir(person_dir):
#         img_path = person_dir + face_id
#         # img_row = tf.gfile.FastGFile(img_path, 'rb').read()

