

class Hyperparameters:
    # random
    random_seed = 1

    # Data
    data_dir = '/home/hcshi/Class_Assignment/01_FaceVerification/webface/'
    train_filename = '/home/hcshi/Class_Assignment/01_FaceVerification/code/inception_res_v2/fake_data/face_train'
    validation_filename = '/home/hcshi/Class_Assignment/01_FaceVerification/code/inception_res_v2/fake_data/face_validation'
    tfrecord_dir = '/home/hcshi/Class_Assignment/01_FaceVerification/code/inception_res_v2/fake_data/'

    split_train_rate = 0.8
    img_height = 299
    img_width = 299
    train_pair_num = 0
    validation_pair_num = 0
    max_anchor = 10

    parallel_calls = 4
    batch_size = 32
