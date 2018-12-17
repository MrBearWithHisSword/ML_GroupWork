

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

    # Transfer learning
    checkpoint_file = './inception_resnet_v2_2016_08_30.ckpt'

    # Train
    log_dir = './logs/'
    parallel_calls = 4
    batch_size = 32
    prefetch_buffer_size = 32
    epoch_num = 20
    initial_learning_rate = 0.0002
    learning_rate_decay_factor = 0.7
    num_epochs_before_decay = 2


