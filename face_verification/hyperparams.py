

class Hyperparameters:
    # random
    random_seed = 1

    # Data
    data_dir = '/home/hcshi/Class_Assignment/01_FaceVerification/webface/'
    train_filename = '/home/hcshi/Class_Assignment/01_FaceVerification/code/inception_res_v2/data/face_train'
    validation_filename = '/home/hcshi/Class_Assignment/01_FaceVerification/code/inception_res_v2/data/face_validation'
    tfrecord_dir = './data/'

    split_train_rate = 0.8
    img_height = 299
    img_width = 299
    train_pair_num = 0
    validation_pair_num = 0
    max_anchor = 10

    # Transfer learning
    checkpoint_file = 'inception_resnet_v2_2016_08_30.ckpt'

    # Train
    log_dir = './train_logs/'
    # parallel_calls = 4
    # batch_size = 32
    # prefetch_buffer_size = 256
    prefetch_buffer_size = 1
    batch_size = 1
    parallel_calls = 2
    epoch_num = 30
    initial_learning_rate = 0.0002
    learning_rate_decay_factor = 0.7
    num_epochs_before_decay = 2
    #******************Gonna change***************************
    num_batchs_per_train_epoch = 1000
    #*********************************************
    # Evaluation
    log_eval = './eval_logs/'
    eval_epoch_num = 1
    # ******************Gonna change***************************
    num_batchs_per_eval_epoch = 1000
