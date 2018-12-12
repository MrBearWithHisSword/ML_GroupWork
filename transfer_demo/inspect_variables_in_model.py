import tensorflow as tf
from tensorflow.python.tools import inspect_checkpoint

ckpt_dir = '/tmp/regression_model/model.ckpt'
inspect_checkpoint.print_tensors_in_checkpoint_file(ckpt_dir, tensor_name=None, all_tensors=True, all_tensor_names=True)
