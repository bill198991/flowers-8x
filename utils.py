import tensorflow as tf

from tensorflow.keras import backend as K



def init_session(gpu_memory_fraction):
    K.tensorflow_backend.set_session(tensorflow_session(gpu_memory_fraction=gpu_memory_fraction))


def reset_session(gpu_memory_fraction):
    K.clear_session()
    init_session(gpu_memory_fraction)

def tensorflow_session(gpu_memory_fraction):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_fraction
    return tf.Session(config=config)