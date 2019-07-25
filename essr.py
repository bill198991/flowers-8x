import tensorflow as tf

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Add, Conv2D, Input, Lambda, Activation,UpSampling2D
from tensorflow.keras.models import Model

from common import SubpixelConv2D, Normalization, Denormalization


def essr(scale = 8, num_filters = 32, num_res_blocks=3 ,res_block_expansion = 6, res_block_scaling=None):
    x_in = Input(shape=(None, None, 3))
    m = Conv2D(num_filters, 3, padding='same')(x_in)

    for i in range(num_res_blocks):
        m = res_block(m, num_filters, res_block_expansion, kernel_size=3, scaling=res_block_scaling)
    m = Conv2D(3 * scale ** 2, 3, padding='same', name=f'conv2d_main_scale_{scale}')(m)
    m = SubpixelConv2D(scale)(m)
    

    # skip branch
    s = UpSampling2D((8,8))(x_in)
    s = Conv2D(num_filters, 5, padding='same', name=f'conv2d_skip_scale_{scale}')(s)
    s = Activation('relu')(s)
    s = Conv2D(num_filters, 3, padding='same', name=f'conv2d_skip_scale_{scale}_2')(s)
    s = Activation('relu')(s)
    s = Conv2D(3,1,padding='same',name=f'conv2d_skip_scale_{scale}_3')(s)
 
    x = Add()([m, s])

    #x = Denormalization()(x)

    return Model(x_in, x, name="essr-b")


def res_block(x_in, num_filters , expansion, kernel_size, scaling):
    x = Conv2D(num_filters * expansion, kernel_size, padding='same')(x_in)
    x = Activation('relu')(x)
    x = Conv2D(num_filters, kernel_size, padding='same')(x)
    x = Add()([x_in, x])
    if scaling:
        x = Lambda(lambda t: t * scaling)(x)
    return x


