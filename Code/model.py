import numpy as np
import tensorflow as tf

# for replicability purposes
tf.random.set_seed(91195003) 
np.random.seed(91190530)
# for an easy reset backend session state 
tf.keras.backend.clear_session()

'''
Define U-Net contraction block
'''
def down_block(x, filters):
    c = tf.keras.layers.Conv2D(filters, kernel_size=3, padding='same', strides=1, activation='relu')(x)
    c = tf.keras.layers.Conv2D(filters, kernel_size=3, padding='same', strides=1, activation='relu')(c)
    mp = tf.keras.layers.MaxPool2D((2, 2), (2, 2))(c)
    return c, mp

'''
Define U-Net expansion block
'''
def up_block(x, skip, filters):
    us = tf.keras.layers.UpSampling2D((2, 2))(x)
    concat = tf.keras.layers.Concatenate()([us, skip])
    c = tf.keras.layers.Conv2D(filters, kernel_size=3, padding='same', strides=1, activation='relu')(concat)
    c = tf.keras.layers.Conv2D(filters, kernel_size=3, padding='same', strides=1, activation='relu')(c)
    return c

'''
Define U-Net bottleneck block
'''
def bottleneck(x, filters):
    c = tf.keras.layers.Conv2D(filters, kernel_size=3, padding='same', strides=1, activation='relu')(x)
    c = tf.keras.layers.Conv2D(filters, kernel_size=3, padding='same', strides=1, activation='relu')(c)
    return c

'''
Build U-Net model
'''
def UNet(image_size):
    # input layer
    inputs = tf.keras.layers.Input((image_size, image_size, 2))

    # contraction network (encoder)
    c1, mp1 = down_block(inputs, 16) # 128x128x2 -> 64x64x16 
    c2, mp2 = down_block(mp1, 32)    # 64x64x16 -> 32x32x32
    c3, mp3 = down_block(mp2, 64)    # 32x32x32 -> 16x16x64
    c4, mp4 = down_block(mp3, 128)   # 16x16x64 -> 8x8x128
    
    # bottleneck
    bn = bottleneck(mp4, 256) # 8x8x128 -> 8x8x256

    # expansion network (decoder)
    u1 = up_block(bn, c4, 128) # 8x8x256 -> 16x16x128
    u2 = up_block(u1, c3, 64)  # 16x16x128 -> 32x32x64
    u3 = up_block(u2, c2, 32)  # 32x32x64 -> 64x64x32
    u4 = up_block(u3, c1, 16)  # 64x64x32 -> 128x128x16

    # final layer
    outputs = tf.keras.layers.Conv2D(1, (1, 1), padding='same', activation='sigmoid')(u4) # 128x128x16 -> 128x128x1

    model = tf.keras.models.Model(inputs, outputs)
    return model