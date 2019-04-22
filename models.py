import tensorflow.keras.layers as layers
from tensorflow.keras.initializers import Constant
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16

import tensorflow.keras.layers as layers
from tensorflow.keras.initializers import Constant
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models, initializers, activations


def conv_block(filters, kernal, padding='same', stride=1, activation=None, add_bn=True, drop_rate=0):
    block = [layers.Conv2D(filters, kernal, activation=None, padding='same', strides=stride)]
    if add_bn:
        block.append(layers.BatchNormalization())
    if activation is not None:
        block.append(activation)
    if drop_rate > 0:
        block.append(layers.SpatialDropout2D(rate=drop_rate))
    return models.Sequential(block)


def build_model(height, width, dropout=0.4):
    
    relu = layers.ReLU()
    leaky_relu = layers.LeakyReLU(0.2)

    inputs = layers.Input(shape=(height, width))
    reshape = layers.Reshape((height, width,1))(inputs)
#     bn1 = layers.BatchNormalization()(reshape)
    b0_out = layers.Lambda(lambda x: x/127.5 - 1)(reshape)

#     conv1 = layers.Conv2D(16, kernel_size = (1,1), padding = 'same', activation = 'relu')(b0_out)
    conv1 = layers.Conv2D(3, kernel_size = (1,1), padding = 'same', activation = 'relu')(b0_out)
    
    vgg_model = VGG16(include_top=False, input_shape=(height, width, 3))
    for layer in vgg_model.layers:
        layer.trainable = False
    block5 = vgg_model.get_layer("block5_conv3").output  # 16, 16, 512
    block4 = vgg_model.get_layer("block4_conv3").output  # 32, 32, 512
    block3 = vgg_model.get_layer("block3_conv3").output  # 64, 64, 256
    block2 = vgg_model.get_layer("block2_conv2").output  # 128, 128, 128
    block1 = vgg_model.get_layer("block1_conv2").output  # 256, 256, 64
    vgg_outputs = [block5, block4, block3, block2, block1]
    vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_outputs)
    block5_out, block4_out, block3_out, block2_out, block1_out = vgg_model(conv1)
    
    # 16, 16
    up4 = layers.UpSampling2D(size=(2, 2))(block5_out)
    up4 = conv_block(512, (3, 3), activation=relu, padding='same', stride=1, add_bn=True, drop_rate=dropout)(up4)
    up4 = layers.concatenate([block4_out, up4], axis=3)    
    up4 = conv_block(512, (3, 3), activation=relu, padding='same', stride=1, add_bn=True, drop_rate=dropout)(up4)
    
    # 32, 32
    up3 = layers.UpSampling2D(size=(2, 2))(up4)
    up3 = conv_block(256, (3, 3), activation=relu, padding='same', stride=1, add_bn=True, drop_rate=dropout/2)(up3)
    up3 = layers.concatenate([block3_out, up3], axis=3)
    up3 = conv_block(256, (3, 3), activation=relu, padding='same', stride=1, add_bn=True, drop_rate=dropout/2)(up3)
   
    # 64, 64
    up2 = layers.UpSampling2D(size=(2, 2))(up3)
    up2 = conv_block(128, (3, 3), activation=relu, padding='same', stride=1, add_bn=True, drop_rate=dropout/3)(up2)
    up3 = layers.concatenate([block2_out, up2], axis=3)
    up2 = conv_block(128, (3, 3), activation=relu, padding='same', stride=1, add_bn=True, drop_rate=dropout/3)(up2)
    
    # 128, 128
    up1 = layers.UpSampling2D(size=(2, 2))(up2)
    up1 = conv_block(64, (3, 3), activation=leaky_relu, padding='same', stride=1, add_bn=True, drop_rate=dropout/4)(up1)
    up1 = layers.concatenate([block1_out, up1], axis=3)
    up1 = conv_block(64, (3, 3), activation=leaky_relu, padding='same', stride=1, add_bn=True, drop_rate=dropout/4)(up1)
    outputs = layers.Conv2D(3, (1, 1), activation=None, padding='same')(up1)
    outputs = layers.Activation('tanh')(outputs)
    outputs = layers.Lambda(lambda x: (x*127.5) + 127.5)(outputs)

#     outputs = layers.BatchNormalization(beta_initializer=Constant(100.))(outputs)

    return Model(inputs=inputs, outputs=outputs), vgg_model


# def build_model(height, width, dropout=0.4):
    
#     relu = layers.ReLU()
#     leaky_relu = layers.LeakyReLU(0.2)

#     inputs = layers.Input(shape=(height, width))
#     reshape = layers.Reshape((height, width,1))(inputs)
# #     bn1 = layers.BatchNormalization()(reshape)
#     b0_out = layers.Lambda(lambda x: x/255. - 0.5)(reshape)

# #     conv1 = layers.Conv2D(16, kernel_size = (1,1), padding = 'same', activation = 'relu')(b0_out)
#     conv1 = layers.Conv2D(3, kernel_size = (1,1), padding = 'same', activation = 'relu')(b0_out)
#     vgg_model = VGG16(include_top=False, input_shape=(height, width, 3))
#     vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.get_layer("block5_conv3").output)
#     for layer in vgg_model.layers:
#         layer.trainable = False
#     vgg_out = vgg_model(conv1)
#     block1 = vgg_model.get_layer("block1_conv2").output  # 256, 256, 64
#     block2 = vgg_model.get_layer("block2_conv2").output  # 128, 128, 128
#     block3 = vgg_model.get_layer("block3_conv3").output  # 64, 64, 256
#     block4 = vgg_model.get_layer("block4_conv3").output  # 32, 32, 512
# #     block5 = vgg_model.get_layer("block5_conv3").output  # 16, 16, 512
    
        
# #     up4 = layers.UpSampling2D(size=(2, 2))(vgg_out)
# #     up4 = conv_block(256, (3, 3), activation=relu, padding='same', stride=1, add_bn=True, drop_rate=dropout)(up4)
# #     up4 = conv_block(256, (3, 3), activation=relu, padding='same', stride=1, add_bn=True, drop_rate=dropout)(up4)
    
#     up3 = layers.UpSampling2D(size=(2, 2))(up4)
#     up3 = conv_block(128, (3, 3), activation=relu, padding='same', stride=1, add_bn=True, drop_rate=dropout/2)(up3)
#     up3 = conv_block(128, (3, 3), activation=relu, padding='same', stride=1, add_bn=True, drop_rate=dropout/2)(up3)
   
#     up2 = layers.UpSampling2D(size=(2, 2))(up3)
#     up2 = conv_block(64, (3, 3), activation=relu, padding='same', stride=1, add_bn=True, drop_rate=dropout/3)(up2)
#     up2 = conv_block(64, (3, 3), activation=relu, padding='same', stride=1, add_bn=True, drop_rate=dropout/3)(up2)
    
#     up1 = layers.UpSampling2D(size=(2, 2))(up2)
#     up1 = conv_block(32, (3, 3), activation=leaky_relu, padding='same', stride=1, add_bn=True, drop_rate=dropout/4)(up1)
#     up1 = layers.concatenate([conv1, up1], axis=3)
#     up1 = conv_block(32, (3, 3), activation=leaky_relu, padding='same', stride=1, add_bn=True, drop_rate=dropout/4)(up1)
#     outputs = layers.Conv2D(3, (1, 1), activation=None, padding='same')(up1)
#     outputs = layers.Activation('tanh')(outputs)
#     outputs = layers.Lambda(lambda x: (x*127.5) + 127.5)(outputs)

# #     outputs = layers.BatchNormalization(beta_initializer=Constant(100.))(outputs)

#     return Model(inputs=inputs, outputs=outputs), vgg_model