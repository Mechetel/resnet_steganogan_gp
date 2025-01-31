import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, BatchNormalization, AveragePooling2D, Conv2D, LeakyReLU
from tensorflow.keras.layers import Add, Concatenate, Lambda, Activation, Layer, Flatten
from tensorflow.keras.activations import sigmoid, tanh


def BasicEncoder(D):
    """
    The BasicEncoder module takes an cover image and a data tensor and combines
    them into a steganographic image.
    Input: (N, 3, H, W), (N, H, W, D)
    Output: (N, 3, H, W)
    """
    Cover = Input(shape=(None, None, 3), name=f'cover_image')
    Message = Input(shape=(None, None, D), name=f'message_data')

    a = Conv2D(32, kernel_size=3, padding='same', activation=LeakyReLU(), name='a_conv')(Cover)
    a = BatchNormalization(name='a_normalize')(a)

    b_concatenate = Concatenate(name='b_concatenate')([a, Message])
    b = Conv2D(32, kernel_size=3, padding='same', activation=LeakyReLU(), name='b_conv')(b_concatenate)
    b = BatchNormalization(name='b_normalize')(b)

    c = Conv2D(32, kernel_size=3, padding='same', activation=LeakyReLU(), name='c_conv')(b)
    c = BatchNormalization(name='c_normalize')(c)

    Encoder_d = Conv2D(3, kernel_size=3, padding='same', activation=tanh, name='Encoder_conv_tanh')(c)
    
    model = Model(inputs=[Cover, Message], outputs=Encoder_d, name='ResnetSteganoGAN_basic_encoder')
    return model

def ResidualEncoder(D):
    """
    The ResidualEncoder module takes an cover image and a data tensor and combines
    them into a steganographic image.
    Input: (N, H, W, 3), (N, H, W, D)
    Output: (N, H, W, 3)
    """
    Cover = Input(shape=(None, None, 3), name=f'cover_image')
    Message = Input(shape=(None, None, D), name=f'message_data')

    a = Conv2D(32, kernel_size=3, padding='same', activation=LeakyReLU(), name='a_conv')(Cover)
    a = BatchNormalization(name='a_normalize')(a)

    b_concatenate = Concatenate(name='b_concatenate')([a, Message])
    b = Conv2D(32, kernel_size=3, padding='same', activation=LeakyReLU(), name='b_conv')(b_concatenate)
    b = BatchNormalization(name='b_normalize')(b)

    c = Conv2D(32, kernel_size=3, padding='same', activation=LeakyReLU(), name='c_conv')(b)
    c = BatchNormalization(name='c_normalize')(c)

    d = Conv2D(3, kernel_size=3, padding='same', name='d_conv')(c)
    
    Encoder_d = Add(name='add_C_d')([Cover, d])
    Encoder_d = Activation(tanh, name='Encoder_activation_tanh')(Encoder_d)
    
    return Model(inputs=[Cover, Message], outputs=Encoder_d, name='ResnetSteganoGAN_residual_encoder')

def DenseEncoder(D):
    """
    The DenseEncoder module takes an cover image and a data tensor and combines
    them into a steganographic image.
    Input: (N, H, W, 3), (N, H, W, D)
    Output: (N, H, W, 3)
    """
    Cover = Input(shape=(None, None, 3), name=f'cover_image')
    Message = Input(shape=(None, None, D), name=f'message_data')

    a = Conv2D(32, kernel_size=3, padding='same', activation=LeakyReLU(), name='a_conv')(Cover)
    a = BatchNormalization(name='a_normalize')(a)

    b_concatenate = Concatenate(name='b_concatenate')([a, Message])
    b = Conv2D(32, kernel_size=3, padding='same', activation=LeakyReLU(), name='b_conv')(b_concatenate)
    b = BatchNormalization(name='b_normalize')(b)

    c_concatenate = Concatenate(name='c_concatenate')([a, b, Message])
    c = Conv2D(32, kernel_size=3, padding='same', activation=LeakyReLU(), name='c_conv')(c_concatenate)
    c = BatchNormalization(name='c_normalize')(c)

    d_concatenate = Concatenate(name='d_concatenate')([a, b, c, Message])
    d = Conv2D(3, kernel_size=3, padding='same', name='d_conv')(d_concatenate)

    Encoder_d = Add(name='add_C_d')([Cover, d])
    Encoder_d = Activation(tanh, name='Encoder_activation_tanh')(Encoder_d)
    
    return Model(inputs=[Cover, Message], outputs=Encoder_d, name='ResnetSteganoGAN_dense_encoder')

def BasicDecoder(D):
    """
    The BasicDecoder module takes an steganographic image and attempts to decode
    the embedded data tensor.
    Input: (N, H, W, 3)
    Output: (N, H, W, D)
    """
    Cover = Input(shape=(None, None, 3), name=f'cover_image')
    
    a = Conv2D(32, kernel_size=3, padding='same', activation=LeakyReLU(), name='a_conv')(Cover)
    a = BatchNormalization(name='a_normalize')(a)

    b = Conv2D(32, kernel_size=3, padding='same', activation=LeakyReLU(), name='b_conv')(a)
    b = BatchNormalization(name='b_normalize')(b)

    c = Conv2D(32, kernel_size=3, padding='same', activation=LeakyReLU(), name='c_conv')(b)
    c = BatchNormalization(name='c_normalize')(c)

    Decoder = Conv2D(D, kernel_size=3, padding='same', activation=sigmoid, name='Decoder_conv_sigmoid')(c)

    return Model(inputs=Cover, outputs=Decoder, name='ResnetSteganoGAN_basic_decoder')

def DenseDecoder(D):
    """
    The DenseDecoder module takes an steganographic image and attempts to decode
    the embedded data tensor.
    Input: (N, H, W, 3)
    Output: (N, H, W, D)
    """
    Cover = Input(shape=(None, None, 3), name=f'cover_image')
    
    a = Conv2D(32, kernel_size=3, padding='same', activation=LeakyReLU(), name='a_conv')(Cover)
    a = BatchNormalization(name='a_normalize')(a)

    b = Conv2D(32, kernel_size=3, padding='same', activation=LeakyReLU(), name='b_conv')(a)
    b = BatchNormalization(name='b_normalize')(b)

    c_concatenate = Concatenate(name='c_concatenate')([a, b])
    c = Conv2D(32, kernel_size=3, padding='same', activation=LeakyReLU(), name='c_conv')(c_concatenate)
    c = BatchNormalization(name='c_normalize')(c)

    Decoder_concatenate = Concatenate(name='Decoder_concatenate')([a, b, c])
    Decoder = Conv2D(D, kernel_size=3, padding='same', activation=sigmoid, name='Decoder_conv_sigmoid')(Decoder_concatenate)

    return Model(inputs=Cover, outputs=Decoder, name='ResnetSteganoGAN_dense_decoder')

def Critic():
    """
    The Critic module takes an image and predicts whether it is a cover
    image or a steganographic image (N, 1).
    Input: (N, H, W, 3)
    Output: (N, 1)
    """
    Stego = Input(shape=(None, None, 3), name=f'stego_image')

    a = Conv2D(32, kernel_size=3, padding='same', activation=LeakyReLU(), name='a_conv_1')(Stego)
    a = Conv2D(32, kernel_size=3, padding='same', activation=LeakyReLU(), name='a_conv_2')(a)
    a = Conv2D(32, kernel_size=3, padding='same', activation=LeakyReLU(), name='a_conv_3')(a)
    score = Conv2D(1, kernel_size=3, padding='same', name='a_conv_4')(a)

    return Model(inputs=Stego, outputs=score, name='ResnetSteganoGAN_critic')