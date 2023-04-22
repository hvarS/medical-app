"""
Attention UNet architecture for multitask learning - Segmentatoin & Localization
"""
from tensorflow.keras import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50

# NOTE: IMAGE CONVENTION IS (W, H, C)

def conv_block(inputs, ch_out, trainit=True, dropout_rate=0.0, block_id=-1):
    prefix = 'conv_' + str(block_id) + '_'
    x = Conv2D(ch_out, kernel_size=3, strides=(1, 1), padding='same', name=prefix+'c1', trainable=trainit)(inputs)
    x = Dropout(dropout_rate, name=prefix+'d1', trainable=trainit)(x)
    x = BatchNormalization(name=prefix+'bn1', trainable=trainit)(x)
    x = ReLU(name=prefix+'relu1', trainable=trainit)(x)
    x = Conv2D(ch_out, kernel_size=3, strides=(1, 1), padding='same', name=prefix+'x2', trainable=trainit)(x)
    x = Dropout(dropout_rate, name=prefix+'d2', trainable=trainit)(x)
    x = BatchNormalization(name=prefix+'bn2', trainable=trainit)(x)
    x = ReLU(name=prefix+'relu2', trainable=trainit)(x)
    return x


def deconv_block(inputs, ch_out, trainit=True, dropout_rate=0.0, block_id=-1):
    prefix = 'deconv_' + str(block_id) + '_'
    x = Conv2DTranspose(ch_out, kernel_size=2, strides=(2, 2), padding='valid', name=prefix+'dc', trainable=trainit)(inputs)
    x = Dropout(dropout_rate, name=prefix+'d', trainable=trainit)(x)
    return x


def attention_block(inputs, inputs_skip, ch_mid, trainit=True, dropout_rate=0.0, block_id=-1):
    prefix = 'att_' + str(block_id) + '_'
    x = Conv2D(ch_mid, kernel_size=1, padding='same', name=prefix+'cx', trainable=trainit)(inputs)
    x = Dropout(dropout_rate, name=prefix+'dx', trainable=trainit)(x)
    x = BatchNormalization(name=prefix+'bnx', trainable=trainit)(x)

    g = Conv2D(ch_mid, kernel_size=1, padding='same', name=prefix+'cs', trainable=trainit)(inputs_skip)
    g = Dropout(dropout_rate, name=prefix+'ds', trainable=trainit)(g)
    g = BatchNormalization(name=prefix+'bns', trainable=trainit)(g)

    psi = Add(name=prefix+'add', trainable=trainit)([g, x])
    psi = ReLU(name=prefix+'relu', trainable=trainit)(psi)
    psi = Conv2D(1, kernel_size=1, padding='same', name=prefix+'c', trainable=trainit)(psi)
    psi = BatchNormalization(name=prefix+'bn', trainable=trainit)(psi)
    psi = Activation('sigmoid', name=prefix+'sig', trainable=trainit)(psi)

    ret = Multiply(name=prefix+'mul', trainable=trainit)([inputs, psi])
    return ret


def attention_activations(inputs, inputs_skip, ch_mid, trainit=True, dropout_rate=0.0, block_id=-1):
    prefix = 'att_' + str(block_id) + '_'
    x = Conv2D(ch_mid, kernel_size=1, padding='same', name=prefix+'cx', trainable=trainit)(inputs)
    x = Dropout(dropout_rate, name=prefix+'dx', trainable=trainit)(x)
    x = BatchNormalization(name=prefix+'bnx', trainable=trainit)(x)

    g = Conv2D(ch_mid, kernel_size=1, padding='same', name=prefix+'cs', trainable=trainit)(inputs_skip)
    g = Dropout(dropout_rate, name=prefix+'ds', trainable=trainit)(g)
    g = BatchNormalization(name=prefix+'bns', trainable=trainit)(g)

    psi = Add(name=prefix+'add', trainable=trainit)([g, x])
    psi = ReLU(name=prefix+'relu', trainable=trainit)(psi)
    psi = Conv2D(1, kernel_size=1, padding='same', name=prefix+'c', trainable=trainit)(psi)
    psi = BatchNormalization(name=prefix+'bn', trainable=trainit)(psi)
    psi = Activation('sigmoid', name=prefix+'sig', trainable=trainit)(psi)

    ret = Multiply(name=prefix+'mul', trainable=trainit)([inputs, psi])
    return ret, psi


def attention_unet(input_shape, out_channels, multiplier, freeze_segmentor, use_constraints, dropout_rate):
    """
    input_shape = (W, H) -- RGB Image
    out_channels = number of output segmentation masks
    multiplier = the scale by which the channels of the network can be increased
    dim1 is the number of values per IR box and dim2...4 are for each tissue parts
    returns a TF Keras model for attention UNet
    """
    drate = dropout_rate
    scale = int(multiplier)
    trainit = not freeze_segmentor

    image_input = Input(shape=(input_shape[0], input_shape[1], 3), name='input')
    mask_inputs = Input(shape=(input_shape[0], input_shape[1], 3), name='input_masks')

    # ENCODER Network
    x1 = conv_block(inputs=image_input, ch_out=8*scale, trainit=trainit, dropout_rate=drate, block_id=1)
    
    x2 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x1)
    x2 = conv_block(x2, 16*scale, trainit, drate, block_id=2)

    x3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x2)
    x3 = conv_block(x3, 32*scale, trainit, drate, block_id=3)

    x4 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x3)
    x4 = conv_block(x4, 64*scale, trainit, drate, block_id=4)

    x5 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x4)
    x5 = conv_block(x5, 128*scale, trainit, drate, block_id=5)

    # DECODER Network
    d5 = deconv_block(inputs=x5, ch_out=64*scale, trainit=trainit, dropout_rate=drate, block_id=5)
    s4 = attention_block(inputs=d5, inputs_skip=x4, ch_mid=32*scale, trainit=trainit, dropout_rate=drate, block_id=5)
    d5 = Concatenate(axis=-1)([d5, s4])
    d5 = conv_block(d5, 64*scale, trainit, drate, block_id=6)

    d4 = deconv_block(inputs=d5, ch_out=32*scale, trainit=trainit, dropout_rate=drate, block_id=4)
    s3 = attention_block(inputs=d4, inputs_skip=x3, ch_mid=16*scale, trainit=trainit, dropout_rate=drate, block_id=4)
    d4 = Concatenate(axis=-1)([d4, s3])
    d4 = conv_block(d4, 32*scale, trainit, drate, block_id=7)

    d3 = deconv_block(inputs=d4, ch_out=16*scale, trainit=trainit, dropout_rate=drate, block_id=3)
    s2 = attention_block(inputs=d3, inputs_skip=x2, ch_mid=8*scale, trainit=trainit, dropout_rate=drate, block_id=3)
    d3 = Concatenate(axis=-1)([d3, s2])
    d3 = conv_block(d3, 16*scale, trainit, drate, block_id=8)

    d2 = deconv_block(inputs=d3, ch_out=8*scale, trainit=trainit, dropout_rate=drate, block_id=2)
    s1 = attention_block(inputs=d2, inputs_skip=x1, ch_mid=4*scale, trainit=trainit, dropout_rate=drate, block_id=2)
    d2 = Concatenate(axis=-1)([d2, s1])
    d2 = conv_block(d2, 8*scale, trainit, drate, block_id=9)

    d1 = Conv2D(filters=out_channels, kernel_size=1, strides=(1, 1), padding='valid', name='conv1x1', trainable=True)(d2)
    d1 = Activation('sigmoid', name='out')(d1)

    # Model the Attention-Unet network
    att_unet = Model(inputs=image_input, outputs=d1, name='attention_unet')
    
    # Tap features and localization
    f1 = conv_block(d2, 16*scale, dropout_rate=drate, block_id=10)

    f2 = MaxPool2D()(f1)
    c2 = attention_block(f2, d3, 8*scale, dropout_rate=drate, block_id=11)
    f2 = Concatenate(axis=-1)([c2, f2])
    f2 = conv_block(f2, 32*scale, dropout_rate=drate, block_id=11)

    f3 = MaxPool2D()(f2)
    c3 = attention_block(f3, d4, 16*scale, dropout_rate=drate, block_id=12)
    f3 = Concatenate(axis=-1)([c3, f3])
    f3 = conv_block(f3, 64*scale, dropout_rate=drate, block_id=12)

    f4 = MaxPool2D()(f3)
    c4 = attention_block(f4, d5, 32*scale, dropout_rate=drate, block_id=13)
    f4 = Concatenate(axis=-1)([c4, f4])
    f4 = conv_block(f4, 128*scale, dropout_rate=drate, block_id=13)

    f5 = MaxPool2D()(f4)
    c5 = attention_block(f5, x5, 64*scale, dropout_rate=drate, block_id=14)
    f5 = Concatenate(axis=-1)([c5, f5])
    f5 = conv_block(f5, 256*scale, dropout_rate=drate, block_id=14)

    grid_out = Conv2D(5, kernel_size=1, strides=(1, 1), padding='same', name='final_conv')(f5)
    labels_out = Lambda(lambda x: x[..., -1], name='lambda0')(grid_out)
    labels_out = Reshape(list(labels_out.shape[1:]) + [1], name='reshape0')(labels_out)
    labels_out = Activation('sigmoid', name='sigmoid0')(labels_out)
    box_out = Lambda(lambda x: x[..., 0:4], name='lambda1')(grid_out)
    grid_out = Concatenate(axis=-1, name='concat0')([box_out, labels_out])
    # grid_out = Activation('relu', name='bounding_boxes')(grid_out)
    # classification = Lambda(lambda x: x[..., -1])(grid_out)
    # classification = Activation('sigmoid', name='classification')(classification)
    # regression = Lambda(lambda x: x[..., 0:4])(grid_out)
    # regression = Activation('relu', name='regression')(regression)

    # If using constraints, pad the output with input masks
    if use_constraints:
        morphed_masks = Reshape([16, 13, 768])(mask_inputs)
        grid_out = Concatenate(axis=-1, name='concat1')([grid_out, morphed_masks])

    # Model the localizer network
    # localizer = Model(inputs=image_input, outputs=[classification, regression], name='localizer')
    localizer = Model(inputs=[image_input, mask_inputs], outputs=[grid_out], name='localizer')

    # Model the combined network
    # anticeliac = Model(inputs=image_input, outputs=[d1, classification, regression], name='anti-celiac')
    anticeliac = Model(inputs=[image_input, mask_inputs], outputs=[d1, grid_out], name='anti_celiac')

    return att_unet, localizer, anticeliac, mask_inputs



def attention_unet_refined(image_size, out_channels, multiplier, freeze_encoder, freeze_decoder, use_constraints, dropout_rate):
    """
    image_size = (W, H) -- RGB Image
    out_channels = number of output segmentation masks
    multiplier = the scale by which the channels of the network can be increased
    dim1 is the number of values per IR box and dim2...4 are for each tissue parts
    returns a TF Keras model for attention UNet
    """
    drate = dropout_rate
    scale = int(multiplier)
    train_enc = not freeze_encoder
    train_dec = not freeze_decoder

    image_input = Input(shape=(image_size[0], image_size[1], 3), name='input')
    mask_inputs = Input(shape=(image_size[0], image_size[1], 3), name='input_masks')
    
    # ENCODER Network
    x1 = conv_block(inputs=image_input, ch_out=8*scale, trainit=train_enc, dropout_rate=drate, block_id=1)
    
    x2 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x1)
    x2 = conv_block(x2, 16*scale, train_enc, drate, block_id=2)

    x3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x2)
    x3 = conv_block(x3, 32*scale, train_enc, drate, block_id=3)

    x4 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x3)
    x4 = conv_block(x4, 64*scale, train_enc, drate, block_id=4)

    x5 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x4)
    x5 = conv_block(x5, 128*scale, train_enc, drate, block_id=5)

    # DECODER Network
    d5 = deconv_block(inputs=x5, ch_out=64*scale, trainit=train_dec, dropout_rate=drate, block_id=5)
    s4 = attention_block(inputs=d5, inputs_skip=x4, ch_mid=32*scale, trainit=train_dec, dropout_rate=drate, block_id=5)
    d5 = Concatenate(axis=-1)([d5, s4])
    d5 = conv_block(d5, 64*scale, train_dec, drate, block_id=6)

    d4 = deconv_block(inputs=d5, ch_out=32*scale, trainit=train_dec, dropout_rate=drate, block_id=4)
    s3 = attention_block(inputs=d4, inputs_skip=x3, ch_mid=16*scale, trainit=train_dec, dropout_rate=drate, block_id=4)
    d4 = Concatenate(axis=-1)([d4, s3])
    d4 = conv_block(d4, 32*scale, train_dec, drate, block_id=7)

    d3 = deconv_block(inputs=d4, ch_out=16*scale, trainit=train_dec, dropout_rate=drate, block_id=3)
    s2 = attention_block(inputs=d3, inputs_skip=x2, ch_mid=8*scale, trainit=train_dec, dropout_rate=drate, block_id=3)
    d3 = Concatenate(axis=-1)([d3, s2])
    d3 = conv_block(d3, 16*scale, train_dec, drate, block_id=8)

    d2 = deconv_block(inputs=d3, ch_out=8*scale, trainit=train_dec, dropout_rate=drate, block_id=2)
    s1 = attention_block(inputs=d2, inputs_skip=x1, ch_mid=4*scale, trainit=train_dec, dropout_rate=drate, block_id=2)
    d2 = Concatenate(axis=-1)([d2, s1])
    d2 = conv_block(d2, 8*scale, train_dec, drate, block_id=9)

    d1 = Conv2D(filters=out_channels, kernel_size=1, strides=(1, 1), padding='valid', name='conv1x1', trainable=train_dec)(d2)
    d1 = Activation('sigmoid', name='out')(d1)

    # Model the Attention-Unet network
    encoder = Model(inputs=image_input, outputs=x5, name='attention_unet_encoder')
    att_unet = Model(inputs=image_input, outputs=d1, name='attention_unet')
    
    # Tap features and localization
    f1 = conv_block(d2, 8*scale, dropout_rate=drate, block_id=10)

    f2 = MaxPool2D()(f1)
    c2 = conv_block(d3, 8*scale, dropout_rate=drate, block_id=11)
    f2 = Concatenate(axis=-1)([c2, f2])
    f2 = conv_block(f2, 16*scale, dropout_rate=drate, block_id=12)

    f3 = MaxPool2D()(f2)
    c3 = conv_block(d4, 16*scale, dropout_rate=drate, block_id=13)
    f3 = Concatenate(axis=-1)([c3, f3])
    f3 = conv_block(f3, 32*scale, dropout_rate=drate, block_id=14)

    f4 = MaxPool2D()(f3)
    c4 = conv_block(d5, 32*scale, dropout_rate=drate, block_id=15)
    f4 = Concatenate(axis=-1)([c4, f4])
    f4 = conv_block(f4, 64*scale, dropout_rate=drate, block_id=16)

    f5 = MaxPool2D()(f4)
    c5 = conv_block(x5, 64*scale, dropout_rate=drate, block_id=17)
    f5 = Concatenate(axis=-1)([c5, f5])
    f5 = conv_block(f5, 128*scale, dropout_rate=drate, block_id=18)

    grid_out = Conv2D(5, kernel_size=1, strides=(1, 1), padding='same', name='final_conv')(f5)
    labels_out = Lambda(lambda x: x[..., -1], name='lambda0')(grid_out)
    labels_out = Reshape(list(labels_out.shape[1:]) + [1], name='reshape0')(labels_out)
    labels_out = Activation('sigmoid', name='sigmoid0')(labels_out)
    box_out = Lambda(lambda x: x[..., 0:4], name='lambda1')(grid_out)
    grid_out = Concatenate(axis=-1, name='concat0')([box_out, labels_out])
    # grid_out = Activation('relu', name='bounding_boxes')(grid_out)
    # classification = Lambda(lambda x: x[..., -1])(grid_out)
    # classification = Activation('sigmoid', name='classification')(classification)
    # regression = Lambda(lambda x: x[..., 0:4])(grid_out)
    # regression = Activation('relu', name='regression')(regression)

    # If using constraints, pad the output with input masks
    if use_constraints:
        morphed_masks = Reshape([16, 13, 768])(mask_inputs)
        grid_out = Concatenate(axis=-1, name='concat1')([grid_out, morphed_masks])

    # Model the localizer network
    # localizer = Model(inputs=image_input, outputs=[classification, regression], name='localizer')
    localizer = Model(inputs=[image_input, mask_inputs], outputs=[grid_out], name='localizer')

    # Model the combined network
    # anticeliac = Model(inputs=image_input, outputs=[d1, classification, regression], name='anti-celiac')
    anticeliac = Model(inputs=[image_input, mask_inputs], outputs=[d1, grid_out], name='anti_celiac')

    return encoder, att_unet, localizer, anticeliac, mask_inputs



def attention_unet_resnet50(input_shape, out_channels, freeze_encoder, encoder_weights, freeze_decoder, dropout_rate):
    drate = dropout_rate
    scale = 16
    train_enc = not freeze_encoder
    train_dec = not freeze_decoder

    image_input = Input(shape=input_shape, name='input')
    
    # Resnet50 backbone
    backbone = ResNet50(include_top=False, weights=encoder_weights, input_tensor=image_input, input_shape=input_shape)

    # Collect the transition layers -- endpoints of each block
    out_ids = []
    for i in range(1, len(backbone.layers)):
        if backbone.layers[i].name.split('_')[0][-1] != backbone.layers[i-1].name.split('_')[0][-1]:
            out_ids.append(i-1)
    out_ids.append(len(backbone.layers) - 1)
    out_ids[1] -= 2  # Because the output after conv and before pool1 is desired
    print(out_ids)

    if not train_enc:
        # Freeze encoder
        for i in range(len(backbone.layers)):
            backbone.layers[i].trainable = False

    # ENCODER Network
    x1 = backbone.layers[out_ids[1]].output    # (W/2, H/2, 128)
    x2 = backbone.layers[out_ids[2]].output    # (W/4, H/4, 256)
    x3 = backbone.layers[out_ids[3]].output    # (W/8, H/8, 512)
    x4 = backbone.layers[out_ids[4]].output    # (W/16, H/16, 1024)
    x5 = backbone.layers[out_ids[5]].output    # (W/32, H/32, 2048)

    # DECODER Network
    d5 = deconv_block(inputs=x5, ch_out=64*scale, trainit=train_dec, dropout_rate=drate, block_id=5)
    s4 = attention_block(inputs=d5, inputs_skip=x4, ch_mid=32*scale, trainit=train_dec, dropout_rate=drate, block_id=5)
    d5 = Concatenate(axis=-1)([d5, s4])
    d5 = conv_block(d5, 64*scale, train_dec, drate, block_id=6)
        # (W/16, H/16, 1024)
    d4 = deconv_block(inputs=d5, ch_out=32*scale, trainit=train_dec, dropout_rate=drate, block_id=4)
    s3 = attention_block(inputs=d4, inputs_skip=x3, ch_mid=16*scale, trainit=train_dec, dropout_rate=drate, block_id=4)
    d4 = Concatenate(axis=-1)([d4, s3])
    d4 = conv_block(d4, 32*scale, train_dec, drate, block_id=7)
        # (W/8, H/8, 512)
    d3 = deconv_block(inputs=d4, ch_out=16*scale, trainit=train_dec, dropout_rate=drate, block_id=3)
    s2 = attention_block(inputs=d3, inputs_skip=x2, ch_mid=8*scale, trainit=train_dec, dropout_rate=drate, block_id=3)
    d3 = Concatenate(axis=-1)([d3, s2])
    d3 = conv_block(d3, 16*scale, train_dec, drate, block_id=8)
        # (W/4, H/4, 256)
    d2 = deconv_block(inputs=d3, ch_out=8*scale, trainit=train_dec, dropout_rate=drate, block_id=2)
    s1 = attention_block(inputs=d2, inputs_skip=x1, ch_mid=4*scale, trainit=train_dec, dropout_rate=drate, block_id=2)
    d2 = Concatenate(axis=-1)([d2, s1])
    d2 = conv_block(d2, 8*scale, train_dec, drate, block_id=9)
        # (W/2, H/2, 128)
    d1 = deconv_block(inputs=d2, ch_out=4*scale, trainit=train_dec, dropout_rate=drate, block_id=1)
    d1 = Conv2D(filters=out_channels, kernel_size=1, strides=(1, 1), padding='valid', name='conv1x1_dec', trainable=train_dec)(d1)
    dec_out = Activation('sigmoid', name='dec_out')(d1)

    att_unet = Model(inputs=image_input, outputs=dec_out, name='attention_unet_resnet50')

    return att_unet




def attention_unet_activations(image_size, out_channels, multiplier, freeze_encoder, freeze_decoder, use_constraints, dropout_rate):
    """
    image_size = (W, H) -- RGB Image
    out_channels = number of output segmentation masks
    multiplier = the scale by which the channels of the network can be increased
    dim1 is the number of values per IR box and dim2...4 are for each tissue parts
    returns a TF Keras model for attention UNet
    """
    drate = dropout_rate
    scale = int(multiplier)
    train_enc = not freeze_encoder
    train_dec = not freeze_decoder

    image_input = Input(shape=(image_size[0], image_size[1], 3), name='input')
    
    # ENCODER Network
    x1 = conv_block(inputs=image_input, ch_out=8*scale, trainit=train_enc, dropout_rate=drate, block_id=1)
    
    x2 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x1)
    x2 = conv_block(x2, 16*scale, train_enc, drate, block_id=2)

    x3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x2)
    x3 = conv_block(x3, 32*scale, train_enc, drate, block_id=3)

    x4 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x3)
    x4 = conv_block(x4, 64*scale, train_enc, drate, block_id=4)

    x5 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x4)
    x5 = conv_block(x5, 128*scale, train_enc, drate, block_id=5)

    # DECODER Network
    d5 = deconv_block(inputs=x5, ch_out=64*scale, trainit=train_dec, dropout_rate=drate, block_id=5)
    s4, act4 = attention_activations(inputs=d5, inputs_skip=x4, ch_mid=32*scale, trainit=train_dec, dropout_rate=drate, block_id=5)
    d5 = Concatenate(axis=-1)([d5, s4])
    d5 = conv_block(d5, 64*scale, train_dec, drate, block_id=6)

    d4 = deconv_block(inputs=d5, ch_out=32*scale, trainit=train_dec, dropout_rate=drate, block_id=4)
    s3, act3 = attention_activations(inputs=d4, inputs_skip=x3, ch_mid=16*scale, trainit=train_dec, dropout_rate=drate, block_id=4)
    d4 = Concatenate(axis=-1)([d4, s3])
    d4 = conv_block(d4, 32*scale, train_dec, drate, block_id=7)

    d3 = deconv_block(inputs=d4, ch_out=16*scale, trainit=train_dec, dropout_rate=drate, block_id=3)
    s2, act2 = attention_activations(inputs=d3, inputs_skip=x2, ch_mid=8*scale, trainit=train_dec, dropout_rate=drate, block_id=3)
    d3 = Concatenate(axis=-1)([d3, s2])
    d3 = conv_block(d3, 16*scale, train_dec, drate, block_id=8)

    d2 = deconv_block(inputs=d3, ch_out=8*scale, trainit=train_dec, dropout_rate=drate, block_id=2)
    s1, act1 = attention_activations(inputs=d2, inputs_skip=x1, ch_mid=4*scale, trainit=train_dec, dropout_rate=drate, block_id=2)
    d2 = Concatenate(axis=-1)([d2, s1])
    d2 = conv_block(d2, 8*scale, train_dec, drate, block_id=9)

    d1 = Conv2D(filters=out_channels, kernel_size=1, strides=(1, 1), padding='valid', name='conv1x1', trainable=train_dec)(d2)
    d1 = Activation('sigmoid', name='out')(d1)

    # Model the Attention-Unet network
    encoder = Model(inputs=image_input, outputs=x5, name='attention_unet_encoder')
    att_unet = Model(inputs=image_input, outputs=d1, name='attention_unet')
    att_acts = Model(inputs=image_input, outputs=[act4, act3, act2, act1], name='attention_activations')
    
    return att_acts



if __name__ == "__main__":

    # model = attention_unet((320, 256), 3, 7, False, 0.1)
    model = attention_unet_resnet50((320, 256, 3), 4, False, False, 0.0)
    model.summary()
    # import tensorflow as tf
    # tf.keras.utils.plot_model(model, show_shapes=True)