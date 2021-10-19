import tensorflow as tf
from tensorflow.keras import layers

from src.backbones.mobilenetv3 import MobileNetV3Large
from src.util import load_yaml


def craft():

    cfg = load_yaml()

    _, _, img_chanel = cfg['input_image_size']

    image_input = layers.Input(shape=(None, None, img_chanel))

    backbone = MobileNetV3Large(cfg['mbv3_alpha'])

    C2, C3, C4, C5 = backbone(image_input)

    in2 = layers.Conv2D(64, (1, 1), padding='same', kernel_initializer='he_normal', name='in2')(C2)
    in3 = layers.Conv2D(128, (1, 1), padding='same', kernel_initializer='he_normal', name='in3')(C3)
    in4 = layers.Conv2D(256, (1, 1), padding='same', kernel_initializer='he_normal', name='in4')(C4)
    in5 = layers.Conv2D(256, (1, 1), padding='same', kernel_initializer='he_normal', name='in5')(C5)

    # 1 / 32
    P5 = layers.BatchNormalization()(in5)
    P5 = layers.Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal')(P5)
    P5 = layers.BatchNormalization()(P5)

    # 1 / 16
    out4 = layers.Add()([in4, layers.UpSampling2D(size=(2, 2))(P5)])
    P4 = layers.BatchNormalization()(out4)
    P4 = layers.Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(P4)
    P4 = layers.BatchNormalization()(P4)

    # 1 / 8
    out3 = layers.Add()([in3, layers.UpSampling2D(size=(2, 2))(P4)])
    P3 = layers.BatchNormalization()(out3)
    P3 = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(P3)
    P3 = layers.BatchNormalization()(P3)
    # 1 / 4
    P2 = layers.Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(
        layers.Add()([in2, layers.UpSampling2D(size=(2, 2))(P3)]))
    P2 = layers.BatchNormalization()(P2)
    P2 = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(P2)
    P2 = layers.BatchNormalization()(P2)

    x = layers.UpSampling2D(size=(2, 2))(P2)
    x = layers.Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = layers.Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = layers.Conv2D(16, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = layers.Conv2D(16, (1, 1), padding='same', kernel_initializer='he_normal')(x)
    x = layers.Conv2D(2, (1, 1), padding='same', kernel_initializer='he_normal')(x)

    return tf.keras.Model(inputs=[image_input], outputs=[x])


if __name__ == '__main__':
    model = craft()
    model.summary()
