import tensorflow as tf

from src.backbones.mobilenetv3.block import BottleNeck, h_swish


class MobileNetV3Large(tf.keras.Model):
    def __init__(self, alpha=1.0):
        super(MobileNetV3Large, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=16,
                                            kernel_size=(3, 3),
                                            strides=2,
                                            padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bneck1 = BottleNeck(in_size=16, exp_size=16, out_size=16, s=1,
                                 is_se_existing=False, NL="RE", k=3, alpha=alpha)
        self.bneck2 = BottleNeck(in_size=16, exp_size=64, out_size=24, s=2,
                                 is_se_existing=False, NL="RE", k=3, alpha=alpha)
        self.bneck3 = BottleNeck(in_size=24, exp_size=72, out_size=24, s=1,
                                 is_se_existing=False, NL="RE", k=3, alpha=alpha)
        self.bneck4 = BottleNeck(in_size=24, exp_size=72, out_size=40, s=2,
                                 is_se_existing=True, NL="RE", k=5, alpha=alpha)
        self.bneck5 = BottleNeck(in_size=40, exp_size=120, out_size=40, s=1,
                                 is_se_existing=True, NL="RE", k=5, alpha=alpha)
        self.bneck6 = BottleNeck(in_size=40, exp_size=120, out_size=40, s=1,
                                 is_se_existing=True, NL="RE", k=5, alpha=alpha)
        self.bneck7 = BottleNeck(in_size=40, exp_size=240, out_size=80, s=2,
                                 is_se_existing=False, NL="HS", k=3, alpha=alpha)
        self.bneck8 = BottleNeck(in_size=80, exp_size=200, out_size=80, s=1,
                                 is_se_existing=False, NL="HS", k=3, alpha=alpha)
        self.bneck9 = BottleNeck(in_size=80, exp_size=184, out_size=80, s=1,
                                 is_se_existing=False, NL="HS", k=3, alpha=alpha)
        self.bneck10 = BottleNeck(in_size=80, exp_size=184, out_size=80, s=1,
                                  is_se_existing=False, NL="HS", k=3, alpha=alpha)
        self.bneck11 = BottleNeck(in_size=80, exp_size=480, out_size=112, s=1,
                                  is_se_existing=True, NL="HS", k=3, alpha=alpha)
        self.bneck12 = BottleNeck(in_size=112, exp_size=672, out_size=112, s=1,
                                  is_se_existing=True, NL="HS", k=3, alpha=alpha)
        self.bneck13 = BottleNeck(in_size=112, exp_size=672, out_size=160, s=2,
                                  is_se_existing=True, NL="HS", k=5, alpha=alpha)
        self.bneck14 = BottleNeck(in_size=160, exp_size=960, out_size=160, s=1,
                                  is_se_existing=True, NL="HS", k=5, alpha=alpha)
        self.bneck15 = BottleNeck(in_size=160, exp_size=960, out_size=160, s=1,
                                  is_se_existing=True, NL="HS", k=5, alpha=alpha)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = h_swish(x)

        x = self.bneck1(x)
        x = self.bneck2(x)
        c2 = self.bneck3(x)
        x = self.bneck4(c2)
        x = self.bneck5(x)
        c3 = self.bneck6(x)
        x = self.bneck7(c3)
        x = self.bneck8(x)
        x = self.bneck9(x)
        x = self.bneck10(x)
        x = self.bneck11(x)
        c4 = self.bneck12(x)
        x = self.bneck13(c4)
        x = self.bneck14(x)
        c5 = self.bneck15(x)

        return c2, c3, c4, c5

if __name__ == '__main__':
    model = MobileNetV3Large()
    model.build(input_shape=(None, 224, 224, 3))
    model.summary()
