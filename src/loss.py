import tensorflow as tf


class CustomLoss(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__()

    def call(self, y_true, y_pred):
        char_pre = tf.reshape(y_pred[:, :, :, 0], [-1], name='char_pre')
        aff_pre = tf.reshape(y_pred[:, :, :, 1], [-1], name='aff_pre')
        char_gt = tf.reshape(y_true[:, :, :, 0], [-1], name='char_gt')
        aff_gt = tf.reshape(y_true[:, :, :, 1], [-1], name='aff_gt')
        char_loss = tf.norm(tf.subtract(char_pre, char_gt))
        aff_loss = tf.norm(tf.subtract(aff_pre, aff_gt))
        return tf.reduce_mean(tf.reduce_sum(tf.add(char_loss, aff_loss)))
