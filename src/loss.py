import tensorflow as tf


class CustomLoss(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__()

    def call(self, y_true, y_pred):

        char_pre = y_pred[:, :, :, 0]
        aff_pre = y_pred[:, :, :, 1]
        char_gt = y_true[:, :, :, 0]
        aff_gt = y_true[:, :, :, 1]
        scp = y_true[:, :, :, 2]
        char_sub = char_pre - char_gt
        aff_sub = aff_pre - aff_gt

        char_loss = tf.square(char_sub)
        aff_loss = tf.square(aff_sub)

        return tf.divide(tf.reduce_sum(tf.add(char_loss, aff_loss)), 4)
