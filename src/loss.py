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
        char_loss = tf.reduce_mean(tf.square(char_pre - char_gt))
        aff_loss = tf.reduce_mean(tf.square(aff_pre - aff_gt))

        char_loss = tf.multiply(scp, char_loss)
        aff_loss = tf.multiply(scp, aff_loss)

        return tf.reduce_mean((tf.add(char_loss, aff_loss)))
