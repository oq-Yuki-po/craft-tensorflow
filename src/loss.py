import tensorflow as tf


class CustomLoss(tf.keras.losses.Loss):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size

    @tf.function
    def single_image_loss(self, pre_loss, loss_label):

        sum_loss = 0
        batch_size = pre_loss.shape[0]
        sum_loss = tf.reduce_mean(tf.reshape(pre_loss, [-1])) * 0
        pre_loss = tf.reshape(pre_loss, [batch_size, -1])
        loss_label = tf.reshape(loss_label, [batch_size, -1])

        for i in range(batch_size):
            average_number = 0
            positive_pixel = len(pre_loss[i][(loss_label[i] >= 0.1)])
            average_number += positive_pixel
            if positive_pixel != 0:
                posi_loss = tf.reduce_mean(pre_loss[i][(loss_label[i] >= 0.1)])
                sum_loss += posi_loss
                if len(pre_loss[i][(loss_label[i] < 0.1)]) < 3 * positive_pixel:
                    nega_loss = tf.reduce_mean(pre_loss[i][(loss_label[i] < 0.1)])
                    average_number += len(pre_loss[i][(loss_label[i] < 0.1)])
                else:
                    nega_loss = tf.reduce_mean(tf.nn.top_k(pre_loss[i][(loss_label[i] < 0.1)], 3 * positive_pixel)[0])
                    average_number += 3 * positive_pixel
                sum_loss += nega_loss
            else:
                nega_loss = tf.reduce_mean(tf.nn.top_k(pre_loss[i], 500)[0])
                average_number += 500
                sum_loss += nega_loss

        return sum_loss

    def call(self, y_true, y_pred):

        region_pre = y_pred[:, :, :, 0]
        aff_pre = y_pred[:, :, :, 1]
        region_gt = y_true[:, :, :, 0]
        aff_gt = y_true[:, :, :, 1]
        scp = y_true[:, :, :, 2]

        region_loss = tf.square(tf.subtract(region_gt, region_pre))
        aff_loss = tf.square(tf.subtract(aff_gt, aff_pre))

        region_loss = tf.matmul(region_loss, scp)
        aff_loss = tf.matmul(aff_loss, scp)

        region_loss = self.single_image_loss(region_loss, region_gt)
        aff_loss = self.single_image_loss(aff_loss, aff_gt)

        return tf.divide(tf.add(region_loss, aff_loss), self.batch_size)
