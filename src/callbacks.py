import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from src.util import loadImage, normalizeMeanVariance, resize_aspect_ratio


def cb_tensorboard(log_dir):
    """TensorBoardのCallback

    Args:
        log_dir ([src]): ログを保存するパス

    Returns:
        [tf.keras.callbacks]: TensorBoardのCallback
    """

    return tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, update_freq=3, profile_batch='1,5')


def cb_epoch_checkpoint(checkpoint_dir):
    """Check pointのCallback

    Args:
        checkpoint_dir (src): check pointを保存するパス

    Returns:
        [tf.keras.callbacks]: check pointのCallback
    """
    checkpoint_path = f"{checkpoint_dir}" + "cp-{epoch:03d}_{loss:.4f}.ckpt"

    return tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                              verbose=1,
                                              save_weights_only=True,
                                              period=1)


class CustomModelCheckpoint(tf.keras.callbacks.Callback):

    def __init__(self, model, checkpoint_dir, all_step=0, save_steps=1000):
        super(CustomModelCheckpoint, self).__init__()
        self.model = model
        self.checkpoint_dir = checkpoint_dir
        self.all_step = all_step
        self.save_steps = save_steps

    def on_train_batch_begin(self, batch, logs=None):
        self.all_step += 1

    def on_batch_end(self, batch, logs=None):
        if self.all_step % self.save_steps == 0:
            file_path = "{}/step_{:06}_loss_{:04f}.ckpt".format(self.checkpoint_dir, self.all_step, logs["loss"])
            self.model.save_weights(filepath=file_path)
            print(f"\nsaved ckpt : {file_path}")


class CustomLearningRateScheduler(tf.keras.callbacks.Callback):

    def __init__(self, all_step=0):
        super(CustomLearningRateScheduler, self).__init__()
        self.all_step = all_step

    def on_train_batch_begin(self, batch, logs=None):
        current_lr = self.model.optimizer.learning_rate.numpy()
        self.all_step += 1

        if self.all_step % 10000 == 0:
            self.model.optimizer.learning_rate.assign(current_lr * 0.8)
            print(f"current lr:{current_lr:.08f} new lr:{current_lr * 0.8:.08f}")


class CheckLearningProcess(tf.keras.callbacks.Callback):

    def __init__(self, image_dir, all_step=0):
        super(CheckLearningProcess, self).__init__()
        self.all_step = all_step
        self.image_dir = image_dir

    def on_train_batch_begin(self, batch, logs=None):

        self.all_step += 1
        if self.all_step % 1 == 0:
            image = loadImage('src/images/sample_01.jpeg')
            org_image_height, org_image_width, _ = image.shape
            image_resized, _, _ = resize_aspect_ratio(image, 1280, cv2.INTER_LINEAR)

            image_resized = normalizeMeanVariance(image_resized)
            inputs = np.expand_dims(image_resized, axis=0)

            results = self.model.predict(inputs)
            region = results[0, :, :, 0]
            affinity = results[0, :, :, 1]
            region = cv2.resize(region, (org_image_width, org_image_height))
            affinity = cv2.resize(affinity, (org_image_width, org_image_height))

            region = (region * 255).astype(np.uint8)
            affinity = (affinity * 255).astype(np.uint8)

            plt.imshow(region)
            plt.savefig(f"{self.image_dir}/region/plt_step_{self.all_step}.jpeg")

            plt.imshow(affinity)
            plt.savefig(f"{self.image_dir}/affinity/plt_step_{self.all_step}.jpeg")

            region_heatmap_img = cv2.applyColorMap(region, cv2.COLORMAP_JET)
            overlay_img = cv2.addWeighted(region_heatmap_img, 0.5, image, 0.5, 0)
            cv2.imwrite(f'{self.image_dir}/region/step_{self.all_step}.jpeg', overlay_img)

            affinity_heatmap_img = cv2.applyColorMap(affinity, cv2.COLORMAP_JET)
            overlay_img = cv2.addWeighted(affinity_heatmap_img, 0.5, image, 0.5, 0)
            cv2.imwrite(f'{self.image_dir}/affinity/step_{self.all_step}.jpeg', overlay_img)
