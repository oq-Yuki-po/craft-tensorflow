import tensorflow as tf


def cb_tensorboard(log_dir):
    """TensorBoardのCallback

    Args:
        log_dir ([src]): ログを保存するパス

    Returns:
        [tf.keras.callbacks]: TensorBoardのCallback
    """

    return tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, update_freq=3)


def cb_checkpoint(checkpoint_dir):
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


class CustomLearningRateScheduler(tf.keras.callbacks.Callback):

    def __init__(self, all_step=0):
        super(CustomLearningRateScheduler, self).__init__()
        self.all_step = all_step

    def on_train_batch_begin(self, batch, logs=None):
        current_lr = self.model.optimizer.learning_rate.numpy()
        self.all_step += 1

        if self.all_step % 3 == 0:
            self.model.optimizer.learning_rate.assign(current_lr * 0.8)
            print(f"current lr:{current_lr:.08f} new lr:{current_lr * 0.8:.08f}")
