
import tensorflow as tf


def cb_tensorboard(log_dir):

    return tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, update_freq=3)


def cb_checkpoint(checkpoint_dir):
    checkpoint_path = f"{checkpoint_dir}" + "cp-{epoch:03d}_{loss:.4f}.ckpt"

    return tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                              verbose=1,
                                              save_weights_only=True,
                                              period=1)
