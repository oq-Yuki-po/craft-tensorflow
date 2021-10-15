import tensorflow as tf
import typer
from tensorflow.keras import optimizers

from src.dataset.generator import generate_dataset
from src.model import craft
from src.util import load_yaml


class CustomAccuracy(tf.keras.losses.Loss):
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


def train():

    cfg = load_yaml()

    train_ds = generate_dataset(is_augment=True)

    train_ds = train_ds.shuffle(buffer_size=1000).repeat().\
        batch(cfg['train_batch_size']).\
        prefetch(tf.data.AUTOTUNE)

    model = craft()

    optimizer = optimizers.Adam()

    model.compile(optimizer=optimizer, loss=CustomAccuracy())

    model.fit(train_ds, epochs=3, steps_per_epoch=100)


if __name__ == "__main__":
    typer.run(train)
