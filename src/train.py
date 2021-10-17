import tensorflow as tf
import typer
from tensorflow.keras import optimizers

from src.dataset.generator import generate_dataset
from src.loss import CustomLoss
from src.model import craft
from src.util import load_yaml


def train():

    cfg = load_yaml()

    train_ds = generate_dataset(is_augment=True)

    batch_size = cfg['train_batch_size']

    train_ds = train_ds.shuffle(buffer_size=1000).repeat().\
        batch(batch_size).\
        prefetch(tf.data.AUTOTUNE)

    model = craft()

    optimizer = optimizers.Adam(learning_rate=cfg['train_initial_lr'])

    model.compile(optimizer=optimizer, loss=CustomLoss())

    steps_per_epoch = cfg['train_data_length'] // batch_size + 1

    model.fit(train_ds,
              epochs=cfg['epochs'],
              steps_per_epoch=steps_per_epoch)


if __name__ == "__main__":
    typer.run(train)
