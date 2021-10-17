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

    train_ds = train_ds.shuffle(buffer_size=1000).repeat().\
        batch(cfg['train_batch_size']).\
        prefetch(tf.data.AUTOTUNE)

    model = craft()

    optimizer = optimizers.Adam()

    model.compile(optimizer=optimizer, loss=CustomLoss())

    model.fit(train_ds, epochs=3, steps_per_epoch=100)


if __name__ == "__main__":
    typer.run(train)
