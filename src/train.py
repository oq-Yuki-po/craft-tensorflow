import datetime
import pathlib
import shutil

import tensorflow as tf
import typer
from tensorflow.keras import optimizers

from src.callbacks import cb_checkpoint, cb_tensorboard
from src.dataset.generator import generate_dataset
from src.loss import CustomLoss
from src.model import craft
from src.util import load_yaml


def train():

    cfg = load_yaml()

    train_ds = generate_dataset(is_augment=True)

    batch_size = cfg['train_batch_size']

    train_ds = train_ds.shuffle(buffer_size=1000).\
        repeat().\
        batch(batch_size).\
        prefetch(tf.data.AUTOTUNE)

    model = craft()

    optimizer = optimizers.Adam(learning_rate=cfg['train_initial_lr'])

    model.compile(optimizer=optimizer, loss=CustomLoss())

    model.summary()

    steps_per_epoch = cfg['train_data_length'] // batch_size + 1

    result_dir = f"results/{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
    checkpoint_dir = f"{result_dir}/checkpoints/"
    log_dir = f"{result_dir}/logs"

    pathlib.Path(result_dir).mkdir(exist_ok=True)
    pathlib.Path(checkpoint_dir).mkdir(exist_ok=True)
    pathlib.Path(log_dir).mkdir(exist_ok=True)

    model.fit(train_ds,
              epochs=cfg['train_epochs'],
              steps_per_epoch=steps_per_epoch,
              callbacks=[cb_tensorboard(log_dir), cb_checkpoint(checkpoint_dir)])

    shutil.copy("src/config.yml", f"{result_dir}/config.yaml")

    model.save(f'{result_dir}/saved_model')

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    tflite_model = converter.convert()

    open(f"{result_dir}/model.tflite", "wb").write(tflite_model)


if __name__ == "__main__":
    typer.run(train)
