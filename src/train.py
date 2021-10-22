import datetime
import os
import pathlib
import shutil

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import typer
from tensorflow.keras import optimizers

from src.callbacks import CheckLearningProcess, CustomLearningRateScheduler, cb_checkpoint, cb_tensorboard
from src.dataset.generator import CraftDataset
from src.loss import CustomLoss
from src.model import craft
from src.util import load_yaml


def train():

    cfg = load_yaml()

    result_dir = f"results/{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
    checkpoint_dir = f"{result_dir}/checkpoints/"
    log_dir = f"{result_dir}/logs"
    image_dir = f"{result_dir}/images"

    pathlib.Path(result_dir).mkdir(exist_ok=True)
    pathlib.Path(checkpoint_dir).mkdir(exist_ok=True)
    pathlib.Path(log_dir).mkdir(exist_ok=True)
    pathlib.Path(image_dir).mkdir(exist_ok=True)

    craft_dataset = CraftDataset()

    train_ds = craft_dataset.generate()

    batch_size = cfg['train_batch_size']

    train_ds = train_ds.shuffle(buffer_size=cfg['train_shuffle_buffer_size'], seed=cfg['train_shuffle_seed']).\
        repeat().\
        batch(batch_size).\
        prefetch(tf.data.AUTOTUNE)

    strategy = tf.distribute.MirroredStrategy()
    print(f'Number of devices: {strategy.num_replicas_in_sync}')

    with strategy.scope():

        model = craft()

        optimizer = optimizers.Adam(learning_rate=cfg['train_initial_lr'])

        model.compile(optimizer=optimizer, loss=CustomLoss())

    model.summary()

    steps_per_epoch = cfg['train_data_length'] // batch_size + 1

    shutil.copy("src/config.yml", f"{result_dir}/config.yaml")

    model.fit(train_ds,
              epochs=cfg['train_epochs'],
              steps_per_epoch=steps_per_epoch,
              callbacks=[cb_tensorboard(log_dir),
                         cb_checkpoint(checkpoint_dir),
                         CustomLearningRateScheduler(),
                         CheckLearningProcess(image_dir)])

    model.save(f'{result_dir}/saved_model')

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    tflite_model = converter.convert()

    open(f"{result_dir}/model.tflite", "wb").write(tflite_model)


if __name__ == "__main__":
    typer.run(train)
